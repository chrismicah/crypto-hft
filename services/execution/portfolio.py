"""Portfolio state tracking and real-time PnL calculation for execution service."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import asyncio
from threading import Lock

from common.logger import get_logger

logger = get_logger(__name__)


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Position:
    """Represents a position in a single asset."""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Initialize calculated fields."""
        self.update_unrealized_pnl(self.market_price)
    
    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if abs(self.quantity) < 1e-8:
            return PositionSide.FLAT
        elif self.quantity > 0:
            return PositionSide.LONG
        else:
            return PositionSide.SHORT
    
    @property
    def market_value(self) -> float:
        """Get current market value of position."""
        return self.quantity * self.market_price
    
    @property
    def notional_value(self) -> float:
        """Get absolute notional value."""
        return abs(self.market_value)
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_market_price(self, price: float, timestamp: datetime = None) -> None:
        """Update market price and recalculate unrealized P&L."""
        self.market_price = price
        self.last_updated = timestamp or datetime.utcnow()
        self.update_unrealized_pnl(price)
    
    def update_unrealized_pnl(self, market_price: float) -> None:
        """Update unrealized P&L based on current market price."""
        if abs(self.quantity) > 1e-8 and self.average_price > 0:
            self.unrealized_pnl = (market_price - self.average_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0
    
    def add_trade(self, quantity: float, price: float, timestamp: datetime = None) -> float:
        """
        Add a trade to the position and return realized P&L.
        
        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Trade price
            timestamp: Trade timestamp
            
        Returns:
            Realized P&L from this trade
        """
        self.last_updated = timestamp or datetime.utcnow()
        realized_pnl = 0.0
        
        if abs(self.quantity) < 1e-8:
            # Opening new position
            self.quantity = quantity
            self.average_price = price
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to existing position
            total_cost = self.average_price * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
            if abs(self.quantity) > 1e-8:
                self.average_price = total_cost / abs(self.quantity)
        else:
            # Reducing or closing position
            if abs(quantity) >= abs(self.quantity):
                # Closing and potentially reversing
                # For long position: selling at higher price = profit
                # For short position: buying at lower price = profit
                if self.quantity > 0:
                    # Closing long position
                    realized_pnl = (price - self.average_price) * self.quantity
                else:
                    # Closing short position
                    realized_pnl = (self.average_price - price) * abs(self.quantity)
                
                remaining_quantity = quantity + self.quantity
                if abs(remaining_quantity) > 1e-8:
                    # Reversing position
                    self.quantity = remaining_quantity
                    self.average_price = price
                else:
                    # Flat position
                    self.quantity = 0.0
                    self.average_price = 0.0
            else:
                # Partial close
                realized_pnl = (price - self.average_price) * (-quantity)
                self.quantity += quantity
        
        self.realized_pnl += realized_pnl
        self.update_unrealized_pnl(self.market_price)
        
        return realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'market_price': self.market_price,
            'market_value': self.market_value,
            'notional_value': self.notional_value,
            'side': self.side.value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class DailyPnL:
    """Daily P&L tracking."""
    date: date
    starting_value: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    high_water_mark: float = 0.0
    low_water_mark: float = 0.0
    max_drawdown: float = 0.0
    
    @property
    def total_pnl(self) -> float:
        """Get total daily P&L."""
        return self.realized_pnl + self.unrealized_pnl - self.fees
    
    @property
    def current_value(self) -> float:
        """Get current portfolio value."""
        return self.starting_value + self.total_pnl
    
    def update_drawdown(self, current_value: float) -> None:
        """Update drawdown calculations."""
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
        
        if current_value < self.low_water_mark:
            self.low_water_mark = current_value
        
        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - current_value) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, current_drawdown)


class PortfolioTracker:
    """
    Real-time portfolio state tracking with P&L calculation.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize portfolio tracker.
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.daily_pnl_history: Dict[date, DailyPnL] = {}
        
        # Thread safety
        self._lock = Lock()
        
        # Performance tracking
        self.total_fees = 0.0
        self.trade_count = 0
        self.last_update_time = datetime.utcnow()
        
        # High water mark tracking
        self.all_time_high = initial_capital
        self.all_time_low = initial_capital
        self.max_drawdown = 0.0
        
        logger.info("Portfolio tracker initialized", initial_capital=initial_capital)
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol (thread-safe)."""
        with self._lock:
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol)
            return self.positions[symbol]
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions (thread-safe copy)."""
        with self._lock:
            return {symbol: Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                average_price=pos.average_price,
                market_price=pos.market_price,
                unrealized_pnl=pos.unrealized_pnl,
                realized_pnl=pos.realized_pnl,
                last_updated=pos.last_updated
            ) for symbol, pos in self.positions.items()}
    
    def update_market_prices(self, prices: Dict[str, float], timestamp: datetime = None) -> None:
        """Update market prices for all positions."""
        timestamp = timestamp or datetime.utcnow()
        
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update_market_price(price, timestamp)
            
            self.last_update_time = timestamp
            self._update_portfolio_metrics()
    
    def add_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        timestamp: datetime = None
    ) -> float:
        """
        Add a trade and return realized P&L.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Trade price
            fee: Trading fee
            timestamp: Trade timestamp
            
        Returns:
            Realized P&L from this trade
        """
        timestamp = timestamp or datetime.utcnow()
        
        with self._lock:
            # Get or create position
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol, market_price=price)
            
            position = self.positions[symbol]
            
            # Add trade to position
            realized_pnl = position.add_trade(quantity, price, timestamp)
            
            # Update portfolio metrics
            self.total_fees += fee
            self.trade_count += 1
            self.last_update_time = timestamp
            
            # Update daily P&L
            today = timestamp.date()
            if today not in self.daily_pnl_history:
                yesterday_value = self.get_portfolio_value(exclude_today=True)
                self.daily_pnl_history[today] = DailyPnL(
                    date=today,
                    starting_value=yesterday_value,
                    high_water_mark=yesterday_value,
                    low_water_mark=yesterday_value
                )
            
            daily_pnl = self.daily_pnl_history[today]
            daily_pnl.realized_pnl += realized_pnl
            daily_pnl.fees += fee
            
            self._update_portfolio_metrics()
            
            logger.debug("Trade added",
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        fee=fee,
                        realized_pnl=realized_pnl)
            
            return realized_pnl
    
    def get_portfolio_value(self, exclude_today: bool = False) -> float:
        """Get current portfolio value."""
        with self._lock:
            cash = self.initial_capital
            
            # Add all realized P&L
            for position in self.positions.values():
                cash += position.realized_pnl
            
            # Add current market value of positions
            for position in self.positions.values():
                cash += position.unrealized_pnl
            
            # Subtract fees
            cash -= self.total_fees
            
            # Exclude today's P&L if requested
            if exclude_today:
                today = datetime.utcnow().date()
                if today in self.daily_pnl_history:
                    daily_pnl = self.daily_pnl_history[today]
                    cash -= daily_pnl.total_pnl
            
            return cash
    
    def get_total_notional(self) -> float:
        """Get total notional value of all positions."""
        with self._lock:
            return sum(pos.notional_value for pos in self.positions.values())
    
    def get_net_exposure(self) -> float:
        """Get net exposure (sum of all position values)."""
        with self._lock:
            return sum(pos.market_value for pos in self.positions.values())
    
    def get_gross_exposure(self) -> float:
        """Get gross exposure (sum of absolute position values)."""
        with self._lock:
            return sum(abs(pos.market_value) for pos in self.positions.values())
    
    def get_daily_pnl(self, target_date: date = None) -> Optional[DailyPnL]:
        """Get daily P&L for a specific date."""
        target_date = target_date or datetime.utcnow().date()
        
        with self._lock:
            if target_date in self.daily_pnl_history:
                daily_pnl = self.daily_pnl_history[target_date]
                
                # Update unrealized P&L for today
                if target_date == datetime.utcnow().date():
                    daily_pnl.unrealized_pnl = sum(
                        pos.unrealized_pnl for pos in self.positions.values()
                    )
                    daily_pnl.update_drawdown(daily_pnl.current_value)
                
                return daily_pnl
            
            return None
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown from all-time high."""
        current_value = self.get_portfolio_value()
        
        if self.all_time_high > 0:
            return (self.all_time_high - current_value) / self.all_time_high
        
        return 0.0
    
    def get_daily_drawdown(self, target_date: date = None) -> float:
        """Get drawdown for a specific day."""
        daily_pnl = self.get_daily_pnl(target_date)
        
        if daily_pnl:
            return daily_pnl.max_drawdown
        
        return 0.0
    
    def get_position_concentration(self, symbol: str) -> float:
        """Get position concentration as percentage of portfolio."""
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value <= 0:
            return 0.0
        
        position = self.get_position(symbol)
        return position.notional_value / portfolio_value
    
    def get_largest_position_concentration(self) -> Tuple[str, float]:
        """Get the largest position concentration."""
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value <= 0:
            return "", 0.0
        
        max_concentration = 0.0
        max_symbol = ""
        
        with self._lock:
            for symbol, position in self.positions.items():
                concentration = position.notional_value / portfolio_value
                if concentration > max_concentration:
                    max_concentration = concentration
                    max_symbol = symbol
        
        return max_symbol, max_concentration
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics (called with lock held)."""
        current_value = self.get_portfolio_value()
        
        # Update all-time high/low
        if current_value > self.all_time_high:
            self.all_time_high = current_value
        
        if current_value < self.all_time_low:
            self.all_time_low = current_value
        
        # Update max drawdown
        if self.all_time_high > 0:
            current_drawdown = (self.all_time_high - current_value) / self.all_time_high
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update today's daily P&L
        today = datetime.utcnow().date()
        if today in self.daily_pnl_history:
            daily_pnl = self.daily_pnl_history[today]
            daily_pnl.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            daily_pnl.update_drawdown(daily_pnl.current_value)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            current_value = self.get_portfolio_value()
            total_return = (current_value - self.initial_capital) / self.initial_capital
            
            # Calculate daily returns
            daily_returns = []
            sorted_dates = sorted(self.daily_pnl_history.keys())
            
            for i, date in enumerate(sorted_dates):
                daily_pnl = self.daily_pnl_history[date]
                if daily_pnl.starting_value > 0:
                    daily_return = daily_pnl.total_pnl / daily_pnl.starting_value
                    daily_returns.append(daily_return)
            
            # Calculate statistics
            if daily_returns:
                avg_daily_return = np.mean(daily_returns)
                daily_volatility = np.std(daily_returns)
                sharpe_ratio = avg_daily_return / daily_volatility if daily_volatility > 0 else 0
                
                # Annualize (assuming 252 trading days)
                annualized_return = avg_daily_return * 252
                annualized_volatility = daily_volatility * np.sqrt(252)
                annualized_sharpe = sharpe_ratio * np.sqrt(252)
            else:
                avg_daily_return = annualized_return = 0
                daily_volatility = annualized_volatility = 0
                sharpe_ratio = annualized_sharpe = 0
            
            return {
                'portfolio_value': current_value,
                'initial_capital': self.initial_capital,
                'total_return': total_return,
                'total_pnl': current_value - self.initial_capital,
                'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values()),
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                'total_fees': self.total_fees,
                'net_pnl': current_value - self.initial_capital - self.total_fees,
                'all_time_high': self.all_time_high,
                'all_time_low': self.all_time_low,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.get_current_drawdown(),
                'gross_exposure': self.get_gross_exposure(),
                'net_exposure': self.get_net_exposure(),
                'total_notional': self.get_total_notional(),
                'trade_count': self.trade_count,
                'avg_daily_return': avg_daily_return,
                'daily_volatility': daily_volatility,
                'sharpe_ratio': sharpe_ratio,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'annualized_sharpe': annualized_sharpe,
                'active_positions': len([p for p in self.positions.values() if p.side != PositionSide.FLAT]),
                'last_update': self.last_update_time.isoformat()
            }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for logging/monitoring."""
        metrics = self.get_performance_metrics()
        
        return {
            'portfolio_value': f"${metrics['portfolio_value']:,.2f}",
            'total_return': f"{metrics['total_return']:.2%}",
            'current_drawdown': f"{metrics['current_drawdown']:.2%}",
            'max_drawdown': f"{metrics['max_drawdown']:.2%}",
            'active_positions': metrics['active_positions'],
            'gross_exposure': f"${metrics['gross_exposure']:,.2f}",
            'trade_count': metrics['trade_count']
        }
    
    def reset_daily_pnl(self, target_date: date = None) -> None:
        """Reset daily P&L for a new trading day."""
        target_date = target_date or datetime.utcnow().date()
        
        with self._lock:
            current_value = self.get_portfolio_value()
            
            self.daily_pnl_history[target_date] = DailyPnL(
                date=target_date,
                starting_value=current_value,
                high_water_mark=current_value,
                low_water_mark=current_value
            )
            
            logger.info("Daily P&L reset",
                       date=target_date,
                       starting_value=current_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio state to dictionary."""
        with self._lock:
            return {
                'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
                'performance_metrics': self.get_performance_metrics(),
                'daily_pnl': {
                    str(date): {
                        'date': str(pnl.date),
                        'starting_value': pnl.starting_value,
                        'total_pnl': pnl.total_pnl,
                        'realized_pnl': pnl.realized_pnl,
                        'unrealized_pnl': pnl.unrealized_pnl,
                        'fees': pnl.fees,
                        'max_drawdown': pnl.max_drawdown,
                        'current_value': pnl.current_value
                    }
                    for date, pnl in self.daily_pnl_history.items()
                }
            }
