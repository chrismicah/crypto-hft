"""Performance metrics calculation for backtesting engine."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

from .models import Fill, Position, Order, BacktestConfig
from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeMetrics:
    """Metrics for individual trades."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    fees: float
    duration_seconds: Optional[float] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def __post_init__(self):
        if self.exit_time and self.entry_time:
            self.duration_seconds = (self.exit_time - self.entry_time).total_seconds()


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for backtesting results."""
    
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional Value at Risk (95%)
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Time-based metrics
    average_trade_duration_hours: float = 0.0
    average_time_in_market: float = 0.0
    
    # Advanced metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    beta: float = 0.0
    
    # Portfolio metrics
    final_portfolio_value: float = 0.0
    peak_portfolio_value: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    
    # Additional statistics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration_days': self.max_drawdown_duration_days,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'average_trade_duration_hours': self.average_trade_duration_hours,
            'average_time_in_market': self.average_time_in_market,
            'information_ratio': self.information_ratio,
            'final_portfolio_value': self.final_portfolio_value,
            'peak_portfolio_value': self.peak_portfolio_value,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio
        }


class PerformanceCalculator:
    """Calculator for backtesting performance metrics."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize performance calculator.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.returns: List[float] = []
        self.trades: List[TradeMetrics] = []
        self.fills: List[Fill] = []
        self.positions: Dict[str, Position] = {}
        
        # Track portfolio state
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.peak_value = config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_start: Optional[datetime] = None
        self.max_drawdown_duration = timedelta(0)
        
        logger.info("Initialized performance calculator", 
                   initial_capital=config.initial_capital)
    
    def add_fill(self, fill: Fill) -> None:
        """Add a fill to the performance tracking."""
        self.fills.append(fill)
        
        # Update position
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = Position(symbol=fill.symbol)
        
        self.positions[fill.symbol].add_fill(fill)
        
        # Update cash
        if fill.side == OrderSide.BUY:
            self.cash -= fill.quantity * fill.price + fill.fee
        else:
            self.cash += fill.quantity * fill.price - fill.fee
        
        logger.debug("Added fill", 
                    symbol=fill.symbol,
                    side=fill.side.value,
                    quantity=fill.quantity,
                    price=fill.price,
                    fee=fill.fee)
    
    def update_portfolio_value(self, timestamp: datetime, market_prices: Dict[str, float]) -> None:
        """Update portfolio value with current market prices."""
        # Calculate position values
        position_value = 0.0
        
        for symbol, position in self.positions.items():
            if not position.is_flat and symbol in market_prices:
                position.update_unrealized_pnl(market_prices[symbol])
                position_value += position.quantity * market_prices[symbol]
        
        total_value = self.cash + position_value
        self.portfolio_values.append((timestamp, total_value))
        
        # Calculate return
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2][1]
            if prev_value > 0:
                return_pct = (total_value - prev_value) / prev_value
                self.returns.append(return_pct)
        
        # Update drawdown tracking
        if total_value > self.peak_value:
            self.peak_value = total_value
            if self.drawdown_start:
                # End of drawdown period
                drawdown_duration = timestamp - self.drawdown_start
                if drawdown_duration > self.max_drawdown_duration:
                    self.max_drawdown_duration = drawdown_duration
                self.drawdown_start = None
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - total_value) / self.peak_value
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            if not self.drawdown_start:
                self.drawdown_start = timestamp
        
        logger.debug("Updated portfolio value",
                    timestamp=timestamp,
                    total_value=total_value,
                    cash=self.cash,
                    position_value=position_value,
                    drawdown=self.current_drawdown)
    
    def add_trade(self, trade: TradeMetrics) -> None:
        """Add a completed trade to the metrics."""
        self.trades.append(trade)
        logger.debug("Added trade", trade_id=trade.trade_id, pnl=trade.pnl)
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating performance metrics",
                   total_fills=len(self.fills),
                   total_trades=len(self.trades),
                   portfolio_snapshots=len(self.portfolio_values))
        
        metrics = PerformanceMetrics()
        
        if not self.portfolio_values:
            logger.warning("No portfolio values to calculate metrics")
            return metrics
        
        # Basic portfolio metrics
        initial_value = self.portfolio_values[0][1]
        final_value = self.portfolio_values[-1][1]
        
        metrics.final_portfolio_value = final_value
        metrics.peak_portfolio_value = self.peak_value
        metrics.total_return = (final_value - initial_value) / initial_value
        
        # Time-based calculations
        start_time = self.portfolio_values[0][0]
        end_time = self.portfolio_values[-1][0]
        total_days = (end_time - start_time).total_seconds() / (24 * 3600)
        
        if total_days > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (365.25 / total_days) - 1
        
        # Return-based metrics
        if self.returns:
            returns_array = np.array(self.returns)
            
            # Volatility (annualized)
            if len(returns_array) > 1:
                # Assume returns are calculated at regular intervals
                # For simplicity, assume daily returns (adjust based on actual frequency)
                periods_per_year = 365.25
                if total_days > 0:
                    periods_per_year = len(returns_array) * 365.25 / total_days
                
                metrics.volatility = np.std(returns_array) * np.sqrt(periods_per_year)
                
                # Sharpe ratio
                if metrics.volatility > 0:
                    excess_return = metrics.annualized_return - self.config.risk_free_rate
                    metrics.sharpe_ratio = excess_return / metrics.volatility
                
                # Sortino ratio (using downside deviation)
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns) * np.sqrt(periods_per_year)
                    if downside_deviation > 0:
                        metrics.sortino_ratio = excess_return / downside_deviation
                
                # Calmar ratio
                if metrics.max_drawdown > 0:
                    metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
                
                # VaR and CVaR
                metrics.var_95 = np.percentile(returns_array, 5)
                worst_5_percent = returns_array[returns_array <= metrics.var_95]
                if len(worst_5_percent) > 0:
                    metrics.cvar_95 = np.mean(worst_5_percent)
                
                # Higher moments
                metrics.skewness = self._calculate_skewness(returns_array)
                metrics.kurtosis = self._calculate_kurtosis(returns_array)
                
                # Tail ratio
                top_10_percent = np.percentile(returns_array, 90)
                bottom_10_percent = np.percentile(returns_array, 10)
                if bottom_10_percent != 0:
                    metrics.tail_ratio = abs(top_10_percent / bottom_10_percent)
        
        # Drawdown metrics
        metrics.max_drawdown = self.max_drawdown
        metrics.max_drawdown_duration_days = self.max_drawdown_duration.total_seconds() / (24 * 3600)
        
        # Trade statistics
        if self.trades:
            metrics.total_trades = len(self.trades)
            
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            if winning_trades:
                metrics.average_win = np.mean([t.pnl for t in winning_trades])
                metrics.largest_win = max([t.pnl for t in winning_trades])
            
            if losing_trades:
                metrics.average_loss = np.mean([t.pnl for t in losing_trades])
                metrics.largest_loss = min([t.pnl for t in losing_trades])
            
            # Profit factor
            total_wins = sum([t.pnl for t in winning_trades])
            total_losses = abs(sum([t.pnl for t in losing_trades]))
            
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses
            
            # Average trade duration
            completed_trades = [t for t in self.trades if t.duration_seconds is not None]
            if completed_trades:
                avg_duration_seconds = np.mean([t.duration_seconds for t in completed_trades])
                metrics.average_trade_duration_hours = avg_duration_seconds / 3600
        
        # Fee and slippage totals
        metrics.total_fees = sum([f.fee for f in self.fills])
        # Note: Slippage calculation would require comparing to theoretical prices
        
        logger.info("Performance metrics calculated",
                   total_return=f"{metrics.total_return:.2%}",
                   sharpe_ratio=f"{metrics.sharpe_ratio:.2f}",
                   max_drawdown=f"{metrics.max_drawdown:.2%}",
                   win_rate=f"{metrics.win_rate:.2%}")
        
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis
    
    def get_portfolio_timeseries(self) -> pd.DataFrame:
        """Get portfolio value time series as DataFrame."""
        if not self.portfolio_values:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.portfolio_values, columns=['timestamp', 'portfolio_value'])
        df['returns'] = df['portfolio_value'].pct_change()
        df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1
        
        # Calculate rolling drawdown
        df['peak'] = df['portfolio_value'].expanding().max()
        df['drawdown'] = (df['portfolio_value'] - df['peak']) / df['peak']
        
        return df
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """Get trade analysis as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'side': trade.side,
                'pnl': trade.pnl,
                'fees': trade.fees,
                'duration_hours': trade.duration_seconds / 3600 if trade.duration_seconds else None,
                'max_favorable_excursion': trade.max_favorable_excursion,
                'max_adverse_excursion': trade.max_adverse_excursion
            })
        
        return pd.DataFrame(trade_data)
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        metrics = self.calculate_metrics()
        
        report = f"""
BACKTESTING PERFORMANCE REPORT
===============================

Portfolio Performance:
- Initial Capital: ${self.initial_capital:,.2f}
- Final Value: ${metrics.final_portfolio_value:,.2f}
- Total Return: {metrics.total_return:.2%}
- Annualized Return: {metrics.annualized_return:.2%}
- Volatility: {metrics.volatility:.2%}

Risk Metrics:
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Calmar Ratio: {metrics.calmar_ratio:.2f}
- Maximum Drawdown: {metrics.max_drawdown:.2%}
- Max Drawdown Duration: {metrics.max_drawdown_duration_days:.1f} days
- VaR (95%): {metrics.var_95:.2%}
- CVaR (95%): {metrics.cvar_95:.2%}

Trade Statistics:
- Total Trades: {metrics.total_trades}
- Winning Trades: {metrics.winning_trades}
- Losing Trades: {metrics.losing_trades}
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Average Win: ${metrics.average_win:.2f}
- Average Loss: ${metrics.average_loss:.2f}
- Largest Win: ${metrics.largest_win:.2f}
- Largest Loss: ${metrics.largest_loss:.2f}
- Average Trade Duration: {metrics.average_trade_duration_hours:.1f} hours

Costs:
- Total Fees: ${metrics.total_fees:.2f}
- Total Slippage: ${metrics.total_slippage:.2f}

Statistical Properties:
- Skewness: {metrics.skewness:.2f}
- Kurtosis: {metrics.kurtosis:.2f}
- Tail Ratio: {metrics.tail_ratio:.2f}
"""
        
        return report
