"""Advanced risk management layer with pre-trade checks and Kelly Criterion sizing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import math

from .portfolio import PortfolioTracker, Position, PositionSide
from common.logger import get_logger

logger = get_logger(__name__)


class RiskCheckResult(Enum):
    """Risk check result enumeration."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    HALTED = "HALTED"


class RiskViolationType(Enum):
    """Types of risk violations."""
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN_EXCEEDED = "MAX_DRAWDOWN_EXCEEDED"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"
    TRADING_HALTED = "TRADING_HALTED"
    INVALID_ORDER = "INVALID_ORDER"


@dataclass
class RiskCheckResponse:
    """Response from risk check."""
    result: RiskCheckResult
    original_quantity: float
    approved_quantity: float
    violation_type: Optional[RiskViolationType] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_approved(self) -> bool:
        """Check if order is approved."""
        return self.result == RiskCheckResult.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.result == RiskCheckResult.REJECTED
    
    @property
    def is_modified(self) -> bool:
        """Check if order is modified."""
        return self.result == RiskCheckResult.MODIFIED
    
    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self.result == RiskCheckResult.HALTED


@dataclass
class RiskParameters:
    """Risk management parameters."""
    # Position sizing
    max_position_size_pct: float = 0.1  # 10% of portfolio per position
    max_total_exposure_pct: float = 1.0  # 100% total exposure
    max_concentration_pct: float = 0.2   # 20% max concentration per asset
    
    # Kelly Criterion parameters
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (conservative)
    min_win_rate: float = 0.4     # Minimum win rate for Kelly calculation
    min_trades_for_kelly: int = 20 # Minimum trades before using Kelly
    
    # Drawdown limits
    max_daily_loss_pct: float = 0.05    # 5% daily loss limit
    max_drawdown_pct: float = 0.15      # 15% max drawdown limit
    
    # Capital requirements
    min_capital_buffer: float = 1000.0  # Minimum cash buffer
    margin_requirement: float = 0.1     # 10% margin requirement
    
    # Trading halts
    enable_trading_halts: bool = True
    halt_on_daily_loss: bool = True
    halt_on_max_drawdown: bool = True
    
    def __post_init__(self):
        """Validate parameters."""
        if not (0 < self.max_position_size_pct <= 1.0):
            raise ValueError("max_position_size_pct must be between 0 and 1")
        
        if not (0 < self.kelly_fraction <= 1.0):
            raise ValueError("kelly_fraction must be between 0 and 1")
        
        if not (0 < self.min_win_rate < 1.0):
            raise ValueError("min_win_rate must be between 0 and 1")


class KellyCriterionCalculator:
    """Kelly Criterion position sizing calculator."""
    
    def __init__(self, lookback_period: int = 100):
        """
        Initialize Kelly calculator.
        
        Args:
            lookback_period: Number of recent trades to consider
        """
        self.lookback_period = lookback_period
        self.trade_history: List[Dict[str, Any]] = []
    
    def add_trade_result(
        self,
        symbol: str,
        pnl: float,
        quantity: float,
        entry_price: float,
        exit_price: float,
        timestamp: datetime = None
    ) -> None:
        """Add a completed trade result."""
        trade = {
            'symbol': symbol,
            'pnl': pnl,
            'quantity': abs(quantity),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': pnl / (abs(quantity) * entry_price) if entry_price > 0 else 0,
            'timestamp': timestamp or datetime.utcnow()
        }
        
        self.trade_history.append(trade)
        
        # Keep only recent trades
        if len(self.trade_history) > self.lookback_period * 2:
            self.trade_history = self.trade_history[-self.lookback_period:]
    
    def calculate_kelly_fraction(
        self,
        symbol: str = None,
        min_trades: int = 20
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate Kelly fraction for position sizing.
        
        Args:
            symbol: Specific symbol to calculate for (None for all trades)
            min_trades: Minimum number of trades required
            
        Returns:
            Tuple of (kelly_fraction, statistics)
        """
        # Filter trades
        if symbol:
            trades = [t for t in self.trade_history if t['symbol'] == symbol]
        else:
            trades = self.trade_history
        
        if len(trades) < min_trades:
            return 0.0, {'reason': 'insufficient_trades', 'trade_count': len(trades)}
        
        # Use recent trades
        recent_trades = trades[-self.lookback_period:]
        returns = [t['return_pct'] for t in recent_trades]
        
        if not returns:
            return 0.0, {'reason': 'no_returns'}
        
        # Calculate statistics
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return 0.0, {'reason': 'no_wins_or_losses'}
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))  # Make positive
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received on the wager (avg_win/avg_loss)
        #       p = probability of winning (win_rate)
        #       q = probability of losing (1 - win_rate)
        
        if avg_loss == 0:
            return 0.0, {'reason': 'zero_average_loss'}
        
        b = avg_win / avg_loss  # Odds ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Ensure non-negative
        kelly_fraction = max(0.0, kelly_fraction)
        
        statistics = {
            'trade_count': len(recent_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'odds_ratio': b,
            'kelly_fraction': kelly_fraction,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
        
        return kelly_fraction, statistics
    
    def get_position_size(
        self,
        symbol: str,
        portfolio_value: float,
        kelly_fraction_multiplier: float = 0.25,
        max_position_pct: float = 0.1
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Get recommended position size using Kelly Criterion.
        
        Args:
            symbol: Trading symbol
            portfolio_value: Current portfolio value
            kelly_fraction_multiplier: Fraction of Kelly to use
            max_position_pct: Maximum position size as percentage of portfolio
            
        Returns:
            Tuple of (position_size_usd, statistics)
        """
        kelly_fraction, stats = self.calculate_kelly_fraction(symbol)
        
        if kelly_fraction <= 0:
            return 0.0, stats
        
        # Apply conservative multiplier
        adjusted_kelly = kelly_fraction * kelly_fraction_multiplier
        
        # Calculate position size
        kelly_position_size = portfolio_value * adjusted_kelly
        max_position_size = portfolio_value * max_position_pct
        
        # Use the smaller of Kelly size or max position size
        recommended_size = min(kelly_position_size, max_position_size)
        
        stats.update({
            'adjusted_kelly_fraction': adjusted_kelly,
            'kelly_position_size': kelly_position_size,
            'max_position_size': max_position_size,
            'recommended_size': recommended_size,
            'size_limited_by': 'kelly' if kelly_position_size <= max_position_size else 'max_position'
        })
        
        return recommended_size, stats


class RiskManager:
    """Advanced risk management layer with pre-trade checks."""
    
    def __init__(
        self,
        portfolio_tracker: PortfolioTracker,
        risk_parameters: RiskParameters,
        kelly_calculator: Optional[KellyCriterionCalculator] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            portfolio_tracker: Portfolio state tracker
            risk_parameters: Risk management parameters
            kelly_calculator: Kelly Criterion calculator
        """
        self.portfolio = portfolio_tracker
        self.params = risk_parameters
        self.kelly_calc = kelly_calculator or KellyCriterionCalculator()
        
        # Risk state
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp: Optional[datetime] = None
        
        # Daily reset tracking
        self.last_daily_reset = datetime.utcnow().date()
        
        logger.info("Risk manager initialized",
                   max_position_size_pct=risk_parameters.max_position_size_pct,
                   max_daily_loss_pct=risk_parameters.max_daily_loss_pct,
                   max_drawdown_pct=risk_parameters.max_drawdown_pct,
                   use_kelly_sizing=risk_parameters.use_kelly_sizing)
    
    def check_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET"
    ) -> RiskCheckResponse:
        """
        Perform comprehensive pre-trade risk checks.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity (positive for buy, negative for sell)
            price: Order price
            order_type: Order type
            
        Returns:
            RiskCheckResponse with approval/rejection decision
        """
        # Check if trading is halted
        if self.trading_halted:
            return RiskCheckResponse(
                result=RiskCheckResult.HALTED,
                original_quantity=quantity,
                approved_quantity=0.0,
                violation_type=RiskViolationType.TRADING_HALTED,
                reason=f"Trading halted: {self.halt_reason}",
                metadata={'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None}
            )
        
        # Validate order
        if abs(quantity) < 1e-8 or price <= 0:
            return RiskCheckResponse(
                result=RiskCheckResult.REJECTED,
                original_quantity=quantity,
                approved_quantity=0.0,
                violation_type=RiskViolationType.INVALID_ORDER,
                reason="Invalid order parameters"
            )
        
        # Check daily reset
        self._check_daily_reset()
        
        # Check daily loss limit
        daily_check = self._check_daily_loss_limit()
        if daily_check.is_halted:
            return daily_check
        
        # Check max drawdown
        drawdown_check = self._check_max_drawdown()
        if drawdown_check.is_halted:
            return drawdown_check
        
        # Check position size limits
        size_check = self._check_position_size(symbol, quantity, price)
        if size_check.is_rejected or size_check.is_halted:
            return size_check
        
        # Use approved quantity from size check
        approved_quantity = size_check.approved_quantity
        
        # Check concentration risk
        concentration_check = self._check_concentration_risk(symbol, approved_quantity, price)
        if concentration_check.is_rejected:
            return concentration_check
        
        # Check capital requirements
        capital_check = self._check_capital_requirements(symbol, approved_quantity, price)
        if capital_check.is_rejected:
            return capital_check
        
        # Final approval
        result = RiskCheckResult.MODIFIED if approved_quantity != quantity else RiskCheckResult.APPROVED
        
        return RiskCheckResponse(
            result=result,
            original_quantity=quantity,
            approved_quantity=approved_quantity,
            reason="Order approved" if result == RiskCheckResult.APPROVED else "Order size modified",
            metadata={
                'portfolio_value': self.portfolio.get_portfolio_value(),
                'position_size_pct': abs(approved_quantity * price) / self.portfolio.get_portfolio_value(),
                'current_drawdown': self.portfolio.get_current_drawdown()
            }
        )
    
    def _check_daily_loss_limit(self) -> RiskCheckResponse:
        """Check daily loss limit."""
        if not self.params.halt_on_daily_loss:
            return RiskCheckResponse(RiskCheckResult.APPROVED, 0, 0)
        
        daily_pnl = self.portfolio.get_daily_pnl()
        
        if daily_pnl:
            daily_loss_pct = -daily_pnl.total_pnl / daily_pnl.starting_value if daily_pnl.starting_value > 0 else 0
            
            if daily_loss_pct > self.params.max_daily_loss_pct:
                self._halt_trading(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                
                return RiskCheckResponse(
                    result=RiskCheckResult.HALTED,
                    original_quantity=0,
                    approved_quantity=0,
                    violation_type=RiskViolationType.DAILY_LOSS_LIMIT,
                    reason=f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.params.max_daily_loss_pct:.2%}",
                    metadata={'daily_loss_pct': daily_loss_pct, 'daily_pnl': daily_pnl.total_pnl}
                )
        
        return RiskCheckResponse(RiskCheckResult.APPROVED, 0, 0)
    
    def _check_max_drawdown(self) -> RiskCheckResponse:
        """Check maximum drawdown limit."""
        if not self.params.halt_on_max_drawdown:
            return RiskCheckResponse(RiskCheckResult.APPROVED, 0, 0)
        
        current_drawdown = self.portfolio.get_current_drawdown()
        
        if current_drawdown > self.params.max_drawdown_pct:
            self._halt_trading(f"Max drawdown exceeded: {current_drawdown:.2%}")
            
            return RiskCheckResponse(
                result=RiskCheckResult.HALTED,
                original_quantity=0,
                approved_quantity=0,
                violation_type=RiskViolationType.MAX_DRAWDOWN_EXCEEDED,
                reason=f"Drawdown {current_drawdown:.2%} exceeds limit {self.params.max_drawdown_pct:.2%}",
                metadata={'current_drawdown': current_drawdown}
            )
        
        return RiskCheckResponse(RiskCheckResult.APPROVED, 0, 0)
    
    def _check_position_size(self, symbol: str, quantity: float, price: float) -> RiskCheckResponse:
        """Check position size limits with Kelly Criterion."""
        portfolio_value = self.portfolio.get_portfolio_value()
        
        if portfolio_value <= 0:
            return RiskCheckResponse(
                result=RiskCheckResult.REJECTED,
                original_quantity=quantity,
                approved_quantity=0,
                violation_type=RiskViolationType.INSUFFICIENT_CAPITAL,
                reason="Insufficient portfolio value"
            )
        
        # Calculate order value
        order_value = abs(quantity * price)
        
        # Get current position
        current_position = self.portfolio.get_position(symbol)
        
        # Calculate new position size after this order
        new_quantity = current_position.quantity + quantity
        new_position_value = abs(new_quantity * price)
        
        # Check against maximum position size
        max_position_value = portfolio_value * self.params.max_position_size_pct
        
        if self.params.use_kelly_sizing:
            # Use Kelly Criterion for sizing
            kelly_size, kelly_stats = self.kelly_calc.get_position_size(
                symbol=symbol,
                portfolio_value=portfolio_value,
                kelly_fraction_multiplier=self.params.kelly_fraction,
                max_position_pct=self.params.max_position_size_pct
            )
            
            # Use Kelly size if available and smaller than max
            if kelly_size > 0 and kelly_stats.get('trade_count', 0) >= self.params.min_trades_for_kelly:
                max_position_value = min(max_position_value, kelly_size)
        
        if new_position_value > max_position_value:
            # Calculate maximum allowed quantity
            if abs(current_position.quantity * price) >= max_position_value:
                # Already at or above limit - reject order
                approved_quantity = 0.0
            else:
                # Reduce order size to fit within limit
                remaining_capacity = max_position_value - abs(current_position.quantity * price)
                max_additional_quantity = remaining_capacity / price
                
                if quantity > 0:
                    approved_quantity = min(quantity, max_additional_quantity)
                else:
                    approved_quantity = max(quantity, -max_additional_quantity)
            
            if abs(approved_quantity) < 1e-8:
                return RiskCheckResponse(
                    result=RiskCheckResult.REJECTED,
                    original_quantity=quantity,
                    approved_quantity=0,
                    violation_type=RiskViolationType.POSITION_SIZE_EXCEEDED,
                    reason=f"Position size would exceed limit: {new_position_value:.2f} > {max_position_value:.2f}",
                    metadata={
                        'max_position_value': max_position_value,
                        'new_position_value': new_position_value,
                        'current_position_value': abs(current_position.quantity * price)
                    }
                )
            else:
                return RiskCheckResponse(
                    result=RiskCheckResult.MODIFIED,
                    original_quantity=quantity,
                    approved_quantity=approved_quantity,
                    reason=f"Order size reduced to fit position limit",
                    metadata={
                        'max_position_value': max_position_value,
                        'approved_position_value': abs((current_position.quantity + approved_quantity) * price)
                    }
                )
        
        return RiskCheckResponse(
            result=RiskCheckResult.APPROVED,
            original_quantity=quantity,
            approved_quantity=quantity
        )
    
    def _check_concentration_risk(self, symbol: str, quantity: float, price: float) -> RiskCheckResponse:
        """Check concentration risk limits."""
        portfolio_value = self.portfolio.get_portfolio_value()
        
        if portfolio_value <= 0:
            return RiskCheckResponse(RiskCheckResult.APPROVED, quantity, quantity)
        
        # Calculate new position concentration
        current_position = self.portfolio.get_position(symbol)
        new_quantity = current_position.quantity + quantity
        new_concentration = abs(new_quantity * price) / portfolio_value
        
        if new_concentration > self.params.max_concentration_pct:
            return RiskCheckResponse(
                result=RiskCheckResult.REJECTED,
                original_quantity=quantity,
                approved_quantity=0,
                violation_type=RiskViolationType.CONCENTRATION_RISK,
                reason=f"Concentration risk: {new_concentration:.2%} > {self.params.max_concentration_pct:.2%}",
                metadata={'new_concentration': new_concentration}
            )
        
        return RiskCheckResponse(RiskCheckResult.APPROVED, quantity, quantity)
    
    def _check_capital_requirements(self, symbol: str, quantity: float, price: float) -> RiskCheckResponse:
        """Check capital and margin requirements."""
        portfolio_value = self.portfolio.get_portfolio_value()
        order_value = abs(quantity * price)
        
        # Check minimum capital buffer
        required_capital = order_value * self.params.margin_requirement + self.params.min_capital_buffer
        
        if portfolio_value < required_capital:
            return RiskCheckResponse(
                result=RiskCheckResult.REJECTED,
                original_quantity=quantity,
                approved_quantity=0,
                violation_type=RiskViolationType.INSUFFICIENT_CAPITAL,
                reason=f"Insufficient capital: need {required_capital:.2f}, have {portfolio_value:.2f}",
                metadata={
                    'required_capital': required_capital,
                    'available_capital': portfolio_value,
                    'order_value': order_value
                }
            )
        
        return RiskCheckResponse(RiskCheckResult.APPROVED, quantity, quantity)
    
    def _halt_trading(self, reason: str) -> None:
        """Halt trading with specified reason."""
        if not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = reason
            self.halt_timestamp = datetime.utcnow()
            
            logger.critical("Trading halted by risk manager", reason=reason)
    
    def resume_trading(self, reason: str = "Manual resume") -> bool:
        """Resume trading if conditions allow."""
        if not self.trading_halted:
            return True
        
        # Check if conditions still warrant halt
        daily_check = self._check_daily_loss_limit()
        drawdown_check = self._check_max_drawdown()
        
        if daily_check.is_halted or drawdown_check.is_halted:
            logger.warning("Cannot resume trading - risk conditions still violated")
            return False
        
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp = None
        
        logger.info("Trading resumed", reason=reason)
        return True
    
    def _check_daily_reset(self) -> None:
        """Check if daily reset is needed."""
        today = datetime.utcnow().date()
        
        if today > self.last_daily_reset:
            self.portfolio.reset_daily_pnl(today)
            self.last_daily_reset = today
            
            # Auto-resume trading on new day if halted for daily loss
            if self.trading_halted and "Daily loss" in self.halt_reason:
                self.resume_trading("New trading day")
    
    def add_trade_result(
        self,
        symbol: str,
        pnl: float,
        quantity: float,
        entry_price: float,
        exit_price: float
    ) -> None:
        """Add trade result for Kelly Criterion calculation."""
        self.kelly_calc.add_trade_result(symbol, pnl, quantity, entry_price, exit_price)
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk management status."""
        portfolio_value = self.portfolio.get_portfolio_value()
        daily_pnl = self.portfolio.get_daily_pnl()
        
        status = {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'portfolio_value': portfolio_value,
            'current_drawdown': self.portfolio.get_current_drawdown(),
            'max_drawdown_limit': self.params.max_drawdown_pct,
            'daily_loss_limit': self.params.max_daily_loss_pct,
            'max_position_size_pct': self.params.max_position_size_pct,
            'use_kelly_sizing': self.params.use_kelly_sizing,
            'kelly_fraction': self.params.kelly_fraction
        }
        
        if daily_pnl:
            daily_loss_pct = -daily_pnl.total_pnl / daily_pnl.starting_value if daily_pnl.starting_value > 0 else 0
            status.update({
                'daily_pnl': daily_pnl.total_pnl,
                'daily_loss_pct': daily_loss_pct,
                'daily_loss_remaining': max(0, self.params.max_daily_loss_pct - daily_loss_pct)
            })
        
        return status
    
    def get_position_sizing_recommendation(self, symbol: str, target_allocation_pct: float = None) -> Dict[str, Any]:
        """Get position sizing recommendation for a symbol."""
        portfolio_value = self.portfolio.get_portfolio_value()
        
        if portfolio_value <= 0:
            return {'recommended_size': 0, 'reason': 'insufficient_capital'}
        
        # Use Kelly Criterion if enabled
        if self.params.use_kelly_sizing:
            kelly_size, kelly_stats = self.kelly_calc.get_position_size(
                symbol=symbol,
                portfolio_value=portfolio_value,
                kelly_fraction_multiplier=self.params.kelly_fraction,
                max_position_pct=self.params.max_position_size_pct
            )
            
            if kelly_stats.get('trade_count', 0) >= self.params.min_trades_for_kelly:
                return {
                    'recommended_size': kelly_size,
                    'method': 'kelly_criterion',
                    'statistics': kelly_stats
                }
        
        # Fallback to percentage-based sizing
        if target_allocation_pct:
            target_size = portfolio_value * min(target_allocation_pct, self.params.max_position_size_pct)
        else:
            target_size = portfolio_value * self.params.max_position_size_pct
        
        return {
            'recommended_size': target_size,
            'method': 'percentage_based',
            'allocation_pct': target_size / portfolio_value
        }
