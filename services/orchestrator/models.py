"""
Data models for the Evolutionary Operation (EVOP) Framework.

This module defines the core data structures used for managing champion-challenger
strategy execution, performance tracking, and evolutionary optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid
from decimal import Decimal


class StrategyStatus(Enum):
    """Status of a strategy instance."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"
    PROMOTING = "promoting"
    DEMOTING = "demoting"


class PromotionReason(Enum):
    """Reason for strategy promotion."""
    SUPERIOR_SHARPE = "superior_sharpe"
    SUPERIOR_CALMAR = "superior_calmar"
    LOWER_DRAWDOWN = "lower_drawdown"
    HIGHER_RETURNS = "higher_returns"
    MANUAL_OVERRIDE = "manual_override"
    CHAMPION_FAILURE = "champion_failure"


@dataclass
class StrategyParameters:
    """Configuration parameters for a trading strategy."""
    
    # Core strategy parameters
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    stop_loss_z_score: float = 4.0
    
    # Risk management parameters
    max_position_size: float = 10000.0
    max_drawdown_percent: float = 10.0
    max_daily_loss: float = 5000.0
    kelly_fraction: float = 0.25
    
    # Model parameters
    kalman_process_noise: float = 1e-5
    kalman_observation_noise: float = 1e-3
    garch_window_size: int = 500
    bocd_hazard_rate: float = 0.004
    
    # Execution parameters
    order_timeout_seconds: int = 30
    max_slippage_bps: int = 10
    min_order_size: float = 100.0
    
    # Custom parameters for experimentation
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entry_z_score': self.entry_z_score,
            'exit_z_score': self.exit_z_score,
            'stop_loss_z_score': self.stop_loss_z_score,
            'max_position_size': self.max_position_size,
            'max_drawdown_percent': self.max_drawdown_percent,
            'max_daily_loss': self.max_daily_loss,
            'kelly_fraction': self.kelly_fraction,
            'kalman_process_noise': self.kalman_process_noise,
            'kalman_observation_noise': self.kalman_observation_noise,
            'garch_window_size': self.garch_window_size,
            'bocd_hazard_rate': self.bocd_hazard_rate,
            'order_timeout_seconds': self.order_timeout_seconds,
            'max_slippage_bps': self.max_slippage_bps,
            'min_order_size': self.min_order_size,
            'custom_params': self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyParameters':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy instance."""
    
    # Basic metrics
    total_pnl: Decimal = Decimal('0.0')
    realized_pnl: Decimal = Decimal('0.0')
    unrealized_pnl: Decimal = Decimal('0.0')
    total_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: Optional[float] = None
    
    # Execution metrics
    avg_fill_time: float = 0.0
    avg_slippage_bps: float = 0.0
    total_fees: Decimal = Decimal('0.0')
    
    # Period metrics
    daily_returns: List[float] = field(default_factory=list)
    monthly_returns: List[float] = field(default_factory=list)
    
    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_trade_metrics(self):
        """Update derived trade metrics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.losing_trades > 0 and self.avg_loss != 0:
            self.profit_factor = (self.winning_trades * self.avg_win) / (self.losing_trades * abs(self.avg_loss))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_pnl': float(self.total_pnl),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_fill_time': self.avg_fill_time,
            'avg_slippage_bps': self.avg_slippage_bps,
            'total_fees': float(self.total_fees),
            'daily_returns': self.daily_returns,
            'monthly_returns': self.monthly_returns,
            'start_time': self.start_time.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class StrategyInstance:
    """Represents a single strategy instance (champion or challenger)."""
    
    # Identity
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Strategy"
    is_champion: bool = False
    
    # Configuration
    parameters: StrategyParameters = field(default_factory=StrategyParameters)
    
    # Status and control
    status: StrategyStatus = StrategyStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    
    # Performance tracking
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Execution tracking
    allocated_capital: Decimal = Decimal('100000.0')  # Default $100k
    current_positions: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def get_runtime(self) -> Optional[timedelta]:
        """Get the total runtime of the strategy."""
        if self.started_at is None:
            return None
        
        end_time = self.stopped_at or datetime.now()
        return end_time - self.started_at
    
    def is_active(self) -> bool:
        """Check if the strategy is actively running."""
        return self.status in [StrategyStatus.RUNNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'instance_id': self.instance_id,
            'name': self.name,
            'is_champion': self.is_champion,
            'parameters': self.parameters.to_dict(),
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'performance': self.performance.to_dict(),
            'allocated_capital': float(self.allocated_capital),
            'current_positions': self.current_positions,
            'description': self.description,
            'tags': self.tags
        }


@dataclass
class PromotionEvent:
    """Records a champion-challenger promotion event."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Strategy information
    old_champion_id: str = ""
    new_champion_id: str = ""
    old_champion_name: str = ""
    new_champion_name: str = ""
    
    # Promotion details
    reason: PromotionReason = PromotionReason.MANUAL_OVERRIDE
    confidence_score: float = 0.0  # Statistical confidence of the decision
    
    # Performance comparison
    old_champion_performance: Dict[str, Any] = field(default_factory=dict)
    new_champion_performance: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    evaluation_period_days: int = 0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'old_champion_id': self.old_champion_id,
            'new_champion_id': self.new_champion_id,
            'old_champion_name': self.old_champion_name,
            'new_champion_name': self.new_champion_name,
            'reason': self.reason.value,
            'confidence_score': self.confidence_score,
            'old_champion_performance': self.old_champion_performance,
            'new_champion_performance': self.new_champion_performance,
            'evaluation_period_days': self.evaluation_period_days,
            'notes': self.notes
        }


@dataclass
class EVOPConfiguration:
    """Configuration for the EVOP framework."""
    
    # Champion-challenger setup
    max_challengers: int = 3
    challenger_capital_fraction: float = 0.2  # Each challenger gets 20% of champion's capital
    
    # Evaluation parameters
    min_evaluation_period_days: int = 7
    max_evaluation_period_days: int = 30
    min_trades_for_evaluation: int = 10
    
    # Promotion criteria
    required_confidence_level: float = 0.95  # Statistical significance threshold
    min_sharpe_improvement: float = 0.1
    min_calmar_improvement: float = 0.1
    max_drawdown_tolerance: float = 0.15  # 15%
    
    # Risk controls
    max_total_allocation: float = 1.0  # 100% of available capital
    emergency_stop_drawdown: float = 0.25  # 25% total portfolio drawdown
    
    # Challenger generation
    parameter_mutation_rate: float = 0.1  # 10% parameter variation
    parameter_mutation_std: float = 0.05  # 5% standard deviation for mutations
    
    # Operational settings
    evaluation_frequency_hours: int = 6  # Check for promotions every 6 hours
    challenger_restart_on_failure: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'max_challengers': self.max_challengers,
            'challenger_capital_fraction': self.challenger_capital_fraction,
            'min_evaluation_period_days': self.min_evaluation_period_days,
            'max_evaluation_period_days': self.max_evaluation_period_days,
            'min_trades_for_evaluation': self.min_trades_for_evaluation,
            'required_confidence_level': self.required_confidence_level,
            'min_sharpe_improvement': self.min_sharpe_improvement,
            'min_calmar_improvement': self.min_calmar_improvement,
            'max_drawdown_tolerance': self.max_drawdown_tolerance,
            'max_total_allocation': self.max_total_allocation,
            'emergency_stop_drawdown': self.emergency_stop_drawdown,
            'parameter_mutation_rate': self.parameter_mutation_rate,
            'parameter_mutation_std': self.parameter_mutation_std,
            'evaluation_frequency_hours': self.evaluation_frequency_hours,
            'challenger_restart_on_failure': self.challenger_restart_on_failure
        }
