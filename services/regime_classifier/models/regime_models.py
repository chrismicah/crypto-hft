"""
Data models for market regime classification system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Market regime classifications."""
    LOW_VOL_BULL = "low_vol_bull"           # Low volatility, upward trend
    LOW_VOL_BEAR = "low_vol_bear"           # Low volatility, downward trend
    LOW_VOL_RANGE = "low_vol_range"         # Low volatility, sideways
    HIGH_VOL_BULL = "high_vol_bull"         # High volatility, upward trend
    HIGH_VOL_BEAR = "high_vol_bear"         # High volatility, downward trend
    HIGH_VOL_RANGE = "high_vol_range"       # High volatility, sideways
    STABLE_RANGE = "stable_range"           # Stable, range-bound market
    TRENDING_UP = "trending_up"             # Strong upward trend
    TRENDING_DOWN = "trending_down"         # Strong downward trend
    CRISIS = "crisis"                       # Market crisis/extreme volatility
    RECOVERY = "recovery"                   # Post-crisis recovery
    UNKNOWN = "unknown"                     # Unclassified/transition state


class RegimeConfidence(str, Enum):
    """Confidence levels for regime classification."""
    VERY_LOW = "very_low"       # < 0.3
    LOW = "low"                 # 0.3 - 0.5
    MEDIUM = "medium"           # 0.5 - 0.7
    HIGH = "high"               # 0.7 - 0.9
    VERY_HIGH = "very_high"     # > 0.9


@dataclass
class MarketFeatures:
    """Market features for regime classification."""
    timestamp: datetime
    
    # Price-based features
    returns_1h: float = 0.0
    returns_4h: float = 0.0
    returns_24h: float = 0.0
    returns_7d: float = 0.0
    
    # Volatility features
    volatility_1h: float = 0.0
    volatility_4h: float = 0.0
    volatility_24h: float = 0.0
    volatility_7d: float = 0.0
    
    # Volume features
    volume_ratio_1h: float = 1.0    # Current volume / average volume
    volume_ratio_4h: float = 1.0
    volume_ratio_24h: float = 1.0
    
    # Technical indicators
    rsi_14: float = 50.0            # RSI (14 periods)
    macd_signal: float = 0.0        # MACD signal
    bollinger_position: float = 0.5  # Position within Bollinger Bands
    
    # Market microstructure
    bid_ask_spread: float = 0.0     # Bid-ask spread
    order_book_imbalance: float = 0.0  # Order book imbalance
    trade_intensity: float = 0.0    # Trades per minute
    
    # Cross-asset features
    btc_correlation: float = 0.0    # Correlation with BTC
    market_beta: float = 1.0        # Market beta
    
    # Funding and derivatives
    funding_rate: float = 0.0       # Perpetual funding rate
    open_interest_change: float = 0.0  # Change in open interest
    
    # Sentiment indicators
    fear_greed_index: Optional[float] = None
    social_sentiment: Optional[float] = None
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array([
            self.returns_1h, self.returns_4h, self.returns_24h, self.returns_7d,
            self.volatility_1h, self.volatility_4h, self.volatility_24h, self.volatility_7d,
            self.volume_ratio_1h, self.volume_ratio_4h, self.volume_ratio_24h,
            self.rsi_14, self.macd_signal, self.bollinger_position,
            self.bid_ask_spread, self.order_book_imbalance, self.trade_intensity,
            self.btc_correlation, self.market_beta,
            self.funding_rate, self.open_interest_change,
            self.fear_greed_index or 50.0, self.social_sentiment or 0.0
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'returns_1h': self.returns_1h,
            'returns_4h': self.returns_4h,
            'returns_24h': self.returns_24h,
            'returns_7d': self.returns_7d,
            'volatility_1h': self.volatility_1h,
            'volatility_4h': self.volatility_4h,
            'volatility_24h': self.volatility_24h,
            'volatility_7d': self.volatility_7d,
            'volume_ratio_1h': self.volume_ratio_1h,
            'volume_ratio_4h': self.volume_ratio_4h,
            'volume_ratio_24h': self.volume_ratio_24h,
            'rsi_14': self.rsi_14,
            'macd_signal': self.macd_signal,
            'bollinger_position': self.bollinger_position,
            'bid_ask_spread': self.bid_ask_spread,
            'order_book_imbalance': self.order_book_imbalance,
            'trade_intensity': self.trade_intensity,
            'btc_correlation': self.btc_correlation,
            'market_beta': self.market_beta,
            'funding_rate': self.funding_rate,
            'open_interest_change': self.open_interest_change,
            'fear_greed_index': self.fear_greed_index,
            'social_sentiment': self.social_sentiment
        }


@dataclass
class RegimeClassification:
    """Result of market regime classification."""
    timestamp: datetime
    regime: MarketRegime
    confidence: float
    confidence_level: RegimeConfidence
    
    # Probability distribution over all regimes
    regime_probabilities: Dict[MarketRegime, float] = field(default_factory=dict)
    
    # Features used for classification
    features: Optional[MarketFeatures] = None
    
    # Model information
    model_version: str = "1.0"
    classification_time: float = 0.0  # Time taken for classification
    
    # Regime transition information
    previous_regime: Optional[MarketRegime] = None
    regime_duration: Optional[timedelta] = None
    transition_probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'regime': self.regime.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'regime_probabilities': {k.value: v for k, v in self.regime_probabilities.items()},
            'features': self.features.to_dict() if self.features else None,
            'model_version': self.model_version,
            'classification_time': self.classification_time,
            'previous_regime': self.previous_regime.value if self.previous_regime else None,
            'regime_duration': self.regime_duration.total_seconds() if self.regime_duration else None,
            'transition_probability': self.transition_probability
        }


@dataclass
class RegimeTransition:
    """Information about regime transitions."""
    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_probability: float
    duration_in_previous: timedelta
    
    # Transition characteristics
    is_gradual: bool = True         # Gradual vs sudden transition
    confidence_drop: float = 0.0    # Drop in confidence during transition
    
    # Market impact
    price_change_during_transition: float = 0.0
    volatility_change: float = 0.0
    volume_change: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'from_regime': self.from_regime.value,
            'to_regime': self.to_regime.value,
            'transition_probability': self.transition_probability,
            'duration_in_previous': self.duration_in_previous.total_seconds(),
            'is_gradual': self.is_gradual,
            'confidence_drop': self.confidence_drop,
            'price_change_during_transition': self.price_change_during_transition,
            'volatility_change': self.volatility_change,
            'volume_change': self.volume_change
        }


@dataclass
class StrategyParameters:
    """Strategy parameters for different market regimes."""
    regime: MarketRegime
    
    # Entry/exit thresholds
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    stop_loss_z_score: float = 4.0
    
    # Position sizing
    max_position_size: float = 1.0
    position_sizing_factor: float = 1.0
    kelly_fraction: float = 0.25
    
    # Risk management
    max_drawdown_threshold: float = 0.05
    volatility_scaling: bool = True
    correlation_threshold: float = 0.7
    
    # Execution parameters
    execution_urgency: float = 0.5      # 0 = patient, 1 = aggressive
    slippage_tolerance: float = 0.001   # 0.1% slippage tolerance
    
    # Timing parameters
    holding_period_target: int = 240    # Target holding period in minutes
    rebalance_frequency: int = 60       # Rebalance frequency in minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'entry_z_score': self.entry_z_score,
            'exit_z_score': self.exit_z_score,
            'stop_loss_z_score': self.stop_loss_z_score,
            'max_position_size': self.max_position_size,
            'position_sizing_factor': self.position_sizing_factor,
            'kelly_fraction': self.kelly_fraction,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'volatility_scaling': self.volatility_scaling,
            'correlation_threshold': self.correlation_threshold,
            'execution_urgency': self.execution_urgency,
            'slippage_tolerance': self.slippage_tolerance,
            'holding_period_target': self.holding_period_target,
            'rebalance_frequency': self.rebalance_frequency
        }


class RegimeModelConfig(BaseModel):
    """Configuration for regime classification models."""
    
    # HMM parameters
    n_components: int = Field(6, description="Number of hidden states (regimes)")
    covariance_type: str = Field("full", description="Covariance type for HMM")
    n_iter: int = Field(100, description="Maximum iterations for training")
    tol: float = Field(1e-2, description="Convergence tolerance")
    
    # Feature parameters
    lookback_periods: List[int] = Field([1, 4, 24, 168], description="Lookback periods in hours")
    feature_scaling: bool = Field(True, description="Whether to scale features")
    feature_selection: bool = Field(True, description="Whether to perform feature selection")
    
    # Classification parameters
    min_confidence_threshold: float = Field(0.3, description="Minimum confidence for classification")
    transition_smoothing: float = Field(0.1, description="Smoothing factor for transitions")
    regime_persistence: int = Field(3, description="Minimum periods to confirm regime change")
    
    # Training parameters
    training_window: int = Field(2160, description="Training window in hours (90 days)")
    retrain_frequency: int = Field(168, description="Retrain frequency in hours (weekly)")
    validation_split: float = Field(0.2, description="Validation split ratio")
    
    class Config:
        use_enum_values = True


@dataclass
class RegimePerformanceMetrics:
    """Performance metrics for regime classification."""
    
    # Classification accuracy
    overall_accuracy: float = 0.0
    regime_specific_accuracy: Dict[MarketRegime, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    
    # Transition detection
    transition_detection_rate: float = 0.0
    false_transition_rate: float = 0.0
    average_transition_delay: float = 0.0  # Hours
    
    # Model performance
    log_likelihood: float = 0.0
    aic_score: float = 0.0
    bic_score: float = 0.0
    
    # Trading performance by regime
    regime_returns: Dict[MarketRegime, float] = field(default_factory=dict)
    regime_sharpe_ratios: Dict[MarketRegime, float] = field(default_factory=dict)
    regime_max_drawdowns: Dict[MarketRegime, float] = field(default_factory=dict)
    
    # Stability metrics
    regime_persistence: Dict[MarketRegime, float] = field(default_factory=dict)  # Average duration
    transition_frequency: float = 0.0  # Transitions per day
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_accuracy': self.overall_accuracy,
            'regime_specific_accuracy': {k.value: v for k, v in self.regime_specific_accuracy.items()},
            'transition_detection_rate': self.transition_detection_rate,
            'false_transition_rate': self.false_transition_rate,
            'average_transition_delay': self.average_transition_delay,
            'log_likelihood': self.log_likelihood,
            'aic_score': self.aic_score,
            'bic_score': self.bic_score,
            'regime_returns': {k.value: v for k, v in self.regime_returns.items()},
            'regime_sharpe_ratios': {k.value: v for k, v in self.regime_sharpe_ratios.items()},
            'regime_max_drawdowns': {k.value: v for k, v in self.regime_max_drawdowns.items()},
            'regime_persistence': {k.value: v for k, v in self.regime_persistence.items()},
            'transition_frequency': self.transition_frequency
        }


@dataclass
class RegimeAlert:
    """Alert for significant regime changes."""
    timestamp: datetime
    alert_type: str  # "regime_change", "high_volatility", "crisis_detected"
    regime: MarketRegime
    previous_regime: Optional[MarketRegime]
    confidence: float
    
    # Alert details
    message: str
    severity: str  # "low", "medium", "high", "critical"
    
    # Recommended actions
    suggested_actions: List[str] = field(default_factory=list)
    
    # Impact assessment
    expected_impact: str = "unknown"  # "low", "medium", "high"
    risk_level: str = "medium"       # "low", "medium", "high", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'regime': self.regime.value,
            'previous_regime': self.previous_regime.value if self.previous_regime else None,
            'confidence': self.confidence,
            'message': self.message,
            'severity': self.severity,
            'suggested_actions': self.suggested_actions,
            'expected_impact': self.expected_impact,
            'risk_level': self.risk_level
        }


# Default strategy parameters for each regime
DEFAULT_STRATEGY_PARAMETERS = {
    MarketRegime.LOW_VOL_BULL: StrategyParameters(
        regime=MarketRegime.LOW_VOL_BULL,
        entry_z_score=1.5,
        exit_z_score=0.3,
        max_position_size=1.2,
        position_sizing_factor=1.1,
        kelly_fraction=0.3,
        execution_urgency=0.3,
        holding_period_target=360
    ),
    MarketRegime.LOW_VOL_BEAR: StrategyParameters(
        regime=MarketRegime.LOW_VOL_BEAR,
        entry_z_score=1.8,
        exit_z_score=0.4,
        max_position_size=0.8,
        position_sizing_factor=0.9,
        kelly_fraction=0.2,
        execution_urgency=0.4,
        holding_period_target=300
    ),
    MarketRegime.LOW_VOL_RANGE: StrategyParameters(
        regime=MarketRegime.LOW_VOL_RANGE,
        entry_z_score=1.2,
        exit_z_score=0.2,
        max_position_size=1.5,
        position_sizing_factor=1.2,
        kelly_fraction=0.35,
        execution_urgency=0.2,
        holding_period_target=180
    ),
    MarketRegime.HIGH_VOL_BULL: StrategyParameters(
        regime=MarketRegime.HIGH_VOL_BULL,
        entry_z_score=2.5,
        exit_z_score=0.8,
        max_position_size=0.6,
        position_sizing_factor=0.7,
        kelly_fraction=0.15,
        execution_urgency=0.7,
        holding_period_target=120
    ),
    MarketRegime.HIGH_VOL_BEAR: StrategyParameters(
        regime=MarketRegime.HIGH_VOL_BEAR,
        entry_z_score=3.0,
        exit_z_score=1.0,
        max_position_size=0.4,
        position_sizing_factor=0.5,
        kelly_fraction=0.1,
        execution_urgency=0.8,
        holding_period_target=90
    ),
    MarketRegime.HIGH_VOL_RANGE: StrategyParameters(
        regime=MarketRegime.HIGH_VOL_RANGE,
        entry_z_score=2.2,
        exit_z_score=0.6,
        max_position_size=0.8,
        position_sizing_factor=0.8,
        kelly_fraction=0.2,
        execution_urgency=0.6,
        holding_period_target=150
    ),
    MarketRegime.STABLE_RANGE: StrategyParameters(
        regime=MarketRegime.STABLE_RANGE,
        entry_z_score=1.0,
        exit_z_score=0.1,
        max_position_size=1.8,
        position_sizing_factor=1.3,
        kelly_fraction=0.4,
        execution_urgency=0.1,
        holding_period_target=480
    ),
    MarketRegime.TRENDING_UP: StrategyParameters(
        regime=MarketRegime.TRENDING_UP,
        entry_z_score=1.8,
        exit_z_score=0.5,
        max_position_size=1.0,
        position_sizing_factor=1.0,
        kelly_fraction=0.25,
        execution_urgency=0.4,
        holding_period_target=240
    ),
    MarketRegime.TRENDING_DOWN: StrategyParameters(
        regime=MarketRegime.TRENDING_DOWN,
        entry_z_score=2.2,
        exit_z_score=0.7,
        max_position_size=0.7,
        position_sizing_factor=0.8,
        kelly_fraction=0.18,
        execution_urgency=0.5,
        holding_period_target=200
    ),
    MarketRegime.CRISIS: StrategyParameters(
        regime=MarketRegime.CRISIS,
        entry_z_score=5.0,
        exit_z_score=2.0,
        max_position_size=0.2,
        position_sizing_factor=0.3,
        kelly_fraction=0.05,
        execution_urgency=0.9,
        holding_period_target=60
    ),
    MarketRegime.RECOVERY: StrategyParameters(
        regime=MarketRegime.RECOVERY,
        entry_z_score=2.8,
        exit_z_score=1.2,
        max_position_size=0.5,
        position_sizing_factor=0.6,
        kelly_fraction=0.12,
        execution_urgency=0.6,
        holding_period_target=180
    ),
    MarketRegime.UNKNOWN: StrategyParameters(
        regime=MarketRegime.UNKNOWN,
        entry_z_score=2.0,
        exit_z_score=0.5,
        max_position_size=0.5,
        position_sizing_factor=0.7,
        kelly_fraction=0.15,
        execution_urgency=0.5,
        holding_period_target=240
    )
}


# Utility functions
def get_confidence_level(confidence: float) -> RegimeConfidence:
    """Convert confidence score to confidence level enum."""
    if confidence < 0.3:
        return RegimeConfidence.VERY_LOW
    elif confidence < 0.5:
        return RegimeConfidence.LOW
    elif confidence < 0.7:
        return RegimeConfidence.MEDIUM
    elif confidence < 0.9:
        return RegimeConfidence.HIGH
    else:
        return RegimeConfidence.VERY_HIGH


def is_high_volatility_regime(regime: MarketRegime) -> bool:
    """Check if regime is a high volatility regime."""
    return regime in [
        MarketRegime.HIGH_VOL_BULL,
        MarketRegime.HIGH_VOL_BEAR,
        MarketRegime.HIGH_VOL_RANGE,
        MarketRegime.CRISIS
    ]


def is_trending_regime(regime: MarketRegime) -> bool:
    """Check if regime is a trending regime."""
    return regime in [
        MarketRegime.LOW_VOL_BULL,
        MarketRegime.LOW_VOL_BEAR,
        MarketRegime.HIGH_VOL_BULL,
        MarketRegime.HIGH_VOL_BEAR,
        MarketRegime.TRENDING_UP,
        MarketRegime.TRENDING_DOWN
    ]


def get_regime_risk_level(regime: MarketRegime) -> str:
    """Get risk level for a regime."""
    risk_levels = {
        MarketRegime.LOW_VOL_BULL: "low",
        MarketRegime.LOW_VOL_BEAR: "low",
        MarketRegime.LOW_VOL_RANGE: "very_low",
        MarketRegime.HIGH_VOL_BULL: "medium",
        MarketRegime.HIGH_VOL_BEAR: "medium",
        MarketRegime.HIGH_VOL_RANGE: "medium",
        MarketRegime.STABLE_RANGE: "very_low",
        MarketRegime.TRENDING_UP: "low",
        MarketRegime.TRENDING_DOWN: "low",
        MarketRegime.CRISIS: "very_high",
        MarketRegime.RECOVERY: "high",
        MarketRegime.UNKNOWN: "medium"
    }
    return risk_levels.get(regime, "medium")
