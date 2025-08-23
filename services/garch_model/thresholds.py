"""Dynamic threshold calculator for entry/exit signals based on GARCH volatility."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import structlog
from dataclasses import dataclass
from collections import deque
import scipy.stats as stats

from .model import GARCHForecast

logger = structlog.get_logger(__name__)


@dataclass
class ThresholdSignal:
    """Container for threshold-based trading signals."""
    pair_id: str
    spread_value: float
    z_score: float
    volatility_forecast: float
    
    # Threshold levels
    entry_threshold_long: float   # Enter long when z-score < this (negative)
    entry_threshold_short: float  # Enter short when z-score > this (positive)
    exit_threshold: float         # Exit when |z-score| < this
    
    # Signal classification
    signal_type: str  # 'entry_long', 'entry_short', 'exit', 'hold'
    signal_strength: float  # 0-1, based on how far past threshold
    
    # Confidence metrics
    confidence_level: float
    volatility_regime: str  # 'low', 'normal', 'high'
    
    timestamp: datetime


@dataclass
class AdaptiveThresholds:
    """Container for adaptive threshold levels."""
    entry_long: float
    entry_short: float
    exit: float
    stop_loss_long: float
    stop_loss_short: float
    volatility_regime: str
    confidence: float
    timestamp: datetime


class DynamicThresholdCalculator:
    """
    Calculates dynamic entry/exit thresholds based on GARCH volatility forecasts.
    
    This class implements adaptive Bollinger Band-style thresholds that adjust
    based on forecasted volatility, creating wider bands during high volatility
    periods and tighter bands during low volatility periods.
    """
    
    def __init__(
        self,
        base_entry_threshold: float = 2.0,    # Base z-score for entry
        base_exit_threshold: float = 0.5,     # Base z-score for exit
        volatility_adjustment_factor: float = 0.5,  # How much volatility affects thresholds
        min_threshold: float = 1.0,           # Minimum threshold level
        max_threshold: float = 4.0,           # Maximum threshold level
        lookback_window: int = 50,            # Window for spread statistics
        volatility_percentiles: Tuple[float, float] = (25, 75)  # For regime classification
    ):
        """
        Initialize the dynamic threshold calculator.
        
        Args:
            base_entry_threshold: Base z-score threshold for entry signals
            base_exit_threshold: Base z-score threshold for exit signals
            volatility_adjustment_factor: Factor for volatility-based adjustments
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            lookback_window: Window size for calculating spread statistics
            volatility_percentiles: Percentiles for volatility regime classification
        """
        self.base_entry_threshold = base_entry_threshold
        self.base_exit_threshold = base_exit_threshold
        self.volatility_adjustment_factor = volatility_adjustment_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.lookback_window = lookback_window
        self.volatility_percentiles = volatility_percentiles
        
        # Rolling data storage
        self.spread_history = deque(maxlen=lookback_window)
        self.volatility_history = deque(maxlen=lookback_window)
        self.timestamp_history = deque(maxlen=lookback_window)
        
        # Statistics
        self.current_mean = 0.0
        self.current_std = 1.0
        self.volatility_percentile_25 = 0.0
        self.volatility_percentile_75 = 0.0
        
        logger.info(
            "Dynamic threshold calculator initialized",
            base_entry_threshold=base_entry_threshold,
            base_exit_threshold=base_exit_threshold,
            volatility_adjustment_factor=volatility_adjustment_factor,
            lookback_window=lookback_window
        )
    
    def update_statistics(
        self,
        spread_value: float,
        volatility_forecast: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update rolling statistics with new spread and volatility data.
        
        Args:
            spread_value: Current spread value
            volatility_forecast: Current volatility forecast
            timestamp: Timestamp of the observation
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Add to rolling windows
        self.spread_history.append(spread_value)
        self.timestamp_history.append(timestamp)
        
        if volatility_forecast is not None:
            self.volatility_history.append(volatility_forecast)
        
        # Update spread statistics
        if len(self.spread_history) >= 2:
            spread_array = np.array(self.spread_history)
            self.current_mean = np.mean(spread_array)
            self.current_std = np.std(spread_array, ddof=1)
        
        # Update volatility percentiles
        if len(self.volatility_history) >= 10:
            vol_array = np.array(self.volatility_history)
            self.volatility_percentile_25 = np.percentile(vol_array, self.volatility_percentiles[0])
            self.volatility_percentile_75 = np.percentile(vol_array, self.volatility_percentiles[1])
        
        logger.debug(
            "Updated threshold statistics",
            spread_mean=self.current_mean,
            spread_std=self.current_std,
            volatility_p25=self.volatility_percentile_25,
            volatility_p75=self.volatility_percentile_75,
            window_size=len(self.spread_history)
        )
    
    def calculate_z_score(self, spread_value: float) -> float:
        """Calculate z-score for the current spread value."""
        if self.current_std == 0:
            return 0.0
        return (spread_value - self.current_mean) / self.current_std
    
    def classify_volatility_regime(self, volatility_forecast: float) -> str:
        """
        Classify the current volatility regime.
        
        Args:
            volatility_forecast: Current volatility forecast
            
        Returns:
            'low', 'normal', or 'high' volatility regime
        """
        if len(self.volatility_history) < 10:
            return 'normal'
        
        if volatility_forecast <= self.volatility_percentile_25:
            return 'low'
        elif volatility_forecast >= self.volatility_percentile_75:
            return 'high'
        else:
            return 'normal'
    
    def calculate_adaptive_thresholds(
        self,
        volatility_forecast: float,
        confidence_level: float = 0.95
    ) -> AdaptiveThresholds:
        """
        Calculate adaptive thresholds based on volatility forecast.
        
        Args:
            volatility_forecast: GARCH volatility forecast
            confidence_level: Confidence level for threshold calculation
            
        Returns:
            AdaptiveThresholds object with calculated threshold levels
        """
        # Classify volatility regime
        vol_regime = self.classify_volatility_regime(volatility_forecast)
        
        # Calculate volatility adjustment
        if len(self.volatility_history) > 0:
            vol_mean = np.mean(self.volatility_history)
            vol_ratio = volatility_forecast / vol_mean if vol_mean > 0 else 1.0
        else:
            vol_ratio = 1.0
        
        # Apply volatility adjustment to base thresholds
        volatility_multiplier = 1.0 + self.volatility_adjustment_factor * (vol_ratio - 1.0)
        
        # Calculate adjusted thresholds
        entry_threshold = self.base_entry_threshold * volatility_multiplier
        exit_threshold = self.base_exit_threshold * volatility_multiplier
        
        # Apply bounds
        entry_threshold = np.clip(entry_threshold, self.min_threshold, self.max_threshold)
        exit_threshold = np.clip(exit_threshold, 0.1, entry_threshold * 0.8)
        
        # Calculate stop-loss thresholds (wider than entry)
        stop_loss_multiplier = 1.5
        stop_loss_threshold = entry_threshold * stop_loss_multiplier
        stop_loss_threshold = np.clip(stop_loss_threshold, entry_threshold, self.max_threshold * 1.5)
        
        # Calculate confidence based on data availability and volatility stability
        data_confidence = min(1.0, len(self.spread_history) / self.lookback_window)
        
        if len(self.volatility_history) > 1:
            vol_stability = 1.0 / (1.0 + np.std(self.volatility_history))
        else:
            vol_stability = 0.5
        
        overall_confidence = (data_confidence + vol_stability) / 2.0
        
        return AdaptiveThresholds(
            entry_long=-entry_threshold,      # Negative for long entries
            entry_short=entry_threshold,      # Positive for short entries
            exit=exit_threshold,
            stop_loss_long=-stop_loss_threshold,
            stop_loss_short=stop_loss_threshold,
            volatility_regime=vol_regime,
            confidence=overall_confidence,
            timestamp=datetime.utcnow()
        )
    
    def generate_signal(
        self,
        pair_id: str,
        spread_value: float,
        garch_forecast: GARCHForecast,
        current_position: Optional[str] = None  # 'long', 'short', or None
    ) -> ThresholdSignal:
        """
        Generate trading signal based on current spread and volatility forecast.
        
        Args:
            pair_id: Trading pair identifier
            spread_value: Current spread value
            garch_forecast: GARCH volatility forecast
            current_position: Current position if any
            
        Returns:
            ThresholdSignal with trading recommendation
        """
        # Update statistics
        self.update_statistics(spread_value, garch_forecast.volatility_forecast)
        
        # Calculate z-score
        z_score = self.calculate_z_score(spread_value)
        
        # Calculate adaptive thresholds
        thresholds = self.calculate_adaptive_thresholds(garch_forecast.volatility_forecast)
        
        # Determine signal type and strength
        signal_type = 'hold'
        signal_strength = 0.0
        
        if current_position is None:
            # No current position - look for entry signals
            if z_score <= thresholds.entry_long:
                signal_type = 'entry_long'
                signal_strength = min(1.0, abs(z_score / thresholds.entry_long))
            elif z_score >= thresholds.entry_short:
                signal_type = 'entry_short'
                signal_strength = min(1.0, abs(z_score / thresholds.entry_short))
        
        else:
            # Have position - look for exit signals
            if current_position == 'long':
                if z_score >= -thresholds.exit or z_score <= thresholds.stop_loss_long:
                    signal_type = 'exit'
                    if z_score <= thresholds.stop_loss_long:
                        signal_strength = 1.0  # Stop loss
                    else:
                        signal_strength = min(1.0, abs(z_score / thresholds.exit))
            
            elif current_position == 'short':
                if z_score <= thresholds.exit or z_score >= thresholds.stop_loss_short:
                    signal_type = 'exit'
                    if z_score >= thresholds.stop_loss_short:
                        signal_strength = 1.0  # Stop loss
                    else:
                        signal_strength = min(1.0, abs(z_score / thresholds.exit))
        
        return ThresholdSignal(
            pair_id=pair_id,
            spread_value=spread_value,
            z_score=z_score,
            volatility_forecast=garch_forecast.volatility_forecast,
            entry_threshold_long=thresholds.entry_long,
            entry_threshold_short=thresholds.entry_short,
            exit_threshold=thresholds.exit,
            signal_type=signal_type,
            signal_strength=signal_strength,
            confidence_level=thresholds.confidence,
            volatility_regime=thresholds.volatility_regime,
            timestamp=datetime.utcnow()
        )
    
    def get_current_thresholds(self, volatility_forecast: float) -> AdaptiveThresholds:
        """Get current threshold levels without generating a signal."""
        return self.calculate_adaptive_thresholds(volatility_forecast)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics and state information."""
        return {
            'current_mean': self.current_mean,
            'current_std': self.current_std,
            'volatility_p25': self.volatility_percentile_25,
            'volatility_p75': self.volatility_percentile_75,
            'spread_history_size': len(self.spread_history),
            'volatility_history_size': len(self.volatility_history),
            'base_entry_threshold': self.base_entry_threshold,
            'base_exit_threshold': self.base_exit_threshold,
            'volatility_adjustment_factor': self.volatility_adjustment_factor
        }
    
    def reset(self) -> None:
        """Reset all statistics and history."""
        self.spread_history.clear()
        self.volatility_history.clear()
        self.timestamp_history.clear()
        self.current_mean = 0.0
        self.current_std = 1.0
        self.volatility_percentile_25 = 0.0
        self.volatility_percentile_75 = 0.0
        
        logger.info("Threshold calculator reset")


class MultiPairThresholdManager:
    """
    Manages threshold calculators for multiple trading pairs.
    """
    
    def __init__(
        self,
        default_base_entry_threshold: float = 2.0,
        default_base_exit_threshold: float = 0.5,
        default_volatility_adjustment_factor: float = 0.5
    ):
        """Initialize the multi-pair threshold manager."""
        self.default_base_entry_threshold = default_base_entry_threshold
        self.default_base_exit_threshold = default_base_exit_threshold
        self.default_volatility_adjustment_factor = default_volatility_adjustment_factor
        
        self.calculators: Dict[str, DynamicThresholdCalculator] = {}
        self.pair_configs: Dict[str, Dict[str, Any]] = {}
        self.current_positions: Dict[str, Optional[str]] = {}
        
        logger.info("Multi-pair threshold manager initialized")
    
    def add_pair(
        self,
        pair_id: str,
        base_entry_threshold: Optional[float] = None,
        base_exit_threshold: Optional[float] = None,
        volatility_adjustment_factor: Optional[float] = None,
        lookback_window: int = 50
    ) -> None:
        """Add a new trading pair for threshold calculation."""
        config = {
            'base_entry_threshold': base_entry_threshold or self.default_base_entry_threshold,
            'base_exit_threshold': base_exit_threshold or self.default_base_exit_threshold,
            'volatility_adjustment_factor': volatility_adjustment_factor or self.default_volatility_adjustment_factor,
            'lookback_window': lookback_window
        }
        
        self.calculators[pair_id] = DynamicThresholdCalculator(**config)
        self.pair_configs[pair_id] = config
        self.current_positions[pair_id] = None
        
        logger.info(
            "Added threshold calculator for pair",
            pair_id=pair_id,
            **config
        )
    
    def generate_signal(
        self,
        pair_id: str,
        spread_value: float,
        garch_forecast: GARCHForecast
    ) -> Optional[ThresholdSignal]:
        """Generate trading signal for a specific pair."""
        if pair_id not in self.calculators:
            logger.warning("Unknown pair for threshold calculation", pair_id=pair_id)
            return None
        
        current_position = self.current_positions.get(pair_id)
        
        signal = self.calculators[pair_id].generate_signal(
            pair_id, spread_value, garch_forecast, current_position
        )
        
        # Update position tracking based on signal
        if signal.signal_type in ['entry_long', 'entry_short']:
            self.current_positions[pair_id] = signal.signal_type.replace('entry_', '')
        elif signal.signal_type == 'exit':
            self.current_positions[pair_id] = None
        
        return signal
    
    def update_position(self, pair_id: str, position: Optional[str]) -> None:
        """Manually update position for a pair."""
        if pair_id in self.current_positions:
            self.current_positions[pair_id] = position
            logger.debug("Updated position", pair_id=pair_id, position=position)
    
    def get_current_thresholds(
        self, 
        pair_id: str, 
        volatility_forecast: float
    ) -> Optional[AdaptiveThresholds]:
        """Get current thresholds for a specific pair."""
        if pair_id not in self.calculators:
            return None
        return self.calculators[pair_id].get_current_thresholds(volatility_forecast)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pairs."""
        stats = {}
        for pair_id, calculator in self.calculators.items():
            stats[pair_id] = {
                **calculator.get_statistics(),
                'current_position': self.current_positions.get(pair_id)
            }
        return stats
