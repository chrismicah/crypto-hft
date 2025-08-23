"""Signal generation logic for trading decisions."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import statistics

logger = structlog.get_logger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT = "exit"
    HOLD = "hold"
    STOP_LOSS = "stop_loss"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class MarketData:
    """Container for market data from various sources."""
    # Order book data
    asset1_bid: Optional[float] = None
    asset1_ask: Optional[float] = None
    asset2_bid: Optional[float] = None
    asset2_ask: Optional[float] = None
    
    # Hedge ratio from Kalman filter
    hedge_ratio: Optional[float] = None
    hedge_ratio_confidence: Optional[float] = None
    
    # Volatility and thresholds from GARCH
    volatility_forecast: Optional[float] = None
    entry_threshold_long: Optional[float] = None
    entry_threshold_short: Optional[float] = None
    exit_threshold: Optional[float] = None
    volatility_regime: Optional[str] = None
    
    # Calculated values
    spread: Optional[float] = None
    z_score: Optional[float] = None
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class TradingSignal:
    """Container for generated trading signals."""
    pair_id: str
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    
    # Trade parameters
    side: Optional[str] = None  # 'long' or 'short'
    entry_price_asset1: Optional[float] = None
    entry_price_asset2: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    # Signal context
    current_spread: Optional[float] = None
    current_z_score: Optional[float] = None
    hedge_ratio: Optional[float] = None
    volatility_forecast: Optional[float] = None
    
    # Risk metrics
    expected_return: Optional[float] = None
    max_drawdown_risk: Optional[float] = None
    
    timestamp: datetime = None
    reason: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class SpreadCalculator:
    """Calculates spread and z-score from market data."""
    
    def __init__(self, lookback_window: int = 100):
        """
        Initialize spread calculator.
        
        Args:
            lookback_window: Number of historical spreads to keep for z-score calculation
        """
        self.lookback_window = lookback_window
        self.spread_history: Dict[str, List[float]] = {}
        self.timestamp_history: Dict[str, List[datetime]] = {}
        
    def calculate_spread(
        self,
        pair_id: str,
        asset1_price: float,
        asset2_price: float,
        hedge_ratio: float,
        method: str = 'ratio'
    ) -> float:
        """
        Calculate spread between two assets.
        
        Args:
            pair_id: Trading pair identifier
            asset1_price: Price of first asset
            asset2_price: Price of second asset
            hedge_ratio: Hedge ratio from Kalman filter
            method: Calculation method ('ratio', 'difference', 'log_ratio')
            
        Returns:
            Calculated spread value
        """
        try:
            if method == 'ratio':
                # Price ratio adjusted by hedge ratio
                spread = asset1_price / asset2_price - hedge_ratio
            elif method == 'difference':
                # Price difference adjusted by hedge ratio
                spread = asset1_price - (hedge_ratio * asset2_price)
            elif method == 'log_ratio':
                # Log price ratio
                spread = np.log(asset1_price / asset2_price) - np.log(hedge_ratio)
            else:
                raise ValueError(f"Unknown spread calculation method: {method}")
            
            # Store in history
            if pair_id not in self.spread_history:
                self.spread_history[pair_id] = []
                self.timestamp_history[pair_id] = []
            
            self.spread_history[pair_id].append(spread)
            self.timestamp_history[pair_id].append(datetime.utcnow())
            
            # Maintain window size
            if len(self.spread_history[pair_id]) > self.lookback_window:
                self.spread_history[pair_id].pop(0)
                self.timestamp_history[pair_id].pop(0)
            
            return spread
            
        except Exception as e:
            logger.error(
                "Failed to calculate spread",
                pair_id=pair_id,
                asset1_price=asset1_price,
                asset2_price=asset2_price,
                hedge_ratio=hedge_ratio,
                method=method,
                error=str(e)
            )
            return 0.0
    
    def calculate_z_score(self, pair_id: str, current_spread: Optional[float] = None) -> Optional[float]:
        """
        Calculate z-score for the current spread.
        
        Args:
            pair_id: Trading pair identifier
            current_spread: Current spread value (uses last calculated if None)
            
        Returns:
            Z-score or None if insufficient data
        """
        try:
            if pair_id not in self.spread_history or len(self.spread_history[pair_id]) < 2:
                return None
            
            spreads = self.spread_history[pair_id]
            
            if current_spread is None:
                current_spread = spreads[-1]
            
            # Calculate rolling statistics
            mean_spread = statistics.mean(spreads)
            std_spread = statistics.stdev(spreads) if len(spreads) > 1 else 0.0
            
            if std_spread == 0:
                return 0.0
            
            z_score = (current_spread - mean_spread) / std_spread
            
            return z_score
            
        except Exception as e:
            logger.error(
                "Failed to calculate z-score",
                pair_id=pair_id,
                current_spread=current_spread,
                error=str(e)
            )
            return None
    
    def get_spread_statistics(self, pair_id: str) -> Dict[str, float]:
        """Get spread statistics for a pair."""
        if pair_id not in self.spread_history or len(self.spread_history[pair_id]) == 0:
            return {}
        
        spreads = self.spread_history[pair_id]
        
        return {
            'mean': statistics.mean(spreads),
            'std': statistics.stdev(spreads) if len(spreads) > 1 else 0.0,
            'min': min(spreads),
            'max': max(spreads),
            'count': len(spreads),
            'current': spreads[-1] if spreads else 0.0
        }


class SignalGenerator:
    """Generates trading signals based on market data and thresholds."""
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        volatility_adjustment: bool = True,
        risk_adjustment: bool = True
    ):
        """
        Initialize signal generator.
        
        Args:
            min_confidence_threshold: Minimum confidence required for signals
            volatility_adjustment: Whether to adjust signals based on volatility
            risk_adjustment: Whether to apply risk-based adjustments
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.volatility_adjustment = volatility_adjustment
        self.risk_adjustment = risk_adjustment
        
        self.spread_calculator = SpreadCalculator()
        
        # Signal history for analysis
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        
    def generate_signal(
        self,
        pair_id: str,
        market_data: MarketData,
        current_position: Optional[str] = None  # 'long', 'short', or None
    ) -> TradingSignal:
        """
        Generate trading signal based on market data.
        
        Args:
            pair_id: Trading pair identifier
            market_data: Current market data
            current_position: Current position if any
            
        Returns:
            Generated trading signal
        """
        try:
            # Calculate spread and z-score
            if (market_data.asset1_bid and market_data.asset1_ask and 
                market_data.asset2_bid and market_data.asset2_ask and 
                market_data.hedge_ratio):
                
                # Use mid prices for spread calculation
                asset1_mid = (market_data.asset1_bid + market_data.asset1_ask) / 2
                asset2_mid = (market_data.asset2_bid + market_data.asset2_ask) / 2
                
                spread = self.spread_calculator.calculate_spread(
                    pair_id, asset1_mid, asset2_mid, market_data.hedge_ratio
                )
                z_score = self.spread_calculator.calculate_z_score(pair_id, spread)
                
                # Update market data
                market_data.spread = spread
                market_data.z_score = z_score
            
            # Generate signal based on current state
            if current_position is None:
                signal = self._generate_entry_signal(pair_id, market_data)
            else:
                signal = self._generate_exit_signal(pair_id, market_data, current_position)
            
            # Apply adjustments
            signal = self._apply_volatility_adjustment(signal, market_data)
            signal = self._apply_risk_adjustment(signal, market_data)
            
            # Store signal history
            if pair_id not in self.signal_history:
                self.signal_history[pair_id] = []
            
            self.signal_history[pair_id].append(signal)
            
            # Maintain history size
            if len(self.signal_history[pair_id]) > 1000:
                self.signal_history[pair_id] = self.signal_history[pair_id][-500:]
            
            logger.debug(
                "Signal generated",
                pair_id=pair_id,
                signal_type=signal.signal_type.value,
                signal_strength=signal.signal_strength.value,
                confidence=signal.confidence,
                z_score=signal.current_z_score,
                reason=signal.reason
            )
            
            return signal
            
        except Exception as e:
            logger.error(
                "Failed to generate signal",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            
            # Return hold signal on error
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.0,
                reason=f"Error generating signal: {str(e)}"
            )
    
    def _generate_entry_signal(self, pair_id: str, market_data: MarketData) -> TradingSignal:
        """Generate entry signal when no position exists."""
        if (market_data.z_score is None or 
            market_data.entry_threshold_long is None or 
            market_data.entry_threshold_short is None):
            
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.0,
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason="Insufficient data for entry signal"
            )
        
        # Check for entry conditions
        if market_data.z_score <= market_data.entry_threshold_long:
            # Long entry signal (spread is low, expect mean reversion)
            signal_strength = self._calculate_signal_strength(
                abs(market_data.z_score), abs(market_data.entry_threshold_long)
            )
            confidence = self._calculate_confidence(market_data, signal_strength)
            
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=signal_strength,
                confidence=confidence,
                side='long',
                entry_price_asset1=market_data.asset1_ask,  # Buy asset1
                entry_price_asset2=market_data.asset2_bid,  # Sell asset2
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason=f"Z-score {market_data.z_score:.3f} below long threshold {market_data.entry_threshold_long:.3f}"
            )
        
        elif market_data.z_score >= market_data.entry_threshold_short:
            # Short entry signal (spread is high, expect mean reversion)
            signal_strength = self._calculate_signal_strength(
                abs(market_data.z_score), abs(market_data.entry_threshold_short)
            )
            confidence = self._calculate_confidence(market_data, signal_strength)
            
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.ENTRY_SHORT,
                signal_strength=signal_strength,
                confidence=confidence,
                side='short',
                entry_price_asset1=market_data.asset1_bid,  # Sell asset1
                entry_price_asset2=market_data.asset2_ask,  # Buy asset2
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason=f"Z-score {market_data.z_score:.3f} above short threshold {market_data.entry_threshold_short:.3f}"
            )
        
        else:
            # Hold signal
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.5,
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason=f"Z-score {market_data.z_score:.3f} within entry thresholds"
            )
    
    def _generate_exit_signal(
        self, 
        pair_id: str, 
        market_data: MarketData, 
        current_position: str
    ) -> TradingSignal:
        """Generate exit signal when position exists."""
        if market_data.z_score is None or market_data.exit_threshold is None:
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.0,
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason="Insufficient data for exit signal"
            )
        
        # Check for exit conditions based on current position
        should_exit = False
        exit_reason = ""
        
        if current_position == 'long':
            # Exit long position when spread returns to normal or goes too far negative
            if abs(market_data.z_score) <= market_data.exit_threshold:
                should_exit = True
                exit_reason = f"Long position: Z-score {market_data.z_score:.3f} within exit threshold {market_data.exit_threshold:.3f}"
            elif market_data.z_score < -4.0:  # Stop loss for long
                should_exit = True
                exit_reason = f"Long position stop loss: Z-score {market_data.z_score:.3f} too negative"
        
        elif current_position == 'short':
            # Exit short position when spread returns to normal or goes too far positive
            if abs(market_data.z_score) <= market_data.exit_threshold:
                should_exit = True
                exit_reason = f"Short position: Z-score {market_data.z_score:.3f} within exit threshold {market_data.exit_threshold:.3f}"
            elif market_data.z_score > 4.0:  # Stop loss for short
                should_exit = True
                exit_reason = f"Short position stop loss: Z-score {market_data.z_score:.3f} too positive"
        
        if should_exit:
            signal_strength = self._calculate_signal_strength(
                abs(market_data.z_score), market_data.exit_threshold
            )
            confidence = self._calculate_confidence(market_data, signal_strength)
            
            signal_type = SignalType.STOP_LOSS if "stop loss" in exit_reason else SignalType.EXIT
            
            return TradingSignal(
                pair_id=pair_id,
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence=confidence,
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason=exit_reason
            )
        
        else:
            # Hold position
            return TradingSignal(
                pair_id=pair_id,
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.5,
                current_spread=market_data.spread,
                current_z_score=market_data.z_score,
                hedge_ratio=market_data.hedge_ratio,
                volatility_forecast=market_data.volatility_forecast,
                reason=f"Holding {current_position} position: Z-score {market_data.z_score:.3f}"
            )
    
    def _calculate_signal_strength(self, signal_magnitude: float, threshold: float) -> SignalStrength:
        """Calculate signal strength based on magnitude relative to threshold."""
        if threshold == 0:
            return SignalStrength.WEAK
        
        ratio = signal_magnitude / threshold
        
        if ratio >= 2.0:
            return SignalStrength.VERY_STRONG
        elif ratio >= 1.5:
            return SignalStrength.STRONG
        elif ratio >= 1.2:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    def _calculate_confidence(self, market_data: MarketData, signal_strength: SignalStrength) -> float:
        """Calculate confidence score for the signal."""
        base_confidence = {
            SignalStrength.WEAK: 0.3,
            SignalStrength.MEDIUM: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 0.9
        }[signal_strength]
        
        # Adjust based on data quality
        confidence_adjustments = []
        
        # Hedge ratio confidence
        if market_data.hedge_ratio_confidence:
            confidence_adjustments.append(market_data.hedge_ratio_confidence)
        
        # Volatility regime adjustment
        if market_data.volatility_regime:
            if market_data.volatility_regime == 'normal':
                confidence_adjustments.append(0.8)
            elif market_data.volatility_regime == 'low':
                confidence_adjustments.append(0.9)  # More confident in low vol
            else:  # high volatility
                confidence_adjustments.append(0.6)  # Less confident in high vol
        
        # Apply adjustments
        if confidence_adjustments:
            avg_adjustment = sum(confidence_adjustments) / len(confidence_adjustments)
            base_confidence = (base_confidence + avg_adjustment) / 2
        
        return min(1.0, max(0.0, base_confidence))
    
    def _apply_volatility_adjustment(self, signal: TradingSignal, market_data: MarketData) -> TradingSignal:
        """Apply volatility-based adjustments to the signal."""
        if not self.volatility_adjustment or not market_data.volatility_regime:
            return signal
        
        # Reduce confidence in high volatility environments
        if market_data.volatility_regime == 'high':
            signal.confidence *= 0.8
            signal.reason += " (reduced confidence due to high volatility)"
        elif market_data.volatility_regime == 'low':
            signal.confidence *= 1.1  # Slight boost in low volatility
            signal.confidence = min(1.0, signal.confidence)
        
        return signal
    
    def _apply_risk_adjustment(self, signal: TradingSignal, market_data: MarketData) -> TradingSignal:
        """Apply risk-based adjustments to the signal."""
        if not self.risk_adjustment:
            return signal
        
        # Filter out low-confidence signals
        if signal.confidence < self.min_confidence_threshold:
            signal.signal_type = SignalType.HOLD
            signal.signal_strength = SignalStrength.WEAK
            signal.reason += f" (filtered: confidence {signal.confidence:.2f} below threshold {self.min_confidence_threshold})"
        
        return signal
    
    def get_signal_history(self, pair_id: str, limit: int = 100) -> List[TradingSignal]:
        """Get recent signal history for a pair."""
        if pair_id not in self.signal_history:
            return []
        
        return self.signal_history[pair_id][-limit:]
    
    def get_signal_statistics(self, pair_id: str) -> Dict[str, Any]:
        """Get signal statistics for a pair."""
        if pair_id not in self.signal_history or not self.signal_history[pair_id]:
            return {}
        
        signals = self.signal_history[pair_id]
        
        signal_counts = {}
        for signal_type in SignalType:
            signal_counts[signal_type.value] = sum(
                1 for s in signals if s.signal_type == signal_type
            )
        
        confidences = [s.confidence for s in signals if s.confidence > 0]
        
        return {
            'total_signals': len(signals),
            'signal_counts': signal_counts,
            'avg_confidence': statistics.mean(confidences) if confidences else 0.0,
            'last_signal_time': signals[-1].timestamp.isoformat() if signals else None,
            'spread_stats': self.spread_calculator.get_spread_statistics(pair_id)
        }
