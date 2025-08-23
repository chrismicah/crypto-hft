"""Circuit breaker logic for risk management based on changepoint detection."""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from .bocd import ChangePointEvent, BOCDWrapper
from common.logger import get_logger

logger = get_logger(__name__)


class RiskState(Enum):
    """Risk management states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALT = "HALT"
    RECOVERY = "RECOVERY"


class HaltReason(Enum):
    """Reasons for trading halt."""
    CHANGEPOINT_DETECTED = "CHANGEPOINT_DETECTED"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class HaltEvent:
    """Represents a trading halt event."""
    timestamp: datetime
    reason: HaltReason
    severity: str
    probability: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """Convert to Kafka message format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': 'TRADING_HALT',
            'reason': self.reason.value,
            'severity': self.severity,
            'probability': self.probability,
            'metadata': self.metadata
        }


@dataclass
class RecoveryEvent:
    """Represents a recovery from halt event."""
    timestamp: datetime
    previous_halt_reason: HaltReason
    recovery_conditions_met: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """Convert to Kafka message format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': 'TRADING_RECOVERY',
            'previous_halt_reason': self.previous_halt_reason.value,
            'recovery_conditions': self.recovery_conditions_met,
            'metadata': self.metadata
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Changepoint detection thresholds
    warning_threshold: float = 0.5
    halt_threshold: float = 0.7
    critical_threshold: float = 0.9
    
    # Time-based parameters
    min_halt_duration: timedelta = timedelta(minutes=5)
    max_halt_duration: timedelta = timedelta(hours=2)
    recovery_observation_period: timedelta = timedelta(minutes=10)
    
    # Consecutive event thresholds
    max_consecutive_warnings: int = 3
    max_consecutive_high_prob_events: int = 2
    
    # Recovery conditions
    recovery_probability_threshold: float = 0.2
    recovery_stability_period: timedelta = timedelta(minutes=5)
    
    # Volatility-based adjustments
    enable_volatility_adjustment: bool = True
    high_volatility_multiplier: float = 0.8  # Lower thresholds in high vol
    low_volatility_multiplier: float = 1.2   # Higher thresholds in low vol
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0 < self.warning_threshold < self.halt_threshold < self.critical_threshold <= 1.0):
            raise ValueError("Thresholds must be in ascending order: 0 < warning < halt < critical <= 1.0")
        
        if self.min_halt_duration >= self.max_halt_duration:
            raise ValueError("min_halt_duration must be less than max_halt_duration")


class CircuitBreaker:
    """
    Circuit breaker for trading halt based on changepoint detection and other risk factors.
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        halt_callback: Optional[Callable[[HaltEvent], None]] = None,
        recovery_callback: Optional[Callable[[RecoveryEvent], None]] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            halt_callback: Callback function for halt events
            recovery_callback: Callback function for recovery events
        """
        self.config = config
        self.halt_callback = halt_callback
        self.recovery_callback = recovery_callback
        
        # State tracking
        self.current_state = RiskState.NORMAL
        self.halt_start_time: Optional[datetime] = None
        self.last_halt_reason: Optional[HaltReason] = None
        
        # Event history
        self.recent_events: List[ChangePointEvent] = []
        self.warning_count = 0
        self.consecutive_high_prob_count = 0
        
        # Recovery tracking
        self.recovery_start_time: Optional[datetime] = None
        self.recovery_observations: List[float] = []
        
        logger.info("Circuit breaker initialized",
                   warning_threshold=config.warning_threshold,
                   halt_threshold=config.halt_threshold,
                   critical_threshold=config.critical_threshold)
    
    def process_changepoint_event(self, event: ChangePointEvent) -> Optional[HaltEvent]:
        """
        Process a changepoint event and determine if action is needed.
        
        Args:
            event: Changepoint event from BOCD
            
        Returns:
            HaltEvent if halt is triggered, None otherwise
        """
        # Store event
        self.recent_events.append(event)
        
        # Keep only recent events (last hour)
        cutoff_time = event.timestamp - timedelta(hours=1)
        self.recent_events = [e for e in self.recent_events if e.timestamp >= cutoff_time]
        
        # Adjust thresholds based on volatility if enabled
        adjusted_config = self._get_adjusted_thresholds(event)
        
        logger.debug("Processing changepoint event",
                    probability=event.probability,
                    confidence=event.confidence_level,
                    current_state=self.current_state.value,
                    warning_threshold=adjusted_config.warning_threshold,
                    halt_threshold=adjusted_config.halt_threshold)
        
        # State machine logic
        if self.current_state == RiskState.NORMAL:
            return self._handle_normal_state(event, adjusted_config)
        elif self.current_state == RiskState.WARNING:
            return self._handle_warning_state(event, adjusted_config)
        elif self.current_state == RiskState.HALT:
            return self._handle_halt_state(event, adjusted_config)
        elif self.current_state == RiskState.RECOVERY:
            return self._handle_recovery_state(event, adjusted_config)
        
        return None
    
    def _handle_normal_state(self, event: ChangePointEvent, config: CircuitBreakerConfig) -> Optional[HaltEvent]:
        """Handle event in normal state."""
        if event.probability >= config.critical_threshold:
            # Immediate halt for critical probability
            return self._trigger_halt(event, HaltReason.CHANGEPOINT_DETECTED, "CRITICAL")
        
        elif event.probability >= config.halt_threshold:
            # Check for consecutive high probability events
            self.consecutive_high_prob_count += 1
            
            if self.consecutive_high_prob_count >= config.max_consecutive_high_prob_events:
                return self._trigger_halt(event, HaltReason.CHANGEPOINT_DETECTED, "HIGH")
            else:
                self._transition_to_warning()
                return None
        
        elif event.probability >= config.warning_threshold:
            self._transition_to_warning()
            return None
        
        else:
            # Reset counters for low probability events
            self.warning_count = 0
            self.consecutive_high_prob_count = 0
            return None
    
    def _handle_warning_state(self, event: ChangePointEvent, config: CircuitBreakerConfig) -> Optional[HaltEvent]:
        """Handle event in warning state."""
        if event.probability >= config.critical_threshold:
            return self._trigger_halt(event, HaltReason.CHANGEPOINT_DETECTED, "CRITICAL")
        
        elif event.probability >= config.halt_threshold:
            self.consecutive_high_prob_count += 1
            
            if self.consecutive_high_prob_count >= config.max_consecutive_high_prob_events:
                return self._trigger_halt(event, HaltReason.CHANGEPOINT_DETECTED, "HIGH")
        
        elif event.probability >= config.warning_threshold:
            self.warning_count += 1
            
            if self.warning_count >= config.max_consecutive_warnings:
                return self._trigger_halt(event, HaltReason.CHANGEPOINT_DETECTED, "MEDIUM")
        
        else:
            # Low probability - return to normal
            self._transition_to_normal()
        
        return None
    
    def _handle_halt_state(self, event: ChangePointEvent, config: CircuitBreakerConfig) -> Optional[HaltEvent]:
        """Handle event in halt state."""
        # Check if minimum halt duration has passed
        if (self.halt_start_time and 
            event.timestamp - self.halt_start_time >= config.min_halt_duration):
            
            # Check recovery conditions
            if self._check_recovery_conditions(event, config):
                self._transition_to_recovery(event.timestamp)
        
        # Check for maximum halt duration
        if (self.halt_start_time and 
            event.timestamp - self.halt_start_time >= config.max_halt_duration):
            
            logger.warning("Maximum halt duration reached, forcing recovery",
                          halt_duration=(event.timestamp - self.halt_start_time).total_seconds())
            self._transition_to_recovery(event.timestamp)
        
        return None
    
    def _handle_recovery_state(self, event: ChangePointEvent, config: CircuitBreakerConfig) -> Optional[HaltEvent]:
        """Handle event in recovery state."""
        # Add observation to recovery tracking
        self.recovery_observations.append(event.probability)
        
        # Keep only recent observations
        cutoff_time = event.timestamp - config.recovery_stability_period
        if self.recovery_start_time and cutoff_time > self.recovery_start_time:
            # Remove old observations
            observation_age = (event.timestamp - self.recovery_start_time).total_seconds()
            stability_period_seconds = config.recovery_stability_period.total_seconds()
            
            if observation_age > stability_period_seconds:
                keep_count = int(len(self.recovery_observations) * 
                               stability_period_seconds / observation_age)
                self.recovery_observations = self.recovery_observations[-keep_count:]
        
        # Check if recovery conditions are still met
        if len(self.recovery_observations) >= 3:  # Minimum observations for stability check
            recent_max = max(self.recovery_observations[-5:])  # Last 5 observations
            recent_mean = sum(self.recovery_observations[-10:]) / min(10, len(self.recovery_observations))
            
            if (recent_max <= config.recovery_probability_threshold and
                recent_mean <= config.recovery_probability_threshold * 1.5):
                
                # Recovery successful
                self._transition_to_normal()
                
                if self.recovery_callback:
                    recovery_event = RecoveryEvent(
                        timestamp=event.timestamp,
                        previous_halt_reason=self.last_halt_reason,
                        recovery_conditions_met=[
                            f"max_probability_below_{config.recovery_probability_threshold}",
                            f"mean_probability_stable",
                            f"stability_period_{config.recovery_stability_period}"
                        ],
                        metadata={
                            'recovery_duration_seconds': (event.timestamp - self.recovery_start_time).total_seconds(),
                            'recovery_observations': len(self.recovery_observations),
                            'final_max_probability': recent_max,
                            'final_mean_probability': recent_mean
                        }
                    )
                    self.recovery_callback(recovery_event)
            
            elif recent_max > config.halt_threshold:
                # Recovery failed - back to halt
                return self._trigger_halt(event, HaltReason.CHANGEPOINT_DETECTED, "RECOVERY_FAILED")
        
        return None
    
    def _trigger_halt(self, event: ChangePointEvent, reason: HaltReason, severity: str) -> HaltEvent:
        """Trigger a trading halt."""
        self.current_state = RiskState.HALT
        self.halt_start_time = event.timestamp
        self.last_halt_reason = reason
        
        # Reset counters
        self.warning_count = 0
        self.consecutive_high_prob_count = 0
        
        halt_event = HaltEvent(
            timestamp=event.timestamp,
            reason=reason,
            severity=severity,
            probability=event.probability,
            metadata={
                'changepoint_confidence': event.confidence_level,
                'run_length': event.run_length,
                'data_point': event.data_point,
                'recent_events_count': len(self.recent_events),
                'warning_count': self.warning_count,
                'consecutive_high_prob_count': self.consecutive_high_prob_count
            }
        )
        
        logger.critical("Trading halt triggered",
                       reason=reason.value,
                       severity=severity,
                       probability=event.probability,
                       confidence=event.confidence_level)
        
        if self.halt_callback:
            self.halt_callback(halt_event)
        
        return halt_event
    
    def _transition_to_warning(self) -> None:
        """Transition to warning state."""
        if self.current_state != RiskState.WARNING:
            logger.warning("Transitioning to WARNING state")
            self.current_state = RiskState.WARNING
    
    def _transition_to_normal(self) -> None:
        """Transition to normal state."""
        if self.current_state != RiskState.NORMAL:
            logger.info("Transitioning to NORMAL state")
            self.current_state = RiskState.NORMAL
            self.warning_count = 0
            self.consecutive_high_prob_count = 0
            self.halt_start_time = None
            self.recovery_start_time = None
            self.recovery_observations = []
    
    def _transition_to_recovery(self, timestamp: datetime) -> None:
        """Transition to recovery state."""
        logger.info("Transitioning to RECOVERY state")
        self.current_state = RiskState.RECOVERY
        self.recovery_start_time = timestamp
        self.recovery_observations = []
    
    def _check_recovery_conditions(self, event: ChangePointEvent, config: CircuitBreakerConfig) -> bool:
        """Check if conditions are met for recovery."""
        # Simple recovery condition: probability below threshold
        return event.probability <= config.recovery_probability_threshold
    
    def _get_adjusted_thresholds(self, event: ChangePointEvent) -> CircuitBreakerConfig:
        """Get thresholds adjusted for current market conditions."""
        if not self.config.enable_volatility_adjustment:
            return self.config
        
        # Calculate recent volatility from metadata
        recent_std = event.metadata.get('recent_std', 1.0)
        
        # Simple volatility classification
        if recent_std > 2.0:  # High volatility
            multiplier = self.config.high_volatility_multiplier
        elif recent_std < 0.5:  # Low volatility
            multiplier = self.config.low_volatility_multiplier
        else:
            multiplier = 1.0
        
        # Create adjusted config
        adjusted_config = CircuitBreakerConfig(
            warning_threshold=self.config.warning_threshold * multiplier,
            halt_threshold=self.config.halt_threshold * multiplier,
            critical_threshold=self.config.critical_threshold * multiplier,
            min_halt_duration=self.config.min_halt_duration,
            max_halt_duration=self.config.max_halt_duration,
            recovery_observation_period=self.config.recovery_observation_period,
            max_consecutive_warnings=self.config.max_consecutive_warnings,
            max_consecutive_high_prob_events=self.config.max_consecutive_high_prob_events,
            recovery_probability_threshold=self.config.recovery_probability_threshold,
            recovery_stability_period=self.config.recovery_stability_period
        )
        
        # Ensure thresholds stay within bounds
        adjusted_config.warning_threshold = max(0.1, min(0.9, adjusted_config.warning_threshold))
        adjusted_config.halt_threshold = max(0.2, min(0.95, adjusted_config.halt_threshold))
        adjusted_config.critical_threshold = max(0.3, min(1.0, adjusted_config.critical_threshold))
        
        return adjusted_config
    
    def manual_halt(self, reason: str = "Manual override") -> HaltEvent:
        """Manually trigger a halt."""
        halt_event = HaltEvent(
            timestamp=datetime.utcnow(),
            reason=HaltReason.MANUAL_OVERRIDE,
            severity="MANUAL",
            probability=1.0,
            metadata={'reason': reason}
        )
        
        self.current_state = RiskState.HALT
        self.halt_start_time = halt_event.timestamp
        self.last_halt_reason = HaltReason.MANUAL_OVERRIDE
        
        logger.warning("Manual halt triggered", reason=reason)
        
        if self.halt_callback:
            self.halt_callback(halt_event)
        
        return halt_event
    
    def manual_recovery(self) -> Optional[RecoveryEvent]:
        """Manually trigger recovery from halt."""
        if self.current_state != RiskState.HALT:
            logger.warning("Cannot manually recover - not in halt state")
            return None
        
        recovery_event = RecoveryEvent(
            timestamp=datetime.utcnow(),
            previous_halt_reason=self.last_halt_reason,
            recovery_conditions_met=["manual_override"],
            metadata={'type': 'manual_recovery'}
        )
        
        self._transition_to_normal()
        
        logger.info("Manual recovery triggered")
        
        if self.recovery_callback:
            self.recovery_callback(recovery_event)
        
        return recovery_event
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        status = {
            'current_state': self.current_state.value,
            'warning_count': self.warning_count,
            'consecutive_high_prob_count': self.consecutive_high_prob_count,
            'recent_events_count': len(self.recent_events),
            'halt_start_time': self.halt_start_time.isoformat() if self.halt_start_time else None,
            'last_halt_reason': self.last_halt_reason.value if self.last_halt_reason else None,
            'recovery_start_time': self.recovery_start_time.isoformat() if self.recovery_start_time else None,
            'recovery_observations_count': len(self.recovery_observations)
        }
        
        if self.halt_start_time:
            status['halt_duration_seconds'] = (datetime.utcnow() - self.halt_start_time).total_seconds()
        
        if self.recovery_observations:
            status['recent_recovery_probability'] = self.recovery_observations[-1]
            status['recovery_probability_trend'] = (
                self.recovery_observations[-1] - self.recovery_observations[0]
                if len(self.recovery_observations) > 1 else 0
            )
        
        return status
