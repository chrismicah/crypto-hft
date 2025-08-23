"""Unit tests for circuit breaker logic."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from services.risk_manager.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, RiskState, HaltReason,
    HaltEvent, RecoveryEvent, ChangePointEvent
)


class TestCircuitBreakerConfig:
    """Test cases for CircuitBreakerConfig."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = CircuitBreakerConfig(
            warning_threshold=0.3,
            halt_threshold=0.6,
            critical_threshold=0.9
        )
        
        assert config.warning_threshold == 0.3
        assert config.halt_threshold == 0.6
        assert config.critical_threshold == 0.9
    
    def test_invalid_threshold_order(self):
        """Test invalid threshold ordering."""
        with pytest.raises(ValueError, match="ascending order"):
            CircuitBreakerConfig(
                warning_threshold=0.8,  # Higher than halt
                halt_threshold=0.6,
                critical_threshold=0.9
            )
    
    def test_invalid_halt_duration(self):
        """Test invalid halt duration configuration."""
        with pytest.raises(ValueError, match="min_halt_duration must be less"):
            CircuitBreakerConfig(
                min_halt_duration=timedelta(hours=3),
                max_halt_duration=timedelta(hours=1)  # Less than min
            )


class TestHaltEvent:
    """Test cases for HaltEvent."""
    
    def test_halt_event_creation(self):
        """Test halt event creation."""
        timestamp = datetime.utcnow()
        
        event = HaltEvent(
            timestamp=timestamp,
            reason=HaltReason.CHANGEPOINT_DETECTED,
            severity="HIGH",
            probability=0.8,
            metadata={'test': 'value'}
        )
        
        assert event.timestamp == timestamp
        assert event.reason == HaltReason.CHANGEPOINT_DETECTED
        assert event.severity == "HIGH"
        assert event.probability == 0.8
        assert event.metadata['test'] == 'value'
    
    def test_kafka_message_conversion(self):
        """Test conversion to Kafka message format."""
        timestamp = datetime.utcnow()
        
        event = HaltEvent(
            timestamp=timestamp,
            reason=HaltReason.CHANGEPOINT_DETECTED,
            severity="CRITICAL",
            probability=0.95
        )
        
        kafka_msg = event.to_kafka_message()
        
        assert kafka_msg['event_type'] == 'TRADING_HALT'
        assert kafka_msg['reason'] == 'CHANGEPOINT_DETECTED'
        assert kafka_msg['severity'] == 'CRITICAL'
        assert kafka_msg['probability'] == 0.95
        assert kafka_msg['timestamp'] == timestamp.isoformat()


class TestRecoveryEvent:
    """Test cases for RecoveryEvent."""
    
    def test_recovery_event_creation(self):
        """Test recovery event creation."""
        timestamp = datetime.utcnow()
        
        event = RecoveryEvent(
            timestamp=timestamp,
            previous_halt_reason=HaltReason.CHANGEPOINT_DETECTED,
            recovery_conditions_met=['condition1', 'condition2']
        )
        
        assert event.timestamp == timestamp
        assert event.previous_halt_reason == HaltReason.CHANGEPOINT_DETECTED
        assert event.recovery_conditions_met == ['condition1', 'condition2']
    
    def test_kafka_message_conversion(self):
        """Test conversion to Kafka message format."""
        timestamp = datetime.utcnow()
        
        event = RecoveryEvent(
            timestamp=timestamp,
            previous_halt_reason=HaltReason.HIGH_VOLATILITY,
            recovery_conditions_met=['volatility_normalized']
        )
        
        kafka_msg = event.to_kafka_message()
        
        assert kafka_msg['event_type'] == 'TRADING_RECOVERY'
        assert kafka_msg['previous_halt_reason'] == 'HIGH_VOLATILITY'
        assert kafka_msg['recovery_conditions'] == ['volatility_normalized']


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CircuitBreakerConfig(
            warning_threshold=0.5,
            halt_threshold=0.7,
            critical_threshold=0.9,
            min_halt_duration=timedelta(minutes=1),
            max_halt_duration=timedelta(minutes=10),
            max_consecutive_warnings=2,
            max_consecutive_high_prob_events=2,
            recovery_probability_threshold=0.2
        )
    
    @pytest.fixture
    def circuit_breaker(self, config):
        """Create circuit breaker instance."""
        halt_callback = Mock()
        recovery_callback = Mock()
        
        return CircuitBreaker(
            config=config,
            halt_callback=halt_callback,
            recovery_callback=recovery_callback
        )
    
    def test_initialization(self, circuit_breaker, config):
        """Test circuit breaker initialization."""
        assert circuit_breaker.config == config
        assert circuit_breaker.current_state == RiskState.NORMAL
        assert circuit_breaker.halt_start_time is None
        assert circuit_breaker.warning_count == 0
        assert circuit_breaker.consecutive_high_prob_count == 0
    
    def test_normal_to_warning_transition(self, circuit_breaker):
        """Test transition from normal to warning state."""
        timestamp = datetime.utcnow()
        
        # Create warning-level event
        event = ChangePointEvent(
            timestamp=timestamp,
            probability=0.6,  # Above warning threshold
            run_length=10,
            data_point=1.0,
            confidence_level="MEDIUM"
        )
        
        halt_event = circuit_breaker.process_changepoint_event(event)
        
        assert halt_event is None  # Should not halt immediately
        assert circuit_breaker.current_state == RiskState.WARNING
    
    def test_critical_probability_immediate_halt(self, circuit_breaker):
        """Test immediate halt on critical probability."""
        timestamp = datetime.utcnow()
        
        # Create critical-level event
        event = ChangePointEvent(
            timestamp=timestamp,
            probability=0.95,  # Above critical threshold
            run_length=5,
            data_point=10.0,
            confidence_level="CRITICAL"
        )
        
        halt_event = circuit_breaker.process_changepoint_event(event)
        
        assert halt_event is not None
        assert halt_event.reason == HaltReason.CHANGEPOINT_DETECTED
        assert halt_event.severity == "CRITICAL"
        assert circuit_breaker.current_state == RiskState.HALT
        assert circuit_breaker.halt_start_time == timestamp
    
    def test_consecutive_high_probability_halt(self, circuit_breaker):
        """Test halt after consecutive high probability events."""
        base_time = datetime.utcnow()
        
        # First high probability event
        event1 = ChangePointEvent(
            timestamp=base_time,
            probability=0.75,  # Above halt threshold
            run_length=10,
            data_point=1.0,
            confidence_level="HIGH"
        )
        
        halt_event1 = circuit_breaker.process_changepoint_event(event1)
        assert halt_event1 is None  # Should not halt on first event
        assert circuit_breaker.current_state == RiskState.WARNING
        assert circuit_breaker.consecutive_high_prob_count == 1
        
        # Second consecutive high probability event
        event2 = ChangePointEvent(
            timestamp=base_time + timedelta(seconds=10),
            probability=0.8,
            run_length=5,
            data_point=2.0,
            confidence_level="HIGH"
        )
        
        halt_event2 = circuit_breaker.process_changepoint_event(event2)
        assert halt_event2 is not None  # Should halt on second consecutive event
        assert halt_event2.reason == HaltReason.CHANGEPOINT_DETECTED
        assert circuit_breaker.current_state == RiskState.HALT
    
    def test_consecutive_warnings_halt(self, circuit_breaker):
        """Test halt after consecutive warning events."""
        base_time = datetime.utcnow()
        
        # Generate consecutive warning events
        for i in range(circuit_breaker.config.max_consecutive_warnings):
            event = ChangePointEvent(
                timestamp=base_time + timedelta(seconds=i * 10),
                probability=0.6,  # Warning level
                run_length=10 - i,
                data_point=float(i),
                confidence_level="MEDIUM"
            )
            
            halt_event = circuit_breaker.process_changepoint_event(event)
            
            if i < circuit_breaker.config.max_consecutive_warnings - 1:
                assert halt_event is None
                assert circuit_breaker.current_state == RiskState.WARNING
            else:
                assert halt_event is not None
                assert circuit_breaker.current_state == RiskState.HALT
    
    def test_low_probability_reset(self, circuit_breaker):
        """Test reset of counters on low probability events."""
        timestamp = datetime.utcnow()
        
        # Generate warning event
        warning_event = ChangePointEvent(
            timestamp=timestamp,
            probability=0.6,
            run_length=10,
            data_point=1.0,
            confidence_level="MEDIUM"
        )
        
        circuit_breaker.process_changepoint_event(warning_event)
        assert circuit_breaker.current_state == RiskState.WARNING
        assert circuit_breaker.warning_count > 0
        
        # Generate low probability event
        low_prob_event = ChangePointEvent(
            timestamp=timestamp + timedelta(seconds=10),
            probability=0.1,  # Below warning threshold
            run_length=15,
            data_point=0.5,
            confidence_level="LOW"
        )
        
        circuit_breaker.process_changepoint_event(low_prob_event)
        assert circuit_breaker.current_state == RiskState.NORMAL
        assert circuit_breaker.warning_count == 0
        assert circuit_breaker.consecutive_high_prob_count == 0
    
    def test_halt_to_recovery_transition(self, circuit_breaker):
        """Test transition from halt to recovery state."""
        base_time = datetime.utcnow()
        
        # Trigger halt
        halt_event = ChangePointEvent(
            timestamp=base_time,
            probability=0.95,
            run_length=5,
            data_point=10.0,
            confidence_level="CRITICAL"
        )
        
        circuit_breaker.process_changepoint_event(halt_event)
        assert circuit_breaker.current_state == RiskState.HALT
        
        # Wait for minimum halt duration and send recovery-level event
        recovery_time = base_time + circuit_breaker.config.min_halt_duration + timedelta(seconds=1)
        
        recovery_event = ChangePointEvent(
            timestamp=recovery_time,
            probability=0.1,  # Below recovery threshold
            run_length=20,
            data_point=1.0,
            confidence_level="LOW"
        )
        
        circuit_breaker.process_changepoint_event(recovery_event)
        assert circuit_breaker.current_state == RiskState.RECOVERY
        assert circuit_breaker.recovery_start_time == recovery_time
    
    def test_recovery_to_normal_transition(self, circuit_breaker):
        """Test successful recovery to normal state."""
        base_time = datetime.utcnow()
        
        # Force into recovery state
        circuit_breaker.current_state = RiskState.RECOVERY
        circuit_breaker.recovery_start_time = base_time
        circuit_breaker.last_halt_reason = HaltReason.CHANGEPOINT_DETECTED
        
        # Send multiple low probability events
        for i in range(5):
            recovery_event = ChangePointEvent(
                timestamp=base_time + timedelta(seconds=i * 10),
                probability=0.15,  # Below recovery threshold
                run_length=25 + i,
                data_point=0.5,
                confidence_level="LOW"
            )
            
            circuit_breaker.process_changepoint_event(recovery_event)
        
        # Should transition to normal after stable low probabilities
        assert circuit_breaker.current_state == RiskState.NORMAL
        assert circuit_breaker.recovery_callback.called
    
    def test_manual_halt(self, circuit_breaker):
        """Test manual halt functionality."""
        reason = "Emergency stop"
        
        halt_event = circuit_breaker.manual_halt(reason)
        
        assert halt_event is not None
        assert halt_event.reason == HaltReason.MANUAL_OVERRIDE
        assert halt_event.severity == "MANUAL"
        assert halt_event.probability == 1.0
        assert halt_event.metadata['reason'] == reason
        assert circuit_breaker.current_state == RiskState.HALT
        assert circuit_breaker.halt_callback.called
    
    def test_manual_recovery(self, circuit_breaker):
        """Test manual recovery functionality."""
        # First put into halt state
        circuit_breaker.manual_halt("Test halt")
        assert circuit_breaker.current_state == RiskState.HALT
        
        # Then recover
        recovery_event = circuit_breaker.manual_recovery()
        
        assert recovery_event is not None
        assert recovery_event.previous_halt_reason == HaltReason.MANUAL_OVERRIDE
        assert "manual_override" in recovery_event.recovery_conditions_met
        assert circuit_breaker.current_state == RiskState.NORMAL
        assert circuit_breaker.recovery_callback.called
    
    def test_manual_recovery_when_not_halted(self, circuit_breaker):
        """Test manual recovery when not in halt state."""
        assert circuit_breaker.current_state == RiskState.NORMAL
        
        recovery_event = circuit_breaker.manual_recovery()
        
        assert recovery_event is None
        assert circuit_breaker.current_state == RiskState.NORMAL
    
    def test_max_halt_duration_enforcement(self, circuit_breaker):
        """Test enforcement of maximum halt duration."""
        base_time = datetime.utcnow()
        
        # Trigger halt
        halt_event = ChangePointEvent(
            timestamp=base_time,
            probability=0.95,
            run_length=5,
            data_point=10.0,
            confidence_level="CRITICAL"
        )
        
        circuit_breaker.process_changepoint_event(halt_event)
        assert circuit_breaker.current_state == RiskState.HALT
        
        # Send event after max halt duration
        max_duration_time = base_time + circuit_breaker.config.max_halt_duration + timedelta(seconds=1)
        
        timeout_event = ChangePointEvent(
            timestamp=max_duration_time,
            probability=0.8,  # Still high probability
            run_length=5,
            data_point=5.0,
            confidence_level="HIGH"
        )
        
        circuit_breaker.process_changepoint_event(timeout_event)
        
        # Should force transition to recovery despite high probability
        assert circuit_breaker.current_state == RiskState.RECOVERY
    
    def test_get_status(self, circuit_breaker):
        """Test status retrieval."""
        status = circuit_breaker.get_status()
        
        assert 'current_state' in status
        assert 'warning_count' in status
        assert 'consecutive_high_prob_count' in status
        assert 'recent_events_count' in status
        assert 'halt_start_time' in status
        assert 'last_halt_reason' in status
        
        assert status['current_state'] == RiskState.NORMAL.value
        assert status['warning_count'] == 0
        assert status['halt_start_time'] is None
    
    def test_volatility_threshold_adjustment(self, circuit_breaker):
        """Test threshold adjustment based on volatility."""
        timestamp = datetime.utcnow()
        
        # Create event with high volatility metadata
        high_vol_event = ChangePointEvent(
            timestamp=timestamp,
            probability=0.6,
            run_length=10,
            data_point=1.0,
            confidence_level="MEDIUM",
            metadata={'recent_std': 3.0}  # High volatility
        )
        
        # Should adjust thresholds for high volatility
        adjusted_config = circuit_breaker._get_adjusted_thresholds(high_vol_event)
        
        # Thresholds should be lower in high volatility
        assert adjusted_config.warning_threshold <= circuit_breaker.config.warning_threshold
        assert adjusted_config.halt_threshold <= circuit_breaker.config.halt_threshold
    
    def test_event_history_management(self, circuit_breaker):
        """Test management of recent events history."""
        base_time = datetime.utcnow()
        
        # Add many events
        for i in range(10):
            event = ChangePointEvent(
                timestamp=base_time + timedelta(seconds=i * 60),
                probability=0.3,
                run_length=10,
                data_point=float(i),
                confidence_level="LOW"
            )
            
            circuit_breaker.process_changepoint_event(event)
        
        assert len(circuit_breaker.recent_events) == 10
        
        # Add event far in the future (should clean old events)
        future_event = ChangePointEvent(
            timestamp=base_time + timedelta(hours=2),  # 2 hours later
            probability=0.4,
            run_length=15,
            data_point=100.0,
            confidence_level="LOW"
        )
        
        circuit_breaker.process_changepoint_event(future_event)
        
        # Old events should be cleaned (only events within last hour kept)
        assert len(circuit_breaker.recent_events) == 1
        assert circuit_breaker.recent_events[0] == future_event


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with realistic scenarios."""
    
    def test_gradual_deterioration_scenario(self):
        """Test gradual deterioration leading to halt."""
        config = CircuitBreakerConfig(
            warning_threshold=0.4,
            halt_threshold=0.6,
            critical_threshold=0.8,
            max_consecutive_warnings=3
        )
        
        halt_callback = Mock()
        circuit_breaker = CircuitBreaker(config, halt_callback=halt_callback)
        
        base_time = datetime.utcnow()
        probabilities = [0.2, 0.3, 0.45, 0.5, 0.55, 0.65]  # Gradual increase
        
        halt_triggered = False
        
        for i, prob in enumerate(probabilities):
            event = ChangePointEvent(
                timestamp=base_time + timedelta(seconds=i * 30),
                probability=prob,
                run_length=20 - i,
                data_point=float(i),
                confidence_level=""
            )
            
            halt_event = circuit_breaker.process_changepoint_event(event)
            
            if halt_event:
                halt_triggered = True
                break
        
        assert halt_triggered
        assert halt_callback.called
        assert circuit_breaker.current_state == RiskState.HALT
    
    def test_false_alarm_recovery_scenario(self):
        """Test recovery from false alarm scenario."""
        config = CircuitBreakerConfig(
            warning_threshold=0.5,
            halt_threshold=0.7,
            critical_threshold=0.9,
            min_halt_duration=timedelta(seconds=30),
            recovery_probability_threshold=0.3
        )
        
        halt_callback = Mock()
        recovery_callback = Mock()
        
        circuit_breaker = CircuitBreaker(
            config,
            halt_callback=halt_callback,
            recovery_callback=recovery_callback
        )
        
        base_time = datetime.utcnow()
        
        # Trigger halt with high probability
        halt_event = ChangePointEvent(
            timestamp=base_time,
            probability=0.95,
            run_length=5,
            data_point=10.0,
            confidence_level="CRITICAL"
        )
        
        circuit_breaker.process_changepoint_event(halt_event)
        assert circuit_breaker.current_state == RiskState.HALT
        
        # Wait for minimum halt duration
        recovery_start_time = base_time + config.min_halt_duration + timedelta(seconds=1)
        
        # Send recovery-level events
        recovery_probabilities = [0.25, 0.2, 0.15, 0.1, 0.05]
        
        for i, prob in enumerate(recovery_probabilities):
            recovery_event = ChangePointEvent(
                timestamp=recovery_start_time + timedelta(seconds=i * 10),
                probability=prob,
                run_length=30 + i,
                data_point=1.0,
                confidence_level="LOW"
            )
            
            circuit_breaker.process_changepoint_event(recovery_event)
        
        # Should have recovered to normal
        assert circuit_breaker.current_state == RiskState.NORMAL
        assert recovery_callback.called
