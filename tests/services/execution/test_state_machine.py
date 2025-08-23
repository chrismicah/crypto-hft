"""Unit tests for strategy state machine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.execution.state_machine import (
    StrategyStateMachine,
    OrderSizer,
    TradingState,
    PositionSide,
    PositionInfo,
    StateTransitionEvent
)
from services.execution.signals import TradingSignal, SignalType, SignalStrength


class TestOrderSizer:
    """Test cases for OrderSizer class."""
    
    def test_initialization(self):
        """Test order sizer initialization."""
        sizer = OrderSizer()
        
        assert sizer.max_position_size == 1000.0
        assert sizer.risk_per_trade == 0.02
        assert sizer.volatility_adjustment is True
        assert sizer.min_position_size == 10.0
    
    def test_initialization_custom_params(self):
        """Test order sizer with custom parameters."""
        sizer = OrderSizer(
            max_position_size=2000.0,
            risk_per_trade=0.01,
            volatility_adjustment=False,
            min_position_size=50.0
        )
        
        assert sizer.max_position_size == 2000.0
        assert sizer.risk_per_trade == 0.01
        assert sizer.volatility_adjustment is False
        assert sizer.min_position_size == 50.0
    
    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        sizer = OrderSizer(volatility_adjustment=False)
        
        # Create mock signal
        signal = Mock()
        signal.confidence = 0.8
        signal.signal_strength.value = 'strong'
        
        account_balance = 10000.0
        
        position_size = sizer.calculate_position_size(account_balance, signal)
        
        # Expected: 10000 * 0.02 * 0.8 * 1.0 = 160
        expected_size = 10000 * 0.02 * 0.8 * 1.0
        assert position_size == expected_size
    
    def test_calculate_position_size_with_bounds(self):
        """Test position size calculation with bounds."""
        sizer = OrderSizer(
            max_position_size=100.0,
            min_position_size=50.0,
            volatility_adjustment=False
        )
        
        # Test maximum bound
        signal_high = Mock()
        signal_high.confidence = 1.0
        signal_high.signal_strength.value = 'very_strong'
        
        position_size = sizer.calculate_position_size(100000.0, signal_high)
        assert position_size == 100.0  # Should be capped at max
        
        # Test minimum bound
        signal_low = Mock()
        signal_low.confidence = 0.1
        signal_low.signal_strength.value = 'weak'
        
        position_size = sizer.calculate_position_size(100.0, signal_low)
        assert position_size == 50.0  # Should be at minimum
    
    def test_calculate_position_size_volatility_adjustment(self):
        """Test position size with volatility adjustment."""
        sizer = OrderSizer(volatility_adjustment=True)
        
        signal = Mock()
        signal.confidence = 0.8
        signal.signal_strength.value = 'medium'
        
        account_balance = 10000.0
        
        # High volatility should reduce size
        high_vol_size = sizer.calculate_position_size(
            account_balance, signal, current_volatility=0.2
        )
        
        # Normal volatility
        normal_vol_size = sizer.calculate_position_size(
            account_balance, signal, current_volatility=0.1
        )
        
        # Low volatility should increase size
        low_vol_size = sizer.calculate_position_size(
            account_balance, signal, current_volatility=0.03
        )
        
        assert high_vol_size < normal_vol_size < low_vol_size
    
    def test_calculate_position_size_stop_loss_adjustment(self):
        """Test position size with stop loss distance adjustment."""
        sizer = OrderSizer(volatility_adjustment=False)
        
        signal = Mock()
        signal.confidence = 0.8
        signal.signal_strength.value = 'medium'
        
        account_balance = 10000.0
        
        # Far stop loss should reduce size
        far_stop_size = sizer.calculate_position_size(
            account_balance, signal, stop_loss_distance=4.0
        )
        
        # Close stop loss should increase size
        close_stop_size = sizer.calculate_position_size(
            account_balance, signal, stop_loss_distance=1.0
        )
        
        assert far_stop_size < close_stop_size


class TestStrategyStateMachine:
    """Test cases for StrategyStateMachine class."""
    
    def test_initialization(self):
        """Test state machine initialization."""
        sm = StrategyStateMachine("BTCETH")
        
        assert sm.pair_id == "BTCETH"
        assert sm.get_current_state() == TradingState.SEARCHING
        assert sm.current_position is None
        assert len(sm.pending_orders) == 0
        assert len(sm.transition_history) == 0
    
    def test_initialization_custom_state(self):
        """Test state machine with custom initial state."""
        sm = StrategyStateMachine("BTCETH", initial_state=TradingState.PAUSED)
        
        assert sm.get_current_state() == TradingState.PAUSED
    
    def test_valid_state_transitions(self):
        """Test valid state transitions."""
        sm = StrategyStateMachine("BTCETH")
        
        # SEARCHING -> ENTERING
        assert sm.get_current_state() == TradingState.SEARCHING
        
        # Create entry signal
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long'
        )
        
        success = sm.process_signal(entry_signal)
        assert success is True
        assert sm.get_current_state() == TradingState.ENTERING
        assert sm.current_position is not None
        assert sm.current_position.side == PositionSide.LONG
        
        # ENTERING -> IN_POSITION
        sm.position_entered()
        assert sm.get_current_state() == TradingState.IN_POSITION
        
        # IN_POSITION -> EXITING
        exit_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.EXIT,
            signal_strength=SignalStrength.MEDIUM,
            confidence=0.7
        )
        
        success = sm.process_signal(exit_signal)
        assert success is True
        assert sm.get_current_state() == TradingState.EXITING
        
        # EXITING -> SEARCHING
        sm.position_exited()
        assert sm.get_current_state() == TradingState.SEARCHING
        assert sm.current_position is None
    
    def test_invalid_state_transitions(self):
        """Test that invalid state transitions are handled properly."""
        sm = StrategyStateMachine("BTCETH")
        
        # Try to process exit signal while searching (should be ignored)
        exit_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.EXIT,
            signal_strength=SignalStrength.MEDIUM,
            confidence=0.7
        )
        
        success = sm.process_signal(exit_signal)
        assert success is True  # Signal processed but ignored
        assert sm.get_current_state() == TradingState.SEARCHING  # State unchanged
    
    def test_entry_signal_processing(self):
        """Test entry signal processing."""
        sm = StrategyStateMachine("BTCETH")
        
        # Test long entry
        long_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long',
            current_spread=0.05,
            current_z_score=-2.5,
            hedge_ratio=15.0
        )
        
        sm.process_signal(long_signal)
        
        assert sm.get_current_state() == TradingState.ENTERING
        assert sm.current_position.side == PositionSide.LONG
        assert sm.current_position.entry_spread == 0.05
        assert sm.current_position.entry_z_score == -2.5
        assert sm.current_position.hedge_ratio == 15.0
        assert sm.current_position.entry_signal == long_signal
        
        # Reset for short entry test
        sm.reset()
        
        # Test short entry
        short_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.MEDIUM,
            confidence=0.7,
            side='short'
        )
        
        sm.process_signal(short_signal)
        
        assert sm.get_current_state() == TradingState.ENTERING
        assert sm.current_position.side == PositionSide.SHORT
    
    def test_stop_loss_signal_processing(self):
        """Test stop loss signal processing."""
        sm = StrategyStateMachine("BTCETH")
        
        # Enter position first
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long'
        )
        
        sm.process_signal(entry_signal)
        sm.position_entered()
        
        assert sm.get_current_state() == TradingState.IN_POSITION
        
        # Process stop loss signal
        stop_loss_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.STOP_LOSS,
            signal_strength=SignalStrength.VERY_STRONG,
            confidence=1.0,
            reason="Stop loss triggered: Z-score -4.5 too negative"
        )
        
        sm.process_signal(stop_loss_signal)
        
        assert sm.get_current_state() == TradingState.EXITING
        assert sm.current_position.exit_signal == stop_loss_signal
        assert "STOP_LOSS" in sm.current_position.notes
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        sm = StrategyStateMachine(
            "BTCETH",
            entry_timeout=timedelta(seconds=1),
            exit_timeout=timedelta(seconds=1),
            max_position_hold_time=timedelta(seconds=2)
        )
        
        # Test entry timeout
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long'
        )
        
        sm.process_signal(entry_signal)
        assert sm.get_current_state() == TradingState.ENTERING
        
        # Simulate timeout by setting entry time in the past
        sm.state_entry_time = datetime.utcnow() - timedelta(seconds=2)
        
        sm.check_timeouts()
        assert sm.get_current_state() == TradingState.ERROR
    
    def test_position_pnl_update(self):
        """Test position PnL updates."""
        sm = StrategyStateMachine("BTCETH")
        
        # Enter long position
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long',
            current_spread=0.05
        )
        
        sm.process_signal(entry_signal)
        sm.position_entered()
        
        # Update PnL with favorable movement
        sm.update_position_pnl(current_spread=0.08, current_z_score=-1.0)
        
        assert sm.current_position.unrealized_pnl > 0  # Profitable
        assert sm.current_position.max_favorable_excursion > 0
        
        # Update PnL with adverse movement
        sm.update_position_pnl(current_spread=0.02, current_z_score=-3.0)
        
        assert sm.current_position.unrealized_pnl < 0  # Loss
        assert sm.current_position.max_adverse_excursion > 0
    
    def test_error_handling(self):
        """Test error state handling."""
        sm = StrategyStateMachine("BTCETH")
        
        # Trigger error
        sm.error_occurred(error_message="Test error")
        
        assert sm.get_current_state() == TradingState.ERROR
        
        # Reset from error
        sm.reset()
        
        assert sm.get_current_state() == TradingState.SEARCHING
        assert sm.current_position is None
        assert len(sm.pending_orders) == 0
    
    def test_pause_resume(self):
        """Test pause and resume functionality."""
        sm = StrategyStateMachine("BTCETH")
        
        # Pause from searching
        sm.pause()
        assert sm.get_current_state() == TradingState.PAUSED
        
        # Resume
        sm.resume()
        assert sm.get_current_state() == TradingState.SEARCHING
    
    def test_state_info(self):
        """Test state information retrieval."""
        sm = StrategyStateMachine("BTCETH")
        
        info = sm.get_state_info()
        
        required_keys = [
            'pair_id', 'current_state', 'time_in_state_seconds',
            'state_entry_time', 'pending_orders', 'last_signal',
            'transition_count'
        ]
        
        for key in required_keys:
            assert key in info
        
        assert info['pair_id'] == "BTCETH"
        assert info['current_state'] == TradingState.SEARCHING.value
        assert info['transition_count'] == 0
        
        # Test with position
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long'
        )
        
        sm.process_signal(entry_signal)
        sm.position_entered()
        
        info = sm.get_state_info()
        
        assert 'position' in info
        assert info['position']['side'] == PositionSide.LONG.value
        assert 'entry_time' in info['position']
        assert 'age_seconds' in info['position']
    
    def test_utility_methods(self):
        """Test utility methods."""
        sm = StrategyStateMachine("BTCETH")
        
        # Initially can enter position
        assert sm.can_enter_position() is True
        assert sm.has_position() is False
        assert sm.get_position_side() is None
        
        # Enter position
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long'
        )
        
        sm.process_signal(entry_signal)
        sm.position_entered()
        
        # Now has position
        assert sm.can_enter_position() is False
        assert sm.has_position() is True
        assert sm.get_position_side() == PositionSide.LONG
    
    def test_transition_history(self):
        """Test transition history tracking."""
        sm = StrategyStateMachine("BTCETH")
        
        initial_count = len(sm.transition_history)
        
        # Make some transitions
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long'
        )
        
        sm.process_signal(entry_signal)  # SEARCHING -> ENTERING
        sm.position_entered()            # ENTERING -> IN_POSITION
        
        # Should have recorded transitions
        assert len(sm.transition_history) > initial_count
        
        # Check transition events
        for event in sm.transition_history:
            assert isinstance(event, StateTransitionEvent)
            assert event.event_type.startswith('transition_to_')
            assert isinstance(event.timestamp, datetime)


class TestPositionInfo:
    """Test cases for PositionInfo dataclass."""
    
    def test_creation(self):
        """Test creating PositionInfo."""
        entry_time = datetime.utcnow()
        
        position = PositionInfo(
            pair_id="BTCETH",
            side=PositionSide.LONG,
            entry_time=entry_time,
            entry_spread=0.05,
            entry_z_score=-2.0,
            hedge_ratio=15.0
        )
        
        assert position.pair_id == "BTCETH"
        assert position.side == PositionSide.LONG
        assert position.entry_time == entry_time
        assert position.entry_spread == 0.05
        assert position.entry_z_score == -2.0
        assert position.hedge_ratio == 15.0
        
        # Check defaults
        assert position.unrealized_pnl == 0.0
        assert position.max_favorable_excursion == 0.0
        assert position.max_adverse_excursion == 0.0
        assert position.notes == ""


@pytest.fixture
def sample_state_machine():
    """Fixture providing a sample state machine."""
    return StrategyStateMachine("BTCETH")


@pytest.fixture
def sample_entry_signal():
    """Fixture providing a sample entry signal."""
    return TradingSignal(
        pair_id="BTCETH",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        confidence=0.8,
        side='long',
        current_spread=0.05,
        current_z_score=-2.5,
        hedge_ratio=15.0
    )


class TestStateMachineIntegration:
    """Integration tests for state machine components."""
    
    def test_complete_trading_cycle(self, sample_state_machine, sample_entry_signal):
        """Test complete trading cycle from entry to exit."""
        sm = sample_state_machine
        
        # Start in searching state
        assert sm.get_current_state() == TradingState.SEARCHING
        
        # Process entry signal
        sm.process_signal(sample_entry_signal)
        assert sm.get_current_state() == TradingState.ENTERING
        assert sm.has_position()
        
        # Simulate successful entry
        sm.position_entered()
        assert sm.get_current_state() == TradingState.IN_POSITION
        
        # Update PnL during position hold
        sm.update_position_pnl(0.08, -1.0)  # Favorable movement
        assert sm.current_position.unrealized_pnl > 0
        
        # Process exit signal
        exit_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.EXIT,
            signal_strength=SignalStrength.MEDIUM,
            confidence=0.7,
            reason="Mean reversion complete"
        )
        
        sm.process_signal(exit_signal)
        assert sm.get_current_state() == TradingState.EXITING
        
        # Simulate successful exit
        sm.position_exited()
        assert sm.get_current_state() == TradingState.SEARCHING
        assert not sm.has_position()
        
        # Check transition history
        assert len(sm.transition_history) >= 4  # At least 4 transitions
    
    def test_error_recovery_cycle(self, sample_state_machine):
        """Test error handling and recovery."""
        sm = sample_state_machine
        
        # Trigger error from any state
        sm.error_occurred("Simulated error")
        assert sm.get_current_state() == TradingState.ERROR
        
        # Reset and recover
        sm.reset()
        assert sm.get_current_state() == TradingState.SEARCHING
        assert not sm.has_position()
        
        # Should be able to trade normally after reset
        entry_signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.MEDIUM,
            confidence=0.7,
            side='short'
        )
        
        sm.process_signal(entry_signal)
        assert sm.get_current_state() == TradingState.ENTERING
        assert sm.current_position.side == PositionSide.SHORT
    
    def test_multiple_signal_processing(self, sample_state_machine):
        """Test processing multiple signals in sequence."""
        sm = sample_state_machine
        
        signals = [
            TradingSignal(
                pair_id="BTCETH",
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.3
            ),
            TradingSignal(
                pair_id="BTCETH",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                confidence=0.8,
                side='long'
            ),
            TradingSignal(
                pair_id="BTCETH",
                signal_type=SignalType.HOLD,
                signal_strength=SignalStrength.WEAK,
                confidence=0.4
            )
        ]
        
        # Process signals
        for signal in signals:
            sm.process_signal(signal)
        
        # Should be in ENTERING state from the entry signal
        assert sm.get_current_state() == TradingState.ENTERING
        assert sm.has_position()
        
        # Last signal should be stored
        assert sm.last_signal == signals[-1]
