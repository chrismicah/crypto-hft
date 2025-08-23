"""Strategy state machine for managing trading states and transitions."""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass, field
from transitions import Machine
import json

from .signals import TradingSignal, SignalType

logger = structlog.get_logger(__name__)


class TradingState(Enum):
    """Trading states for the strategy state machine."""
    SEARCHING = "searching"      # Looking for entry opportunities
    ENTERING = "entering"        # In process of entering position
    IN_POSITION = "in_position"  # Holding position
    EXITING = "exiting"         # In process of exiting position
    ERROR = "error"             # Error state requiring intervention
    PAUSED = "paused"           # Temporarily paused


class PositionSide(Enum):
    """Position sides."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class PositionInfo:
    """Information about current position."""
    pair_id: str
    side: PositionSide
    entry_time: datetime
    entry_spread: float
    entry_z_score: float
    hedge_ratio: float
    
    # Order information
    asset1_order_id: Optional[str] = None
    asset2_order_id: Optional[str] = None
    asset1_fill_price: Optional[float] = None
    asset2_fill_price: Optional[float] = None
    asset1_quantity: Optional[float] = None
    asset2_quantity: Optional[float] = None
    
    # Performance tracking
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    # Risk management
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    
    # Metadata
    entry_signal: Optional[TradingSignal] = None
    exit_signal: Optional[TradingSignal] = None
    notes: str = ""


@dataclass
class StateTransitionEvent:
    """Event that triggers state transitions."""
    event_type: str
    signal: Optional[TradingSignal] = None
    market_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OrderSizer:
    """Calculates position sizes based on risk parameters."""
    
    def __init__(
        self,
        max_position_size: float = 1000.0,  # Maximum position size in USDT
        risk_per_trade: float = 0.02,       # Risk 2% per trade
        volatility_adjustment: bool = True,
        min_position_size: float = 10.0     # Minimum position size in USDT
    ):
        """
        Initialize order sizer.
        
        Args:
            max_position_size: Maximum position size in quote currency
            risk_per_trade: Maximum risk per trade as fraction of capital
            volatility_adjustment: Whether to adjust size based on volatility
            min_position_size: Minimum position size in quote currency
        """
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.volatility_adjustment = volatility_adjustment
        self.min_position_size = min_position_size
        
    def calculate_position_size(
        self,
        account_balance: float,
        signal: TradingSignal,
        current_volatility: Optional[float] = None,
        stop_loss_distance: Optional[float] = None
    ) -> float:
        """
        Calculate appropriate position size.
        
        Args:
            account_balance: Current account balance
            signal: Trading signal with confidence and strength
            current_volatility: Current volatility forecast
            stop_loss_distance: Distance to stop loss in z-score units
            
        Returns:
            Position size in quote currency (USDT)
        """
        try:
            # Base position size from account balance and risk
            base_size = account_balance * self.risk_per_trade
            
            # Adjust based on signal confidence
            confidence_multiplier = signal.confidence
            adjusted_size = base_size * confidence_multiplier
            
            # Adjust based on signal strength
            strength_multipliers = {
                'weak': 0.5,
                'medium': 0.75,
                'strong': 1.0,
                'very_strong': 1.25
            }
            
            strength_multiplier = strength_multipliers.get(
                signal.signal_strength.value, 0.75
            )
            adjusted_size *= strength_multiplier
            
            # Volatility adjustment
            if self.volatility_adjustment and current_volatility:
                # Reduce size in high volatility environments
                if current_volatility > 0.15:  # High volatility
                    adjusted_size *= 0.7
                elif current_volatility < 0.05:  # Low volatility
                    adjusted_size *= 1.2
            
            # Stop loss adjustment
            if stop_loss_distance and stop_loss_distance > 0:
                # Reduce size if stop loss is far away
                if stop_loss_distance > 3.0:
                    adjusted_size *= 0.8
                elif stop_loss_distance < 1.5:
                    adjusted_size *= 1.1
            
            # Apply bounds
            final_size = max(
                self.min_position_size,
                min(adjusted_size, self.max_position_size)
            )
            
            logger.debug(
                "Position size calculated",
                base_size=base_size,
                confidence_multiplier=confidence_multiplier,
                strength_multiplier=strength_multiplier,
                final_size=final_size,
                signal_confidence=signal.confidence,
                signal_strength=signal.signal_strength.value
            )
            
            return final_size
            
        except Exception as e:
            logger.error("Failed to calculate position size", error=str(e), exc_info=True)
            return self.min_position_size


class StrategyStateMachine:
    """
    State machine for managing trading strategy states and transitions.
    
    States:
    - SEARCHING: Looking for entry opportunities
    - ENTERING: In process of entering position
    - IN_POSITION: Holding position
    - EXITING: In process of exiting position
    - ERROR: Error state requiring intervention
    - PAUSED: Temporarily paused
    """
    
    def __init__(
        self,
        pair_id: str,
        initial_state: TradingState = TradingState.SEARCHING,
        max_position_hold_time: timedelta = timedelta(hours=24),
        entry_timeout: timedelta = timedelta(minutes=5),
        exit_timeout: timedelta = timedelta(minutes=5)
    ):
        """
        Initialize strategy state machine.
        
        Args:
            pair_id: Trading pair identifier
            initial_state: Initial state
            max_position_hold_time: Maximum time to hold a position
            entry_timeout: Timeout for entering positions
            exit_timeout: Timeout for exiting positions
        """
        self.pair_id = pair_id
        self.max_position_hold_time = max_position_hold_time
        self.entry_timeout = entry_timeout
        self.exit_timeout = exit_timeout
        
        # Current state information
        self.current_position: Optional[PositionInfo] = None
        self.pending_orders: List[str] = []
        self.last_signal: Optional[TradingSignal] = None
        self.state_entry_time: datetime = datetime.utcnow()
        
        # State transition history
        self.transition_history: List[StateTransitionEvent] = []
        
        # Order sizer
        self.order_sizer = OrderSizer()
        
        # Define states
        states = [state.value for state in TradingState]
        
        # Define transitions
        transitions = [
            # From SEARCHING
            {
                'trigger': 'entry_signal_received',
                'source': TradingState.SEARCHING.value,
                'dest': TradingState.ENTERING.value,
                'before': 'on_entry_signal'
            },
            {
                'trigger': 'pause',
                'source': TradingState.SEARCHING.value,
                'dest': TradingState.PAUSED.value
            },
            
            # From ENTERING
            {
                'trigger': 'position_entered',
                'source': TradingState.ENTERING.value,
                'dest': TradingState.IN_POSITION.value,
                'before': 'on_position_entered'
            },
            {
                'trigger': 'entry_failed',
                'source': TradingState.ENTERING.value,
                'dest': TradingState.SEARCHING.value,
                'before': 'on_entry_failed'
            },
            {
                'trigger': 'entry_timeout',
                'source': TradingState.ENTERING.value,
                'dest': TradingState.ERROR.value,
                'before': 'on_entry_timeout'
            },
            
            # From IN_POSITION
            {
                'trigger': 'exit_signal_received',
                'source': TradingState.IN_POSITION.value,
                'dest': TradingState.EXITING.value,
                'before': 'on_exit_signal'
            },
            {
                'trigger': 'stop_loss_triggered',
                'source': TradingState.IN_POSITION.value,
                'dest': TradingState.EXITING.value,
                'before': 'on_stop_loss'
            },
            {
                'trigger': 'position_timeout',
                'source': TradingState.IN_POSITION.value,
                'dest': TradingState.EXITING.value,
                'before': 'on_position_timeout'
            },
            
            # From EXITING
            {
                'trigger': 'position_exited',
                'source': TradingState.EXITING.value,
                'dest': TradingState.SEARCHING.value,
                'before': 'on_position_exited'
            },
            {
                'trigger': 'exit_failed',
                'source': TradingState.EXITING.value,
                'dest': TradingState.ERROR.value,
                'before': 'on_exit_failed'
            },
            {
                'trigger': 'exit_timeout',
                'source': TradingState.EXITING.value,
                'dest': TradingState.ERROR.value,
                'before': 'on_exit_timeout'
            },
            
            # From ERROR
            {
                'trigger': 'reset',
                'source': TradingState.ERROR.value,
                'dest': TradingState.SEARCHING.value,
                'before': 'on_reset'
            },
            {
                'trigger': 'pause',
                'source': TradingState.ERROR.value,
                'dest': TradingState.PAUSED.value
            },
            
            # From PAUSED
            {
                'trigger': 'resume',
                'source': TradingState.PAUSED.value,
                'dest': TradingState.SEARCHING.value
            },
            
            # Error transitions from any state
            {
                'trigger': 'error_occurred',
                'source': '*',
                'dest': TradingState.ERROR.value,
                'before': 'on_error'
            }
        ]
        
        # Initialize state machine
        self.machine = Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial=initial_state.value,
            after_state_change='on_state_changed'
        )
        
        logger.info(
            "Strategy state machine initialized",
            pair_id=pair_id,
            initial_state=initial_state.value
        )
    
    def process_signal(self, signal: TradingSignal) -> bool:
        """
        Process a trading signal and trigger appropriate state transitions.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            True if signal was processed successfully
        """
        try:
            self.last_signal = signal
            current_state = TradingState(self.state)
            
            logger.debug(
                "Processing signal",
                pair_id=self.pair_id,
                current_state=current_state.value,
                signal_type=signal.signal_type.value,
                signal_strength=signal.signal_strength.value,
                confidence=signal.confidence
            )
            
            # Process signal based on current state and signal type
            if current_state == TradingState.SEARCHING:
                if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                    return self.entry_signal_received(signal)
            
            elif current_state == TradingState.IN_POSITION:
                if signal.signal_type in [SignalType.EXIT, SignalType.STOP_LOSS]:
                    if signal.signal_type == SignalType.STOP_LOSS:
                        return self.stop_loss_triggered(signal)
                    else:
                        return self.exit_signal_received(signal)
            
            # Signal not applicable to current state
            logger.debug(
                "Signal not applicable to current state",
                pair_id=self.pair_id,
                current_state=current_state.value,
                signal_type=signal.signal_type.value
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to process signal",
                pair_id=self.pair_id,
                error=str(e),
                exc_info=True
            )
            self.error_occurred(error_message=str(e))
            return False
    
    def check_timeouts(self) -> None:
        """Check for state timeouts and trigger appropriate transitions."""
        try:
            current_state = TradingState(self.state)
            time_in_state = datetime.utcnow() - self.state_entry_time
            
            if current_state == TradingState.ENTERING and time_in_state > self.entry_timeout:
                logger.warning(
                    "Entry timeout",
                    pair_id=self.pair_id,
                    time_in_state=time_in_state.total_seconds()
                )
                self.entry_timeout()
            
            elif current_state == TradingState.EXITING and time_in_state > self.exit_timeout:
                logger.warning(
                    "Exit timeout",
                    pair_id=self.pair_id,
                    time_in_state=time_in_state.total_seconds()
                )
                self.exit_timeout()
            
            elif (current_state == TradingState.IN_POSITION and 
                  self.current_position and 
                  datetime.utcnow() - self.current_position.entry_time > self.max_position_hold_time):
                logger.warning(
                    "Position timeout",
                    pair_id=self.pair_id,
                    position_age=(datetime.utcnow() - self.current_position.entry_time).total_seconds()
                )
                self.position_timeout()
                
        except Exception as e:
            logger.error("Error checking timeouts", pair_id=self.pair_id, error=str(e))
            self.error_occurred(error_message=f"Timeout check error: {str(e)}")
    
    def update_position_pnl(self, current_spread: float, current_z_score: float) -> None:
        """Update position PnL and risk metrics."""
        if not self.current_position:
            return
        
        try:
            # Calculate unrealized PnL based on spread movement
            if self.current_position.side == PositionSide.LONG:
                # Long position profits when spread increases (mean reversion)
                pnl_factor = current_spread - self.current_position.entry_spread
            else:
                # Short position profits when spread decreases
                pnl_factor = self.current_position.entry_spread - current_spread
            
            # Estimate PnL (simplified calculation)
            estimated_pnl = pnl_factor * 100  # Scale factor for visualization
            self.current_position.unrealized_pnl = estimated_pnl
            
            # Update max favorable/adverse excursion
            if estimated_pnl > self.current_position.max_favorable_excursion:
                self.current_position.max_favorable_excursion = estimated_pnl
            
            if estimated_pnl < -abs(self.current_position.max_adverse_excursion):
                self.current_position.max_adverse_excursion = abs(estimated_pnl)
            
        except Exception as e:
            logger.error("Failed to update position PnL", error=str(e))
    
    # State transition callbacks
    def on_entry_signal(self, signal: TradingSignal) -> None:
        """Handle entry signal."""
        logger.info(
            "Entry signal received",
            pair_id=self.pair_id,
            signal_type=signal.signal_type.value,
            side=signal.side,
            confidence=signal.confidence
        )
        
        # Create position info
        self.current_position = PositionInfo(
            pair_id=self.pair_id,
            side=PositionSide.LONG if signal.side == 'long' else PositionSide.SHORT,
            entry_time=datetime.utcnow(),
            entry_spread=signal.current_spread or 0.0,
            entry_z_score=signal.current_z_score or 0.0,
            hedge_ratio=signal.hedge_ratio or 1.0,
            entry_signal=signal
        )
    
    def on_position_entered(self) -> None:
        """Handle successful position entry."""
        logger.info(
            "Position entered successfully",
            pair_id=self.pair_id,
            side=self.current_position.side.value if self.current_position else None
        )
    
    def on_entry_failed(self) -> None:
        """Handle failed position entry."""
        logger.warning("Position entry failed", pair_id=self.pair_id)
        self.current_position = None
        self.pending_orders.clear()
    
    def on_entry_timeout(self) -> None:
        """Handle entry timeout."""
        logger.error("Entry timeout occurred", pair_id=self.pair_id)
        self.current_position = None
        self.pending_orders.clear()
    
    def on_exit_signal(self, signal: TradingSignal) -> None:
        """Handle exit signal."""
        logger.info(
            "Exit signal received",
            pair_id=self.pair_id,
            signal_type=signal.signal_type.value,
            reason=signal.reason
        )
        
        if self.current_position:
            self.current_position.exit_signal = signal
    
    def on_stop_loss(self, signal: TradingSignal) -> None:
        """Handle stop loss signal."""
        logger.warning(
            "Stop loss triggered",
            pair_id=self.pair_id,
            reason=signal.reason
        )
        
        if self.current_position:
            self.current_position.exit_signal = signal
            self.current_position.notes += f" STOP_LOSS: {signal.reason}"
    
    def on_position_timeout(self) -> None:
        """Handle position timeout."""
        logger.warning("Position timeout - forcing exit", pair_id=self.pair_id)
        
        if self.current_position:
            self.current_position.notes += " TIMEOUT_EXIT"
    
    def on_position_exited(self) -> None:
        """Handle successful position exit."""
        logger.info(
            "Position exited successfully",
            pair_id=self.pair_id,
            pnl=self.current_position.unrealized_pnl if self.current_position else None
        )
        
        # Archive position info before clearing
        if self.current_position:
            # Could store in database or file for analysis
            pass
        
        self.current_position = None
        self.pending_orders.clear()
    
    def on_exit_failed(self) -> None:
        """Handle failed position exit."""
        logger.error("Position exit failed", pair_id=self.pair_id)
    
    def on_exit_timeout(self) -> None:
        """Handle exit timeout."""
        logger.error("Exit timeout occurred", pair_id=self.pair_id)
    
    def on_error(self, error_message: str = "") -> None:
        """Handle error state."""
        logger.error(
            "State machine error",
            pair_id=self.pair_id,
            error_message=error_message
        )
    
    def on_reset(self) -> None:
        """Handle reset from error state."""
        logger.info("State machine reset", pair_id=self.pair_id)
        self.current_position = None
        self.pending_orders.clear()
    
    def on_state_changed(self) -> None:
        """Handle state change."""
        new_state = TradingState(self.state)
        self.state_entry_time = datetime.utcnow()
        
        # Record transition
        event = StateTransitionEvent(
            event_type=f"transition_to_{new_state.value}",
            signal=self.last_signal,
            timestamp=self.state_entry_time
        )
        self.transition_history.append(event)
        
        # Maintain history size
        if len(self.transition_history) > 1000:
            self.transition_history = self.transition_history[-500:]
        
        logger.info(
            "State changed",
            pair_id=self.pair_id,
            new_state=new_state.value,
            timestamp=self.state_entry_time.isoformat()
        )
    
    def get_current_state(self) -> TradingState:
        """Get current trading state."""
        return TradingState(self.state)
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive state information."""
        current_state = self.get_current_state()
        time_in_state = datetime.utcnow() - self.state_entry_time
        
        info = {
            'pair_id': self.pair_id,
            'current_state': current_state.value,
            'time_in_state_seconds': time_in_state.total_seconds(),
            'state_entry_time': self.state_entry_time.isoformat(),
            'pending_orders': self.pending_orders.copy(),
            'last_signal': {
                'type': self.last_signal.signal_type.value,
                'confidence': self.last_signal.confidence,
                'timestamp': self.last_signal.timestamp.isoformat()
            } if self.last_signal else None,
            'transition_count': len(self.transition_history)
        }
        
        if self.current_position:
            position_age = datetime.utcnow() - self.current_position.entry_time
            info['position'] = {
                'side': self.current_position.side.value,
                'entry_time': self.current_position.entry_time.isoformat(),
                'age_seconds': position_age.total_seconds(),
                'entry_spread': self.current_position.entry_spread,
                'entry_z_score': self.current_position.entry_z_score,
                'hedge_ratio': self.current_position.hedge_ratio,
                'unrealized_pnl': self.current_position.unrealized_pnl,
                'max_favorable_excursion': self.current_position.max_favorable_excursion,
                'max_adverse_excursion': self.current_position.max_adverse_excursion,
                'notes': self.current_position.notes
            }
        
        return info
    
    def can_enter_position(self) -> bool:
        """Check if can enter a new position."""
        return self.get_current_state() == TradingState.SEARCHING
    
    def has_position(self) -> bool:
        """Check if currently holding a position."""
        return self.current_position is not None
    
    def get_position_side(self) -> Optional[PositionSide]:
        """Get current position side."""
        return self.current_position.side if self.current_position else None
