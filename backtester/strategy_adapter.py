"""Strategy adapter to run existing strategy code in backtesting engine."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from .engine import BacktestEngine
from .models import BacktestEvent, EventType, OrderSide, OrderType
from services.execution.signals import SignalGenerator, MarketData, TradingSignal, SignalType
from services.execution.state_machine import StrategyStateMachine, TradingState, PositionSide
from common.logger import get_logger

logger = get_logger(__name__)


class BacktestStrategyAdapter:
    """Adapter to run existing strategy logic in backtesting engine."""
    
    def __init__(self, symbols: List[str]):
        """
        Initialize strategy adapter.
        
        Args:
            symbols: List of symbols to trade
        """
        self.symbols = symbols
        self.signal_generator = SignalGenerator()
        self.state_machines: Dict[str, StrategyStateMachine] = {}
        
        # Initialize state machines for each trading pair
        self.trading_pairs = [
            ('BTC', 'ETH'),
            ('BTC', 'ADA'),
            ('ETH', 'ADA')
        ]
        
        for asset1, asset2 in self.trading_pairs:
            pair_id = f"{asset1}{asset2}"
            self.state_machines[pair_id] = StrategyStateMachine(pair_id)
        
        # Data aggregation
        self.market_data: Dict[str, MarketData] = {}
        self.hedge_ratios: Dict[str, float] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        
        logger.info("Strategy adapter initialized", 
                   symbols=symbols, 
                   trading_pairs=self.trading_pairs)
    
    async def on_event(self, engine: BacktestEngine, event: BacktestEvent) -> None:
        """Handle backtesting events."""
        try:
            if event.event_type == EventType.MARKET_DATA:
                await self._handle_market_data(engine, event)
            elif event.event_type == EventType.ORDER_FILLED:
                await self._handle_order_filled(engine, event)
            
        except Exception as e:
            logger.error("Error in strategy adapter", error=str(e), exc_info=True)
    
    async def _handle_market_data(self, engine: BacktestEngine, event: BacktestEvent) -> None:
        """Handle market data events."""
        symbol = event.symbol
        
        # Update market data
        order_book = getattr(event, 'order_book', None)
        trade = getattr(event, 'trade', None)
        
        if order_book:
            # Convert order book to MarketData
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            mid_price = order_book.get_mid_price()
            
            if best_bid and best_ask and mid_price:
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=event.timestamp,
                    bid_price=best_bid.price,
                    ask_price=best_ask.price,
                    mid_price=mid_price,
                    bid_quantity=best_bid.quantity,
                    ask_quantity=best_ask.quantity,
                    spread=order_book.get_spread() or 0.0
                )
                
                self.market_data[symbol] = market_data
                
                # Generate synthetic hedge ratios and thresholds for backtesting
                await self._generate_synthetic_signals(symbol, market_data)
                
                # Check for trading opportunities
                await self._check_trading_opportunities(engine, symbol)
        
        elif trade:
            # Update last price from trade
            if symbol in self.market_data:
                self.market_data[symbol].mid_price = trade.price
    
    async def _generate_synthetic_signals(self, symbol: str, market_data: MarketData) -> None:
        """Generate synthetic hedge ratios and thresholds for backtesting."""
        # For backtesting, we'll use simplified synthetic signals
        # In a real implementation, these would come from Kafka topics
        
        # Generate synthetic hedge ratios
        for asset1, asset2 in self.trading_pairs:
            pair_id = f"{asset1}{asset2}"
            
            if symbol == f"{asset1}USDT":
                # Simple synthetic hedge ratio (in reality this comes from Kalman filter)
                base_ratio = 16.67 if asset1 == 'BTC' and asset2 == 'ETH' else 100.0
                # Add some noise
                import numpy as np
                noise = np.random.normal(0, 0.1)
                self.hedge_ratios[pair_id] = base_ratio * (1 + noise)
        
        # Generate synthetic thresholds
        for pair_id in self.state_machines.keys():
            # Simple synthetic thresholds (in reality these come from GARCH model)
            self.thresholds[pair_id] = {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0
            }
    
    async def _check_trading_opportunities(self, engine: BacktestEngine, symbol: str) -> None:
        """Check for trading opportunities."""
        # Check each trading pair
        for asset1, asset2 in self.trading_pairs:
            pair_id = f"{asset1}{asset2}"
            asset1_symbol = f"{asset1}USDT"
            asset2_symbol = f"{asset2}USDT"
            
            # Need both assets' market data
            if asset1_symbol not in self.market_data or asset2_symbol not in self.market_data:
                continue
            
            # Need hedge ratio and thresholds
            if pair_id not in self.hedge_ratios or pair_id not in self.thresholds:
                continue
            
            market_data1 = self.market_data[asset1_symbol]
            market_data2 = self.market_data[asset2_symbol]
            hedge_ratio = self.hedge_ratios[pair_id]
            thresholds = self.thresholds[pair_id]
            
            # Calculate spread and z-score
            spread = market_data1.mid_price - hedge_ratio * market_data2.mid_price
            
            # Simple z-score calculation (in reality this would use historical data)
            # For backtesting, we'll use a synthetic approach
            z_score = spread / (market_data1.mid_price * 0.01)  # Assume 1% volatility
            
            # Generate trading signal
            signal = self._generate_trading_signal(
                pair_id, z_score, thresholds, engine.current_time
            )
            
            if signal:
                await self._execute_signal(engine, signal, pair_id, hedge_ratio)
    
    def _generate_trading_signal(
        self,
        pair_id: str,
        z_score: float,
        thresholds: Dict[str, float],
        timestamp: datetime
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on z-score and thresholds."""
        state_machine = self.state_machines[pair_id]
        current_state = state_machine.get_current_state()
        
        entry_threshold = thresholds['entry_threshold']
        exit_threshold = thresholds['exit_threshold']
        stop_loss_threshold = thresholds['stop_loss_threshold']
        
        # Entry signals
        if current_state == TradingState.SEARCHING:
            if z_score > entry_threshold:
                return TradingSignal(
                    pair_id=pair_id,
                    signal_type=SignalType.ENTRY,
                    side=PositionSide.SHORT,  # Short when spread is high
                    confidence=min(abs(z_score) / entry_threshold, 1.0),
                    z_score=z_score,
                    timestamp=timestamp
                )
            elif z_score < -entry_threshold:
                return TradingSignal(
                    pair_id=pair_id,
                    signal_type=SignalType.ENTRY,
                    side=PositionSide.LONG,  # Long when spread is low
                    confidence=min(abs(z_score) / entry_threshold, 1.0),
                    z_score=z_score,
                    timestamp=timestamp
                )
        
        # Exit signals
        elif current_state == TradingState.IN_POSITION:
            position_side = state_machine.position_side
            
            # Normal exit
            if (position_side == PositionSide.LONG and z_score > -exit_threshold) or \
               (position_side == PositionSide.SHORT and z_score < exit_threshold):
                return TradingSignal(
                    pair_id=pair_id,
                    signal_type=SignalType.EXIT,
                    side=position_side,
                    confidence=0.8,
                    z_score=z_score,
                    timestamp=timestamp
                )
            
            # Stop loss
            elif (position_side == PositionSide.LONG and z_score < -stop_loss_threshold) or \
                 (position_side == PositionSide.SHORT and z_score > stop_loss_threshold):
                return TradingSignal(
                    pair_id=pair_id,
                    signal_type=SignalType.STOP_LOSS,
                    side=position_side,
                    confidence=1.0,
                    z_score=z_score,
                    timestamp=timestamp
                )
        
        return None
    
    async def _execute_signal(
        self,
        engine: BacktestEngine,
        signal: TradingSignal,
        pair_id: str,
        hedge_ratio: float
    ) -> None:
        """Execute a trading signal."""
        state_machine = self.state_machines[pair_id]
        
        # Extract asset symbols
        asset1 = pair_id[:3]
        asset2 = pair_id[3:]
        asset1_symbol = f"{asset1}USDT"
        asset2_symbol = f"{asset2}USDT"
        
        # Get current prices
        price1 = engine.get_last_price(asset1_symbol)
        price2 = engine.get_last_price(asset2_symbol)
        
        if not price1 or not price2:
            return
        
        # Calculate position sizes
        portfolio_value = engine.get_portfolio_value()
        position_size_usd = portfolio_value * 0.1  # Risk 10% per trade
        
        # Calculate quantities
        total_value = price1 + hedge_ratio * price2
        quantity1 = position_size_usd / total_value
        quantity2 = quantity1 * hedge_ratio
        
        try:
            if signal.signal_type == SignalType.ENTRY:
                # Update state machine
                if signal.side == PositionSide.LONG:
                    state_machine.enter_long_position(signal.confidence, signal.z_score)
                    # Long spread: Buy asset1, Sell asset2
                    engine.place_order(asset1_symbol, OrderSide.BUY, OrderType.MARKET, quantity1)
                    engine.place_order(asset2_symbol, OrderSide.SELL, OrderType.MARKET, quantity2)
                else:
                    state_machine.enter_short_position(signal.confidence, signal.z_score)
                    # Short spread: Sell asset1, Buy asset2
                    engine.place_order(asset1_symbol, OrderSide.SELL, OrderType.MARKET, quantity1)
                    engine.place_order(asset2_symbol, OrderSide.BUY, OrderType.MARKET, quantity2)
                
                logger.info("Entered position",
                           pair_id=pair_id,
                           side=signal.side.value,
                           z_score=signal.z_score,
                           confidence=signal.confidence)
            
            elif signal.signal_type in [SignalType.EXIT, SignalType.STOP_LOSS]:
                # Get current position
                position1 = engine.get_position(asset1_symbol)
                position2 = engine.get_position(asset2_symbol)
                
                # Close positions
                if not position1.is_flat:
                    close_side = OrderSide.SELL if position1.quantity > 0 else OrderSide.BUY
                    engine.place_order(asset1_symbol, close_side, OrderType.MARKET, abs(position1.quantity))
                
                if not position2.is_flat:
                    close_side = OrderSide.SELL if position2.quantity > 0 else OrderSide.BUY
                    engine.place_order(asset2_symbol, close_side, OrderType.MARKET, abs(position2.quantity))
                
                # Update state machine
                state_machine.exit_position(signal.z_score)
                
                logger.info("Exited position",
                           pair_id=pair_id,
                           signal_type=signal.signal_type.value,
                           z_score=signal.z_score)
        
        except Exception as e:
            logger.error("Error executing signal",
                        pair_id=pair_id,
                        signal_type=signal.signal_type.value,
                        error=str(e))
    
    async def _handle_order_filled(self, engine: BacktestEngine, event: BacktestEvent) -> None:
        """Handle order filled events."""
        # This could be used to update strategy state based on fills
        pass
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        return {
            pair_id: {
                'state': state_machine.get_current_state().value,
                'position_side': state_machine.position_side.value if state_machine.position_side else None,
                'entry_time': state_machine.entry_time.isoformat() if state_machine.entry_time else None,
                'entry_z_score': state_machine.entry_z_score,
                'confidence': state_machine.confidence
            }
            for pair_id, state_machine in self.state_machines.items()
        }
