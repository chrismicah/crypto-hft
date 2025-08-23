"""Main execution service orchestrator for trading strategy."""

import asyncio
import json
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import threading
import time
from collections import defaultdict, deque

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import settings
from .exchange import BinanceTestnetExchange, ExchangeManager
from .signals import SignalGenerator, MarketData, TradingSignal, SignalType
from .state_machine import StrategyStateMachine, TradingState, PositionSide

logger = structlog.get_logger(__name__)


class DataAggregator:
    """Aggregates data from multiple Kafka topics."""
    
    def __init__(self, data_ttl: timedelta = timedelta(seconds=30)):
        """
        Initialize data aggregator.
        
        Args:
            data_ttl: Time-to-live for cached data
        """
        self.data_ttl = data_ttl
        
        # Data caches with timestamps
        self.order_book_data: Dict[str, Dict[str, Any]] = {}
        self.hedge_ratios: Dict[str, Dict[str, Any]] = {}
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        
        # Timestamps for data freshness
        self.last_update_times: Dict[str, Dict[str, datetime]] = {
            'order_book': {},
            'hedge_ratios': {},
            'thresholds': {}
        }
    
    def update_order_book(self, pair_id: str, data: Dict[str, Any]) -> None:
        """Update order book data."""
        self.order_book_data[pair_id] = data
        self.last_update_times['order_book'][pair_id] = datetime.utcnow()
        
        logger.debug("Order book data updated", pair_id=pair_id)
    
    def update_hedge_ratio(self, pair_id: str, data: Dict[str, Any]) -> None:
        """Update hedge ratio data."""
        self.hedge_ratios[pair_id] = data
        self.last_update_times['hedge_ratios'][pair_id] = datetime.utcnow()
        
        logger.debug("Hedge ratio data updated", pair_id=pair_id)
    
    def update_thresholds(self, pair_id: str, data: Dict[str, Any]) -> None:
        """Update threshold data."""
        self.thresholds[pair_id] = data
        self.last_update_times['thresholds'][pair_id] = datetime.utcnow()
        
        logger.debug("Threshold data updated", pair_id=pair_id)
    
    def get_market_data(self, pair_id: str) -> Optional[MarketData]:
        """
        Get aggregated market data for a pair.
        
        Args:
            pair_id: Trading pair identifier
            
        Returns:
            MarketData object if all required data is available and fresh
        """
        try:
            current_time = datetime.utcnow()
            
            # Check data freshness
            for data_type in ['order_book', 'hedge_ratios', 'thresholds']:
                if (pair_id not in self.last_update_times[data_type] or
                    current_time - self.last_update_times[data_type][pair_id] > self.data_ttl):
                    logger.debug(
                        "Stale or missing data",
                        pair_id=pair_id,
                        data_type=data_type,
                        last_update=self.last_update_times[data_type].get(pair_id)
                    )
                    return None
            
            # Get data
            order_book = self.order_book_data.get(pair_id, {})
            hedge_ratio_data = self.hedge_ratios.get(pair_id, {})
            threshold_data = self.thresholds.get(pair_id, {})
            
            # Extract order book prices
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                logger.debug("Missing order book data", pair_id=pair_id)
                return None
            
            # Determine asset prices from pair_id
            # Assuming pair_id format like "BTCETH" means BTC/ETH
            if pair_id == "BTCETH":
                # For BTCETH pair, we need BTCUSDT and ETHUSDT prices
                # This is a simplification - in reality we'd need separate order books
                asset1_bid = bids[0][0] if bids else None  # BTC bid
                asset1_ask = asks[0][0] if asks else None  # BTC ask
                asset2_bid = bids[0][0] / 15.0 if bids else None  # ETH bid (rough estimate)
                asset2_ask = asks[0][0] / 15.0 if asks else None  # ETH ask (rough estimate)
            else:
                # Default handling
                asset1_bid = bids[0][0] if bids else None
                asset1_ask = asks[0][0] if asks else None
                asset2_bid = bids[0][0] if bids else None
                asset2_ask = asks[0][0] if asks else None
            
            # Create market data
            market_data = MarketData(
                asset1_bid=asset1_bid,
                asset1_ask=asset1_ask,
                asset2_bid=asset2_bid,
                asset2_ask=asset2_ask,
                hedge_ratio=hedge_ratio_data.get('hedge_ratio'),
                hedge_ratio_confidence=hedge_ratio_data.get('confidence_interval_95'),
                volatility_forecast=threshold_data.get('volatility_forecast'),
                entry_threshold_long=threshold_data.get('entry_threshold_long'),
                entry_threshold_short=threshold_data.get('entry_threshold_short'),
                exit_threshold=threshold_data.get('exit_threshold'),
                volatility_regime=threshold_data.get('volatility_regime'),
                timestamp=current_time
            )
            
            return market_data
            
        except Exception as e:
            logger.error(
                "Failed to aggregate market data",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get status of all cached data."""
        current_time = datetime.utcnow()
        status = {}
        
        for pair_id in set(
            list(self.order_book_data.keys()) +
            list(self.hedge_ratios.keys()) +
            list(self.thresholds.keys())
        ):
            pair_status = {}
            
            for data_type in ['order_book', 'hedge_ratios', 'thresholds']:
                last_update = self.last_update_times[data_type].get(pair_id)
                if last_update:
                    age = (current_time - last_update).total_seconds()
                    is_fresh = age <= self.data_ttl.total_seconds()
                    pair_status[data_type] = {
                        'last_update': last_update.isoformat(),
                        'age_seconds': age,
                        'is_fresh': is_fresh
                    }
                else:
                    pair_status[data_type] = {
                        'last_update': None,
                        'age_seconds': None,
                        'is_fresh': False
                    }
            
            status[pair_id] = pair_status
        
        return status


class ExecutionService:
    """
    Main execution service that orchestrates trading decisions.
    
    Consumes from multiple Kafka topics:
    - order_book_updates: Order book data
    - signals-hedge-ratio: Hedge ratios from Kalman filter
    - signals-thresholds: Thresholds from GARCH volatility service
    
    Executes trades on Binance Testnet via CCXT.
    """
    
    def __init__(self):
        """Initialize the execution service."""
        # Core components
        self.data_aggregator = DataAggregator()
        self.signal_generator = SignalGenerator()
        self.exchange: Optional[BinanceTestnetExchange] = None
        self.exchange_manager: Optional[ExchangeManager] = None
        
        # State machines for each trading pair
        self.state_machines: Dict[str, StrategyStateMachine] = {}
        
        # Kafka consumers
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.consumer_threads: Dict[str, threading.Thread] = {}
        
        # Service state
        self.running = False
        self.stats = {
            'messages_processed': 0,
            'signals_generated': 0,
            'orders_placed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }
        
        # Configuration
        self.trading_pairs = [
            {
                'pair_id': 'BTCETH',
                'asset1_symbol': 'BTCUSDT',
                'asset2_symbol': 'ETHUSDT',
                'enabled': True
            },
            {
                'pair_id': 'BTCADA',
                'asset1_symbol': 'BTCUSDT',
                'asset2_symbol': 'ADAUSDT',
                'enabled': True
            }
        ]
        
        # Kafka topics
        self.kafka_topics = {
            'order_book': 'order_book_updates',
            'hedge_ratios': 'signals-hedge-ratio',
            'thresholds': 'signals-thresholds'
        }
        
        logger.info("Execution service initialized")
    
    async def start(self) -> None:
        """Start the execution service."""
        try:
            logger.info("Starting execution service")
            
            # Initialize components
            await self._initialize_components()
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Start service
            self.running = True
            self.stats['start_time'] = datetime.utcnow()
            
            # Start Kafka consumers
            self._start_kafka_consumers()
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            logger.error("Failed to start execution service", error=str(e), exc_info=True)
            raise
        finally:
            await self._cleanup()
    
    async def _initialize_components(self) -> None:
        """Initialize exchange and state machines."""
        try:
            # Initialize exchange
            self.exchange = BinanceTestnetExchange(
                api_key=settings.binance_api_key,
                secret_key=settings.binance_secret_key,
                testnet=settings.binance_testnet
            )
            
            success = await self.exchange.initialize()
            if not success:
                raise Exception("Failed to initialize exchange")
            
            self.exchange_manager = ExchangeManager(self.exchange)
            await self.exchange_manager.initialize()
            
            # Initialize state machines for each trading pair
            for pair_config in self.trading_pairs:
                if pair_config['enabled']:
                    pair_id = pair_config['pair_id']
                    self.state_machines[pair_id] = StrategyStateMachine(pair_id)
            
            logger.info(
                "Components initialized",
                exchange_initialized=True,
                trading_pairs=[p['pair_id'] for p in self.trading_pairs if p['enabled']]
            )
            
        except Exception as e:
            logger.error("Failed to initialize components", error=str(e), exc_info=True)
            raise
    
    def _start_kafka_consumers(self) -> None:
        """Start Kafka consumer threads."""
        try:
            # Order book consumer
            self.consumers['order_book'] = KafkaConsumer(
                self.kafka_topics['order_book'],
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id='execution-service-orderbook',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Hedge ratio consumer
            self.consumers['hedge_ratios'] = KafkaConsumer(
                self.kafka_topics['hedge_ratios'],
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id='execution-service-hedgeratios',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Thresholds consumer
            self.consumers['thresholds'] = KafkaConsumer(
                self.kafka_topics['thresholds'],
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id='execution-service-thresholds',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Start consumer threads
            self.consumer_threads['order_book'] = threading.Thread(
                target=self._order_book_consumer_loop,
                daemon=True
            )
            self.consumer_threads['hedge_ratios'] = threading.Thread(
                target=self._hedge_ratio_consumer_loop,
                daemon=True
            )
            self.consumer_threads['thresholds'] = threading.Thread(
                target=self._thresholds_consumer_loop,
                daemon=True
            )
            
            # Start threads
            for thread in self.consumer_threads.values():
                thread.start()
            
            logger.info("Kafka consumers started")
            
        except Exception as e:
            logger.error("Failed to start Kafka consumers", error=str(e), exc_info=True)
            raise
    
    def _order_book_consumer_loop(self) -> None:
        """Order book consumer loop."""
        logger.info("Starting order book consumer loop")
        
        try:
            for message in self.consumers['order_book']:
                if not self.running:
                    break
                
                try:
                    data = message.value
                    symbol = data.get('symbol', '')
                    
                    # Map symbol to pair_id (simplified)
                    pair_id = None
                    if symbol == 'BTCUSDT':
                        pair_id = 'BTCETH'  # Simplified mapping
                    elif symbol == 'ETHUSDT':
                        pair_id = 'BTCETH'
                    
                    if pair_id:
                        self.data_aggregator.update_order_book(pair_id, data)
                    
                    self.stats['messages_processed'] += 1
                    
                except Exception as e:
                    logger.error("Error processing order book message", error=str(e))
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error("Order book consumer loop error", error=str(e))
            self.running = False
    
    def _hedge_ratio_consumer_loop(self) -> None:
        """Hedge ratio consumer loop."""
        logger.info("Starting hedge ratio consumer loop")
        
        try:
            for message in self.consumers['hedge_ratios']:
                if not self.running:
                    break
                
                try:
                    data = message.value
                    pair_id = data.get('pair_id')
                    
                    if pair_id:
                        self.data_aggregator.update_hedge_ratio(pair_id, data)
                    
                    self.stats['messages_processed'] += 1
                    
                except Exception as e:
                    logger.error("Error processing hedge ratio message", error=str(e))
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error("Hedge ratio consumer loop error", error=str(e))
            self.running = False
    
    def _thresholds_consumer_loop(self) -> None:
        """Thresholds consumer loop."""
        logger.info("Starting thresholds consumer loop")
        
        try:
            for message in self.consumers['thresholds']:
                if not self.running:
                    break
                
                try:
                    data = message.value
                    pair_id = data.get('pair_id')
                    
                    if pair_id:
                        self.data_aggregator.update_thresholds(pair_id, data)
                    
                    self.stats['messages_processed'] += 1
                    
                except Exception as e:
                    logger.error("Error processing thresholds message", error=str(e))
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error("Thresholds consumer loop error", error=str(e))
            self.running = False
    
    async def _main_loop(self) -> None:
        """Main processing loop."""
        logger.info("Starting main processing loop")
        
        while self.running:
            try:
                # Process each trading pair
                for pair_config in self.trading_pairs:
                    if not pair_config['enabled']:
                        continue
                    
                    pair_id = pair_config['pair_id']
                    await self._process_trading_pair(pair_id, pair_config)
                
                # Check state machine timeouts
                for state_machine in self.state_machines.values():
                    state_machine.check_timeouts()
                
                # Update exchange positions
                if self.exchange_manager:
                    await self.exchange_manager.update_positions()
                
                # Log stats periodically
                if self.stats['messages_processed'] % 100 == 0 and self.stats['messages_processed'] > 0:
                    self._log_stats()
                
                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("Error in main loop", error=str(e), exc_info=True)
                self.stats['errors'] += 1
                await asyncio.sleep(5)  # Longer sleep on error
    
    async def _process_trading_pair(self, pair_id: str, pair_config: Dict[str, Any]) -> None:
        """Process a single trading pair."""
        try:
            # Get aggregated market data
            market_data = self.data_aggregator.get_market_data(pair_id)
            if not market_data:
                return  # Skip if data not available
            
            # Get state machine
            state_machine = self.state_machines.get(pair_id)
            if not state_machine:
                return
            
            # Get current position
            current_position = None
            if state_machine.has_position():
                position_side = state_machine.get_position_side()
                current_position = position_side.value if position_side else None
            
            # Generate trading signal
            signal = self.signal_generator.generate_signal(
                pair_id, market_data, current_position
            )
            
            self.stats['signals_generated'] += 1
            
            # Update position PnL if in position
            if state_machine.has_position() and market_data.spread and market_data.z_score:
                state_machine.update_position_pnl(market_data.spread, market_data.z_score)
            
            # Process signal through state machine
            signal_processed = state_machine.process_signal(signal)
            
            if signal_processed:
                # Execute trades based on state transitions
                await self._execute_state_actions(pair_id, pair_config, state_machine, signal)
            
        except Exception as e:
            logger.error(
                "Error processing trading pair",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            self.stats['errors'] += 1
    
    async def _execute_state_actions(
        self,
        pair_id: str,
        pair_config: Dict[str, Any],
        state_machine: StrategyStateMachine,
        signal: TradingSignal
    ) -> None:
        """Execute actions based on state machine transitions."""
        try:
            current_state = state_machine.get_current_state()
            
            # Handle entering state - place orders
            if current_state == TradingState.ENTERING and signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                success = await self._place_entry_orders(pair_id, pair_config, state_machine, signal)
                
                if success:
                    state_machine.position_entered()
                    self.stats['positions_opened'] += 1
                    logger.info("Position entered successfully", pair_id=pair_id)
                else:
                    state_machine.entry_failed()
                    logger.warning("Failed to enter position", pair_id=pair_id)
            
            # Handle exiting state - place exit orders
            elif current_state == TradingState.EXITING and signal.signal_type in [SignalType.EXIT, SignalType.STOP_LOSS]:
                success = await self._place_exit_orders(pair_id, pair_config, state_machine, signal)
                
                if success:
                    state_machine.position_exited()
                    self.stats['positions_closed'] += 1
                    logger.info("Position exited successfully", pair_id=pair_id)
                else:
                    state_machine.exit_failed()
                    logger.warning("Failed to exit position", pair_id=pair_id)
            
        except Exception as e:
            logger.error(
                "Error executing state actions",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            state_machine.error_occurred(error_message=str(e))
    
    async def _place_entry_orders(
        self,
        pair_id: str,
        pair_config: Dict[str, Any],
        state_machine: StrategyStateMachine,
        signal: TradingSignal
    ) -> bool:
        """Place entry orders for a pairs trade."""
        try:
            if not self.exchange_manager:
                return False
            
            # Get account balance for position sizing
            balance = await self.exchange.get_balance()
            if not balance or 'USDT' not in balance:
                logger.error("Failed to get account balance")
                return False
            
            usdt_balance = balance['USDT'].free
            
            # Calculate position size
            position_size = state_machine.order_sizer.calculate_position_size(
                account_balance=usdt_balance,
                signal=signal,
                current_volatility=signal.volatility_forecast
            )
            
            # Get hedge ratio
            hedge_ratio = signal.hedge_ratio or 1.0
            
            # Execute pairs trade
            order1, order2 = await self.exchange_manager.execute_pair_trade(
                pair_id=pair_id,
                asset1_symbol=pair_config['asset1_symbol'],
                asset2_symbol=pair_config['asset2_symbol'],
                side=signal.side,
                hedge_ratio=hedge_ratio,
                notional_amount=position_size,
                order_type='market'
            )
            
            if order1 and order2:
                # Update state machine with order info
                if state_machine.current_position:
                    state_machine.current_position.asset1_order_id = order1.order_id
                    state_machine.current_position.asset2_order_id = order2.order_id
                    state_machine.current_position.asset1_fill_price = order1.price
                    state_machine.current_position.asset2_fill_price = order2.price
                    state_machine.current_position.asset1_quantity = order1.amount
                    state_machine.current_position.asset2_quantity = order2.amount
                
                self.stats['orders_placed'] += 2
                
                logger.info(
                    "Entry orders placed",
                    pair_id=pair_id,
                    side=signal.side,
                    position_size=position_size,
                    order1_id=order1.order_id,
                    order2_id=order2.order_id
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "Failed to place entry orders",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def _place_exit_orders(
        self,
        pair_id: str,
        pair_config: Dict[str, Any],
        state_machine: StrategyStateMachine,
        signal: TradingSignal
    ) -> bool:
        """Place exit orders for a pairs trade."""
        try:
            if not self.exchange_manager or not state_machine.current_position:
                return False
            
            position = state_machine.current_position
            
            # Determine exit side (opposite of entry)
            if position.side == PositionSide.LONG:
                exit_side = 'short'  # Close long position
            else:
                exit_side = 'long'   # Close short position
            
            # Execute pairs trade to close position
            order1, order2 = await self.exchange_manager.execute_pair_trade(
                pair_id=pair_id,
                asset1_symbol=pair_config['asset1_symbol'],
                asset2_symbol=pair_config['asset2_symbol'],
                side=exit_side,
                hedge_ratio=position.hedge_ratio,
                notional_amount=abs(position.unrealized_pnl) + 100,  # Simplified
                order_type='market'
            )
            
            if order1 and order2:
                self.stats['orders_placed'] += 2
                
                logger.info(
                    "Exit orders placed",
                    pair_id=pair_id,
                    exit_side=exit_side,
                    pnl=position.unrealized_pnl,
                    order1_id=order1.order_id,
                    order2_id=order2.order_id
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "Failed to place exit orders",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal", signal=signum)
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _log_stats(self) -> None:
        """Log service statistics."""
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        # Get state machine states
        state_summary = {}
        for pair_id, sm in self.state_machines.items():
            state_summary[pair_id] = sm.get_current_state().value
        
        logger.info(
            "Execution service statistics",
            uptime_seconds=uptime,
            messages_processed=self.stats['messages_processed'],
            signals_generated=self.stats['signals_generated'],
            orders_placed=self.stats['orders_placed'],
            positions_opened=self.stats['positions_opened'],
            positions_closed=self.stats['positions_closed'],
            errors=self.stats['errors'],
            state_summary=state_summary,
            data_status=self.data_aggregator.get_data_status()
        )
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up execution service")
        
        try:
            # Close Kafka consumers
            for consumer in self.consumers.values():
                consumer.close()
            
            # Close exchange connection
            if self.exchange:
                await self.exchange.close()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e), exc_info=True)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'uptime_seconds': (
                (datetime.utcnow() - self.stats['start_time']).total_seconds()
                if self.stats['start_time'] else 0
            ),
            'stats': self.stats.copy(),
            'state_machines': {
                pair_id: sm.get_state_info()
                for pair_id, sm in self.state_machines.items()
            },
            'exchange_connected': self.exchange is not None,
            'data_status': self.data_aggregator.get_data_status(),
            'active_consumers': len([t for t in self.consumer_threads.values() if t.is_alive()])
        }


async def main():
    """Main entry point."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create and start service
    service = ExecutionService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error("Service failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
