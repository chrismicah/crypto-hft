"""Main service for Kalman Filter Hedge Ratio calculation."""

import asyncio
import json
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import threading
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import settings
from .filter import PairTradingKalmanFilter
from .state import StateManager

logger = structlog.get_logger(__name__)


class KalmanFilterService:
    """
    Main service for consuming tick data and producing hedge ratio signals.
    
    Consumes from: trade-ticks topic
    Produces to: signals-hedge-ratio topic
    """
    
    def __init__(self):
        self.pair_filter = PairTradingKalmanFilter()
        self.state_manager = StateManager()
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.running = False
        self.stats = {
            'messages_processed': 0,
            'signals_published': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }
        
        # Configuration
        self.consumer_topic = "trade-ticks"
        self.producer_topic = "signals-hedge-ratio"
        self.consumer_group = "kalman-filter-service"
        
        # Trading pairs configuration
        self.trading_pairs = [
            {
                'pair_id': 'BTCETH',
                'asset1': 'BTCUSDT',
                'asset2': 'ETHUSDT',
                'initial_hedge_ratio': 15.0,  # Rough BTC/ETH ratio
                'process_variance': 1e-4,
                'observation_variance': 1e-2
            },
            {
                'pair_id': 'BTCADA',
                'asset1': 'BTCUSDT',
                'asset2': 'ADAUSDT',
                'initial_hedge_ratio': 100000.0,  # Rough BTC/ADA ratio
                'process_variance': 1e-3,
                'observation_variance': 1e-1
            }
        ]
        
        logger.info("Kalman Filter Service initialized")
    
    async def start(self) -> None:
        """Start the service."""
        try:
            logger.info("Starting Kalman Filter Service")
            
            # Initialize components
            await self._initialize_components()
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Start processing
            self.running = True
            self.stats['start_time'] = datetime.utcnow()
            
            # Start consumer in background thread
            consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
            consumer_thread.start()
            
            # Start state manager auto-save
            self.state_manager.start_auto_save()
            
            logger.info("Kalman Filter Service started successfully")
            
            # Keep main thread alive
            while self.running:
                await asyncio.sleep(1)
                
                # Log stats periodically
                if self.stats['messages_processed'] % 1000 == 0 and self.stats['messages_processed'] > 0:
                    self._log_stats()
            
        except Exception as e:
            logger.error("Failed to start service", error=str(e), exc_info=True)
            raise
        finally:
            await self._cleanup()
    
    async def _initialize_components(self) -> None:
        """Initialize Kafka consumer, producer, and trading pairs."""
        try:
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                self.consumer_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id=self.consumer_group,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=100,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            
            # Load or initialize trading pairs
            loaded_pair_filter = self.state_manager.load_pair_filters("main")
            
            if loaded_pair_filter:
                self.pair_filter = loaded_pair_filter
                logger.info("Loaded existing trading pairs from state")
            else:
                # Initialize new trading pairs
                for pair_config in self.trading_pairs:
                    self.pair_filter.add_pair(**pair_config)
                logger.info("Initialized new trading pairs")
            
            # Register for auto-save
            self.state_manager.register_pair_filter_for_auto_save("main", self.pair_filter)
            
            logger.info(
                "Components initialized",
                consumer_topic=self.consumer_topic,
                producer_topic=self.producer_topic,
                trading_pairs=list(self.pair_filter.filters.keys())
            )
            
        except Exception as e:
            logger.error("Failed to initialize components", error=str(e), exc_info=True)
            raise
    
    def _consumer_loop(self) -> None:
        """Main consumer loop running in background thread."""
        logger.info("Starting consumer loop")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    # Process the message
                    self._process_tick_message(message.value)
                    self.stats['messages_processed'] += 1
                    self.stats['last_message_time'] = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(
                        "Error processing message",
                        error=str(e),
                        message=message.value,
                        exc_info=True
                    )
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error("Consumer loop error", error=str(e), exc_info=True)
            self.running = False
    
    def _process_tick_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single tick message from Kafka.
        
        Expected message format:
        {
            "symbol": "BTCUSDT",
            "price": 45000.0,
            "timestamp": "2023-01-01T12:00:00Z",
            "volume": 1.5
        }
        """
        try:
            symbol = message.get('symbol')
            price = float(message.get('price', 0))
            timestamp_str = message.get('timestamp')
            
            if not symbol or price <= 0:
                logger.warning("Invalid tick message", message=message)
                return
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Store price for pair matching
            self._update_price_cache(symbol, price, timestamp)
            
            # Check for pair updates
            self._check_and_update_pairs(timestamp)
            
        except Exception as e:
            logger.error("Error processing tick message", error=str(e), message=message)
            raise
    
    def _update_price_cache(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update the price cache for pair matching."""
        if not hasattr(self, '_price_cache'):
            self._price_cache = {}
        
        self._price_cache[symbol] = {
            'price': price,
            'timestamp': timestamp
        }
    
    def _check_and_update_pairs(self, current_timestamp: datetime) -> None:
        """Check if we can update any trading pairs with current prices."""
        if not hasattr(self, '_price_cache'):
            return
        
        for pair_id, pair_info in self.pair_filter.pair_configs.items():
            asset1 = pair_info['asset1']
            asset2 = pair_info['asset2']
            
            # Check if we have recent prices for both assets
            asset1_data = self._price_cache.get(asset1)
            asset2_data = self._price_cache.get(asset2)
            
            if not asset1_data or not asset2_data:
                continue
            
            # Check if prices are recent enough (within 10 seconds)
            max_age_seconds = 10
            if (current_timestamp - asset1_data['timestamp']).total_seconds() > max_age_seconds:
                continue
            if (current_timestamp - asset2_data['timestamp']).total_seconds() > max_age_seconds:
                continue
            
            # Update the pair
            result = self.pair_filter.update_pair(
                pair_id=pair_id,
                asset1_price=asset1_data['price'],
                asset2_price=asset2_data['price'],
                timestamp=current_timestamp
            )
            
            if result:
                # Publish signal
                self._publish_hedge_ratio_signal(result)
    
    def _publish_hedge_ratio_signal(self, signal_data: Dict[str, Any]) -> None:
        """Publish hedge ratio signal to Kafka."""
        try:
            # Add service metadata
            signal_message = {
                **signal_data,
                'service': 'kalman-filter',
                'published_at': datetime.utcnow().isoformat()
            }
            
            # Send to Kafka
            future = self.producer.send(self.producer_topic, signal_message)
            
            # Wait for confirmation (with timeout)
            record_metadata = future.get(timeout=1)
            
            self.stats['signals_published'] += 1
            
            logger.debug(
                "Hedge ratio signal published",
                pair_id=signal_data['pair_id'],
                hedge_ratio=signal_data['hedge_ratio'],
                topic=record_metadata.topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset
            )
            
        except Exception as e:
            logger.error(
                "Failed to publish hedge ratio signal",
                error=str(e),
                signal_data=signal_data,
                exc_info=True
            )
            self.stats['errors'] += 1
    
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
        
        logger.info(
            "Service statistics",
            uptime_seconds=uptime,
            messages_processed=self.stats['messages_processed'],
            signals_published=self.stats['signals_published'],
            errors=self.stats['errors'],
            messages_per_second=self.stats['messages_processed'] / max(uptime, 1),
            last_message_time=self.stats['last_message_time'].isoformat() if self.stats['last_message_time'] else None
        )
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources")
        
        try:
            # Stop state manager
            self.state_manager.stop_auto_save()
            
            # Save final state
            if self.pair_filter:
                self.state_manager.save_pair_filters(self.pair_filter, "main")
            
            # Close Kafka connections
            if self.consumer:
                self.consumer.close()
            
            if self.producer:
                self.producer.flush()
                self.producer.close()
            
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
            'trading_pairs': self.pair_filter.get_all_pairs(),
            'kafka_connected': bool(self.consumer and self.producer)
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
    service = KalmanFilterService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error("Service failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
