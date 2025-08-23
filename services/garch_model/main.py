"""Main service for GARCH volatility forecasting and dynamic threshold calculation."""

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
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import settings
from .model import MultiPairGARCHManager, GARCHForecast
from .thresholds import MultiPairThresholdManager, ThresholdSignal

logger = structlog.get_logger(__name__)


class GARCHVolatilityService:
    """
    Main service for GARCH volatility forecasting and dynamic threshold calculation.
    
    Consumes from: spread-data topic
    Produces to: signals-thresholds topic
    """
    
    def __init__(self):
        self.garch_manager = MultiPairGARCHManager()
        self.threshold_manager = MultiPairThresholdManager()
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'forecasts_generated': 0,
            'signals_published': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }
        
        # Configuration
        self.consumer_topic = "spread-data"
        self.producer_topic = "signals-thresholds"
        self.consumer_group = "garch-volatility-service"
        
        # Trading pairs configuration
        self.trading_pairs = [
            {
                'pair_id': 'BTCETH',
                'garch_window_size': 252,
                'garch_min_observations': 50,
                'garch_refit_frequency': 10,
                'threshold_base_entry': 2.0,
                'threshold_base_exit': 0.5,
                'threshold_volatility_adjustment': 0.5,
                'threshold_lookback_window': 50
            },
            {
                'pair_id': 'BTCADA',
                'garch_window_size': 252,
                'garch_min_observations': 50,
                'garch_refit_frequency': 15,  # Less frequent refitting for more volatile pair
                'threshold_base_entry': 2.5,  # Higher threshold for more volatile pair
                'threshold_base_exit': 0.7,
                'threshold_volatility_adjustment': 0.6,
                'threshold_lookback_window': 75
            },
            {
                'pair_id': 'ETHADA',
                'garch_window_size': 200,
                'garch_min_observations': 40,
                'garch_refit_frequency': 12,
                'threshold_base_entry': 2.2,
                'threshold_base_exit': 0.6,
                'threshold_volatility_adjustment': 0.4,
                'threshold_lookback_window': 60
            }
        ]
        
        logger.info("GARCH Volatility Service initialized")
    
    async def start(self) -> None:
        """Start the service."""
        try:
            logger.info("Starting GARCH Volatility Service")
            
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
            
            logger.info("GARCH Volatility Service started successfully")
            
            # Keep main thread alive
            while self.running:
                await asyncio.sleep(1)
                
                # Log stats periodically
                if self.stats['messages_processed'] % 100 == 0 and self.stats['messages_processed'] > 0:
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
                max_poll_records=50,  # Smaller batches for real-time processing
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
            
            # Initialize trading pairs
            for pair_config in self.trading_pairs:
                pair_id = pair_config['pair_id']
                
                # Add to GARCH manager
                self.garch_manager.add_pair(
                    pair_id=pair_id,
                    window_size=pair_config['garch_window_size'],
                    min_observations=pair_config['garch_min_observations'],
                    refit_frequency=pair_config['garch_refit_frequency']
                )
                
                # Add to threshold manager
                self.threshold_manager.add_pair(
                    pair_id=pair_id,
                    base_entry_threshold=pair_config['threshold_base_entry'],
                    base_exit_threshold=pair_config['threshold_base_exit'],
                    volatility_adjustment_factor=pair_config['threshold_volatility_adjustment'],
                    lookback_window=pair_config['threshold_lookback_window']
                )
            
            logger.info(
                "Components initialized",
                consumer_topic=self.consumer_topic,
                producer_topic=self.producer_topic,
                trading_pairs=[p['pair_id'] for p in self.trading_pairs]
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
                    self._process_spread_message(message.value)
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
    
    def _process_spread_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single spread message from Kafka.
        
        Expected message format:
        {
            "pair_id": "BTCETH",
            "spread_value": 0.0234,
            "asset1_price": 45000.0,
            "asset2_price": 3000.0,
            "hedge_ratio": 15.0,
            "timestamp": "2023-01-01T12:00:00Z"
        }
        """
        try:
            pair_id = message.get('pair_id')
            spread_value = float(message.get('spread_value', 0))
            timestamp_str = message.get('timestamp')
            
            if not pair_id or not np.isfinite(spread_value):
                logger.warning("Invalid spread message", message=message)
                return
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Add observation to GARCH model
            success = self.garch_manager.add_observation(pair_id, spread_value, timestamp)
            if not success:
                logger.warning("Failed to add GARCH observation", pair_id=pair_id)
                return
            
            # Generate volatility forecast
            garch_forecast = self.garch_manager.forecast_volatility(pair_id, horizon=1)
            if garch_forecast is None:
                logger.debug("No GARCH forecast available yet", pair_id=pair_id)
                return
            
            self.stats['forecasts_generated'] += 1
            
            # Generate threshold signal
            threshold_signal = self.threshold_manager.generate_signal(
                pair_id, spread_value, garch_forecast
            )
            
            if threshold_signal is None:
                logger.warning("Failed to generate threshold signal", pair_id=pair_id)
                return
            
            # Publish signal
            self._publish_threshold_signal(threshold_signal, garch_forecast, message)
            
        except Exception as e:
            logger.error("Error processing spread message", error=str(e), message=message)
            raise
    
    def _publish_threshold_signal(
        self, 
        threshold_signal: ThresholdSignal,
        garch_forecast: GARCHForecast,
        original_message: Dict[str, Any]
    ) -> None:
        """Publish threshold signal to Kafka."""
        try:
            # Create comprehensive signal message
            signal_message = {
                # Original spread data
                'pair_id': threshold_signal.pair_id,
                'spread_value': threshold_signal.spread_value,
                'timestamp': threshold_signal.timestamp.isoformat(),
                
                # GARCH forecast data
                'volatility_forecast': garch_forecast.volatility_forecast,
                'variance_forecast': garch_forecast.variance_forecast,
                'volatility_confidence_lower': garch_forecast.confidence_interval_lower,
                'volatility_confidence_upper': garch_forecast.confidence_interval_upper,
                'model_aic': garch_forecast.model_aic,
                'model_bic': garch_forecast.model_bic,
                
                # Threshold and signal data
                'z_score': threshold_signal.z_score,
                'entry_threshold_long': threshold_signal.entry_threshold_long,
                'entry_threshold_short': threshold_signal.entry_threshold_short,
                'exit_threshold': threshold_signal.exit_threshold,
                'signal_type': threshold_signal.signal_type,
                'signal_strength': threshold_signal.signal_strength,
                'confidence_level': threshold_signal.confidence_level,
                'volatility_regime': threshold_signal.volatility_regime,
                
                # Service metadata
                'service': 'garch-volatility',
                'published_at': datetime.utcnow().isoformat(),
                'forecast_horizon': garch_forecast.forecast_horizon,
                
                # Include original message data for context
                'original_data': {
                    key: value for key, value in original_message.items()
                    if key not in ['pair_id', 'spread_value', 'timestamp']
                }
            }
            
            # Send to Kafka
            future = self.producer.send(self.producer_topic, signal_message)
            
            # Wait for confirmation (with timeout)
            record_metadata = future.get(timeout=1)
            
            self.stats['signals_published'] += 1
            
            logger.debug(
                "Threshold signal published",
                pair_id=threshold_signal.pair_id,
                signal_type=threshold_signal.signal_type,
                z_score=threshold_signal.z_score,
                volatility_forecast=garch_forecast.volatility_forecast,
                topic=record_metadata.topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset
            )
            
        except Exception as e:
            logger.error(
                "Failed to publish threshold signal",
                error=str(e),
                pair_id=threshold_signal.pair_id,
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
        
        # Get GARCH diagnostics
        garch_diagnostics = self.garch_manager.get_all_diagnostics()
        threshold_stats = self.threshold_manager.get_all_statistics()
        
        logger.info(
            "Service statistics",
            uptime_seconds=uptime,
            messages_processed=self.stats['messages_processed'],
            forecasts_generated=self.stats['forecasts_generated'],
            signals_published=self.stats['signals_published'],
            errors=self.stats['errors'],
            messages_per_second=self.stats['messages_processed'] / max(uptime, 1),
            last_message_time=self.stats['last_message_time'].isoformat() if self.stats['last_message_time'] else None,
            garch_models_fitted=sum(1 for d in garch_diagnostics.values() if d.get('model_fitted', False)),
            total_garch_observations=sum(d.get('total_observations', 0) for d in garch_diagnostics.values())
        )
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources")
        
        try:
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
        garch_diagnostics = self.garch_manager.get_all_diagnostics()
        threshold_stats = self.threshold_manager.get_all_statistics()
        
        return {
            'status': 'healthy' if self.running else 'stopped',
            'uptime_seconds': (
                (datetime.utcnow() - self.stats['start_time']).total_seconds()
                if self.stats['start_time'] else 0
            ),
            'stats': self.stats.copy(),
            'garch_models': {
                pair_id: {
                    'fitted': diag.get('model_fitted', False),
                    'observations': diag.get('total_observations', 0),
                    'successful_fits': diag.get('successful_fits', 0),
                    'failed_fits': diag.get('failed_fits', 0),
                    'last_fit': diag.get('last_fit_time'),
                    'model_persistence': diag.get('model_persistence', 0)
                }
                for pair_id, diag in garch_diagnostics.items()
            },
            'threshold_calculators': {
                pair_id: {
                    'spread_history_size': stats.get('spread_history_size', 0),
                    'volatility_history_size': stats.get('volatility_history_size', 0),
                    'current_position': stats.get('current_position'),
                    'current_mean': stats.get('current_mean', 0),
                    'current_std': stats.get('current_std', 1)
                }
                for pair_id, stats in threshold_stats.items()
            },
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
    service = GARCHVolatilityService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error("Service failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
