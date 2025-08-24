"""
On-Chain Data Service
Ingests blockchain metrics and generates trading signals.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
import sys

from kafka import KafkaProducer
from kafka.errors import KafkaError
import structlog
from pydantic import BaseSettings
import aiohttp

from common.logger import get_logger
from common.metrics import ServiceMetrics
from .clients.glassnode_client import GlassnodeClient, GlassnodeAPIError
from .clients.cryptoquant_client import CryptoQuantClient, CryptoQuantAPIError
from .processors.feature_engineering import OnChainFeatureEngineer, FeatureConfig
from .signals.signal_generator import OnChainSignalGenerator, ModelConfig
from .models import (
    OnChainMetrics, OnChainSignal, OnChainAlert,
    OnChainDataSource, OnChainMetricType
)


class OnChainDataConfig(BaseSettings):
    """Configuration for on-chain data service."""
    
    # Service settings
    service_name: str = "onchain-data"
    service_port: int = 8006
    log_level: str = "INFO"
    
    # API keys
    glassnode_api_key: str = ""
    cryptoquant_api_key: str = ""
    
    # Data collection settings
    collection_interval: int = 3600  # 1 hour
    symbols: str = "BTC,ETH,ADA,BNB,SOL"
    lookback_days: int = 30
    
    # Kafka settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_onchain_metrics: str = "onchain-metrics"
    kafka_topic_onchain_signals: str = "onchain-signals"
    kafka_topic_onchain_alerts: str = "onchain-alerts"
    
    # ML model settings
    model_retrain_interval: int = 86400  # 24 hours
    min_training_samples: int = 100
    enable_model_training: bool = True
    model_save_path: str = "/app/data/onchain_models.pkl"
    
    # Feature engineering
    feature_windows: str = "7,14,30"  # Days for moving averages
    volatility_windows: str = "7,14"   # Days for volatility calculation
    
    # Rate limiting
    api_rate_limit_per_minute: int = 100
    
    class Config:
        env_prefix = "ONCHAIN_"


class OnChainDataService:
    """
    Main service for on-chain data ingestion and signal generation.
    
    Capabilities:
    - Fetches on-chain metrics from multiple sources (Glassnode, CryptoQuant)
    - Engineers features for ML models
    - Generates trading signals using ensemble models
    - Publishes signals to Kafka for consumption by trading services
    - Monitors API health and data quality
    """
    
    def __init__(self, config: OnChainDataConfig):
        self.config = config
        self.logger = get_logger(config.service_name)
        
        # Metrics
        self.metrics = ServiceMetrics(
            service_name=config.service_name,
            port=config.service_port
        )
        
        # API clients
        self.glassnode_client: Optional[GlassnodeClient] = None
        self.cryptoquant_client: Optional[CryptoQuantClient] = None
        
        # Processing pipeline
        self.feature_engineer = OnChainFeatureEngineer(
            config=FeatureConfig(
                lookback_windows=[int(x) for x in config.feature_windows.split(",")],
                volatility_windows=[int(x) for x in config.volatility_windows.split(",")]
            )
        )
        
        self.signal_generator = OnChainSignalGenerator(
            config=ModelConfig(
                min_training_samples=config.min_training_samples
            )
        )
        
        # Kafka producer
        self.kafka_producer: Optional[KafkaProducer] = None
        
        # State management
        self.symbols = [s.strip() for s in config.symbols.split(",")]
        self.running = False
        self.last_collection_time = {}
        self.last_model_training = None
        
        # Data storage
        self.historical_metrics = {}  # symbol -> List[OnChainMetrics]
        self.historical_signals = {}  # symbol -> List[OnChainSignal]
        
        # Health monitoring
        self.api_health = {
            "glassnode": {"status": "unknown", "last_check": None},
            "cryptoquant": {"status": "unknown", "last_check": None}
        }
    
    async def start(self):
        """Start the on-chain data service."""
        self.logger.info("Starting on-chain data service")
        
        try:
            await self._initialize_clients()
            await self._initialize_kafka()
            await self._load_existing_models()
            
            self.running = True
            self.logger.info(f"Service started successfully on port {self.config.service_port}")
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            raise
    
    async def stop(self):
        """Stop the service gracefully."""
        self.logger.info("Stopping on-chain data service")
        self.running = False
        
        # Close clients
        if self.glassnode_client:
            await self.glassnode_client.close()
        if self.cryptoquant_client:
            await self.cryptoquant_client.close()
        
        # Close Kafka producer
        if self.kafka_producer:
            self.kafka_producer.close()
        
        self.logger.info("Service stopped")
    
    async def _initialize_clients(self):
        """Initialize API clients."""
        self.logger.info("Initializing API clients")
        
        # Initialize Glassnode client
        if self.config.glassnode_api_key:
            self.glassnode_client = GlassnodeClient(
                api_key=self.config.glassnode_api_key,
                rate_limit_per_minute=self.config.api_rate_limit_per_minute
            )
            self.logger.info("Glassnode client initialized")
        else:
            self.logger.warning("No Glassnode API key provided")
        
        # Initialize CryptoQuant client
        if self.config.cryptoquant_api_key:
            self.cryptoquant_client = CryptoQuantClient(
                api_key=self.config.cryptoquant_api_key,
                rate_limit_per_minute=self.config.api_rate_limit_per_minute
            )
            self.logger.info("CryptoQuant client initialized")
        else:
            self.logger.warning("No CryptoQuant API key provided")
        
        if not self.glassnode_client and not self.cryptoquant_client:
            raise ValueError("At least one API key must be provided")
    
    async def _initialize_kafka(self):
        """Initialize Kafka producer."""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers.split(","),
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True
            )
            self.logger.info("Kafka producer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def _load_existing_models(self):
        """Load existing ML models if available."""
        if os.path.exists(self.config.model_save_path):
            try:
                self.signal_generator.load_models(self.config.model_save_path)
                self.logger.info("Loaded existing ML models")
            except Exception as e:
                self.logger.warning(f"Failed to load existing models: {e}")
    
    async def _main_loop(self):
        """Main processing loop."""
        self.logger.info("Starting main processing loop")
        
        while self.running:
            try:
                loop_start = datetime.utcnow()
                
                # Check if it's time to collect data
                if self._should_collect_data():
                    await self._collect_and_process_data()
                
                # Check if it's time to retrain models
                if self._should_retrain_models():
                    await self._retrain_models()
                
                # Update metrics
                self.metrics.processing_time.observe(
                    (datetime.utcnow() - loop_start).total_seconds()
                )
                
                # Sleep until next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.metrics.errors.inc()
                await asyncio.sleep(60)
    
    def _should_collect_data(self) -> bool:
        """Check if it's time to collect data."""
        now = datetime.utcnow()
        
        # Check each symbol
        for symbol in self.symbols:
            last_collection = self.last_collection_time.get(symbol)
            if (last_collection is None or 
                (now - last_collection).total_seconds() >= self.config.collection_interval):
                return True
        
        return False
    
    def _should_retrain_models(self) -> bool:
        """Check if it's time to retrain models."""
        if not self.config.enable_model_training:
            return False
        
        now = datetime.utcnow()
        
        if (self.last_model_training is None or
            (now - self.last_model_training).total_seconds() >= self.config.model_retrain_interval):
            
            # Check if we have enough data
            total_samples = sum(len(metrics) for metrics in self.historical_metrics.values())
            return total_samples >= self.config.min_training_samples
        
        return False
    
    async def _collect_and_process_data(self):
        """Collect and process on-chain data for all symbols."""
        self.logger.info("Collecting on-chain data")
        
        collection_tasks = []
        for symbol in self.symbols:
            if self._should_collect_for_symbol(symbol):
                task = self._collect_symbol_data(symbol)
                collection_tasks.append(task)
        
        if collection_tasks:
            results = await asyncio.gather(*collection_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                symbol = self.symbols[i] if i < len(self.symbols) else "unknown"
                
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to collect data for {symbol}: {result}")
                    self.metrics.errors.inc()
                else:
                    self.logger.info(f"Successfully collected data for {symbol}")
                    self.metrics.data_points_collected.inc()
    
    def _should_collect_for_symbol(self, symbol: str) -> bool:
        """Check if we should collect data for a specific symbol."""
        now = datetime.utcnow()
        last_collection = self.last_collection_time.get(symbol)
        
        return (last_collection is None or 
                (now - last_collection).total_seconds() >= self.config.collection_interval)
    
    async def _collect_symbol_data(self, symbol: str):
        """Collect and process data for a single symbol."""
        self.logger.info(f"Collecting data for {symbol}")
        
        since = datetime.utcnow() - timedelta(days=self.config.lookback_days)
        until = datetime.utcnow()
        
        # Collect metrics from available sources
        all_metrics = []
        
        # Glassnode data
        if self.glassnode_client:
            try:
                async with self.glassnode_client:
                    glassnode_metrics = await self.glassnode_client.get_comprehensive_metrics(
                        symbol=symbol, since=since, until=until
                    )
                    all_metrics.extend(glassnode_metrics)
                    
                self.api_health["glassnode"] = {
                    "status": "healthy",
                    "last_check": datetime.utcnow()
                }
                
            except GlassnodeAPIError as e:
                self.logger.error(f"Glassnode API error for {symbol}: {e}")
                self.api_health["glassnode"] = {
                    "status": "error",
                    "last_check": datetime.utcnow(),
                    "error": str(e)
                }
        
        # CryptoQuant data
        if self.cryptoquant_client:
            try:
                async with self.cryptoquant_client:
                    cq_data = await self.cryptoquant_client.get_comprehensive_institutional_data(
                        symbol=symbol, since=since, until=until
                    )
                    
                    # Convert CryptoQuant data to OnChainMetrics
                    cq_metrics = self._convert_cryptoquant_data(cq_data, symbol)
                    all_metrics.extend(cq_metrics)
                    
                self.api_health["cryptoquant"] = {
                    "status": "healthy", 
                    "last_check": datetime.utcnow()
                }
                
            except CryptoQuantAPIError as e:
                self.logger.error(f"CryptoQuant API error for {symbol}: {e}")
                self.api_health["cryptoquant"] = {
                    "status": "error",
                    "last_check": datetime.utcnow(),
                    "error": str(e)
                }
        
        # Store and process metrics
        if all_metrics:
            # Update historical data
            if symbol not in self.historical_metrics:
                self.historical_metrics[symbol] = []
            
            self.historical_metrics[symbol].extend(all_metrics)
            
            # Keep only recent data to manage memory
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            self.historical_metrics[symbol] = [
                m for m in self.historical_metrics[symbol] 
                if m.timestamp >= cutoff_date
            ]
            
            # Publish metrics to Kafka
            await self._publish_metrics(all_metrics, symbol)
            
            # Generate signals
            await self._generate_and_publish_signals(all_metrics, symbol)
            
            self.last_collection_time[symbol] = datetime.utcnow()
            
            self.logger.info(f"Processed {len(all_metrics)} metrics for {symbol}")
        else:
            self.logger.warning(f"No metrics collected for {symbol}")
    
    def _convert_cryptoquant_data(
        self,
        cq_data: Dict[str, List[Any]],
        symbol: str
    ) -> List[OnChainMetrics]:
        """Convert CryptoQuant data to OnChainMetrics format."""
        metrics = []
        
        # Process exchange flows
        for flow in cq_data.get("exchange_flows", []):
            metric = OnChainMetrics(
                symbol=symbol,
                timestamp=flow.timestamp,
                exchange_inflow_btc=flow.total_exchange_inflow_1h,
                exchange_outflow_btc=flow.total_exchange_outflow_1h,
                exchange_net_flow_btc=flow.total_exchange_net_flow_1h
            )
            metrics.append(metric)
        
        return metrics
    
    async def _publish_metrics(self, metrics: List[OnChainMetrics], symbol: str):
        """Publish metrics to Kafka."""
        if not self.kafka_producer:
            return
        
        for metric in metrics:
            try:
                message = {
                    "symbol": metric.symbol,
                    "timestamp": metric.timestamp.isoformat(),
                    "exchange_inflow_btc": metric.exchange_inflow_btc,
                    "exchange_outflow_btc": metric.exchange_outflow_btc,
                    "exchange_net_flow_btc": metric.exchange_net_flow_btc,
                    "active_addresses_24h": metric.active_addresses_24h,
                    "transaction_count_24h": metric.transaction_count_24h,
                    "whale_transaction_count": metric.whale_transaction_count,
                    "hash_rate_7d_ma": metric.hash_rate_7d_ma,
                    "mvrv_ratio": metric.mvrv_ratio,
                    "nvt_ratio": metric.nvt_ratio,
                    "source": "onchain_data_service"
                }
                
                self.kafka_producer.send(
                    self.config.kafka_topic_onchain_metrics,
                    key=f"{symbol}_{int(metric.timestamp.timestamp())}",
                    value=message
                )
                
                self.metrics.messages_sent.inc()
                
            except KafkaError as e:
                self.logger.error(f"Failed to publish metric to Kafka: {e}")
                self.metrics.errors.inc()
    
    async def _generate_and_publish_signals(
        self,
        metrics: List[OnChainMetrics],
        symbol: str
    ):
        """Generate and publish trading signals."""
        if not self.signal_generator.is_fitted:
            self.logger.debug(f"Models not trained yet, skipping signal generation for {symbol}")
            return
        
        # Engineer features
        feature_df = self.feature_engineer.engineer_features(
            metrics, normalize=True, fit_scaler=False
        )
        
        if feature_df.empty:
            return
        
        # Convert to feature sets
        feature_sets = self.feature_engineer.create_feature_sets(feature_df, symbol)
        
        # Generate signals
        signals = self.signal_generator.generate_batch_signals(feature_sets, symbol)
        
        if signals:
            # Store signals
            if symbol not in self.historical_signals:
                self.historical_signals[symbol] = []
            
            self.historical_signals[symbol].extend(signals)
            
            # Publish signals
            await self._publish_signals(signals)
            
            # Generate and publish alerts
            alerts = self.signal_generator.generate_alerts(signals)
            if alerts:
                await self._publish_alerts(alerts)
            
            self.logger.info(f"Generated and published {len(signals)} signals for {symbol}")
    
    async def _publish_signals(self, signals: List[OnChainSignal]):
        """Publish signals to Kafka."""
        if not self.kafka_producer:
            return
        
        for signal in signals:
            try:
                message = {
                    "signal_id": signal.signal_id,
                    "symbol": signal.symbol,
                    "timestamp": signal.timestamp.isoformat(),
                    "signal_type": signal.signal_type,
                    "strength": signal.strength.value,
                    "confidence": signal.confidence,
                    "score": signal.score,
                    "primary_metrics": [m.value for m in signal.primary_metrics],
                    "time_horizon": signal.time_horizon,
                    "model_version": signal.model_version
                }
                
                self.kafka_producer.send(
                    self.config.kafka_topic_onchain_signals,
                    key=signal.signal_id,
                    value=message
                )
                
                self.metrics.signals_generated.inc()
                
            except KafkaError as e:
                self.logger.error(f"Failed to publish signal to Kafka: {e}")
                self.metrics.errors.inc()
    
    async def _publish_alerts(self, alerts: List[OnChainAlert]):
        """Publish alerts to Kafka."""
        if not self.kafka_producer:
            return
        
        for alert in alerts:
            try:
                message = {
                    "alert_id": alert.alert_id,
                    "symbol": alert.symbol,
                    "timestamp": alert.timestamp.isoformat(),
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "title": alert.title,
                    "description": alert.description,
                    "metric_type": alert.metric_type.value,
                    "threshold_value": alert.threshold_value,
                    "actual_value": alert.actual_value,
                    "confidence": alert.confidence,
                    "recommended_action": alert.recommended_action
                }
                
                self.kafka_producer.send(
                    self.config.kafka_topic_onchain_alerts,
                    key=alert.alert_id,
                    value=message
                )
                
                self.metrics.alerts_generated.inc()
                
            except KafkaError as e:
                self.logger.error(f"Failed to publish alert to Kafka: {e}")
                self.metrics.errors.inc()
    
    async def _retrain_models(self):
        """Retrain ML models with recent data."""
        self.logger.info("Retraining ML models")
        
        # Gather all historical metrics
        all_feature_sets = []
        
        for symbol, metrics in self.historical_metrics.items():
            if len(metrics) < 10:  # Need minimum data
                continue
            
            # Engineer features
            feature_df = self.feature_engineer.engineer_features(
                metrics, normalize=True, fit_scaler=True
            )
            
            if not feature_df.empty:
                feature_sets = self.feature_engineer.create_feature_sets(feature_df, symbol)
                all_feature_sets.extend(feature_sets)
        
        if len(all_feature_sets) >= self.config.min_training_samples:
            try:
                # Train models
                metrics = self.signal_generator.train_models(all_feature_sets)
                
                # Save models
                os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
                self.signal_generator.save_models(self.config.model_save_path)
                
                self.last_model_training = datetime.utcnow()
                
                self.logger.info(f"Models retrained successfully: {metrics}")
                
            except Exception as e:
                self.logger.error(f"Failed to retrain models: {e}")
                self.metrics.errors.inc()
        else:
            self.logger.warning(f"Insufficient data for retraining: {len(all_feature_sets)} < {self.config.min_training_samples}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "status": "healthy" if self.running else "stopped",
            "api_health": self.api_health,
            "symbols": self.symbols,
            "data_points": {
                symbol: len(metrics) 
                for symbol, metrics in self.historical_metrics.items()
            },
            "signals_generated": {
                symbol: len(signals)
                for symbol, signals in self.historical_signals.items()  
            },
            "model_status": self.signal_generator.get_model_summary(),
            "last_collection": self.last_collection_time,
            "last_model_training": self.last_model_training.isoformat() if self.last_model_training else None
        }


async def main():
    """Main entry point."""
    # Load configuration
    config = OnChainDataConfig()
    
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
    
    # Create service
    service = OnChainDataService(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        asyncio.create_task(service.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await service.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
