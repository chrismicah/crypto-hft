"""Risk Manager Service with Bayesian Online Changepoint Detection."""

import asyncio
import json
import signal
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import aiohttp
from aiohttp import web
import redis.asyncio as redis

from .bocd import BOCDWrapper, AdaptiveBOCDWrapper, ChangePointEvent
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, HaltEvent, RecoveryEvent, RiskState
from common.logger import configure_logging, get_logger, get_trade_logger, get_performance_logger
from common.config import BaseSettings

# Configure logging
configure_logging(
    service_name="risk-manager-service",
    log_level="INFO",
    log_format="json"
)

logger = get_logger(__name__)
trade_logger = get_trade_logger("risk-manager-service")
perf_logger = get_performance_logger("risk-manager-service")


class RiskManagerSettings(BaseSettings):
    """Settings for Risk Manager service."""
    
    # Kafka settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_spread_topic: str = "spread-data"
    kafka_halt_topic: str = "trading-halt"
    kafka_recovery_topic: str = "trading-recovery"
    kafka_consumer_group: str = "risk-manager-group"
    
    # BOCD settings
    bocd_hazard_rate: float = 1/250
    bocd_max_run_length: int = 500
    bocd_min_observations: int = 10
    use_adaptive_bocd: bool = True
    
    # Circuit breaker settings
    warning_threshold: float = 0.5
    halt_threshold: float = 0.7
    critical_threshold: float = 0.9
    min_halt_duration_minutes: int = 5
    max_halt_duration_hours: int = 2
    recovery_observation_period_minutes: int = 10
    max_consecutive_warnings: int = 3
    max_consecutive_high_prob_events: int = 2
    recovery_probability_threshold: float = 0.2
    
    # Service settings
    health_check_port: int = 8002
    metrics_port: int = 8003
    redis_url: str = "redis://localhost:6379"
    service_name: str = "risk-manager-service"
    service_version: str = "1.0.0"
    
    # Monitoring settings
    status_update_interval_seconds: int = 30
    metrics_update_interval_seconds: int = 10


class RiskManagerService:
    """Main Risk Manager service class."""
    
    def __init__(self, settings: RiskManagerSettings):
        """Initialize Risk Manager service."""
        self.settings = settings
        self.running = False
        
        # Components
        self.bocd: Optional[BOCDWrapper] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.kafka_consumer: Optional[KafkaConsumer] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Health check app
        self.health_app: Optional[web.Application] = None
        self.health_runner: Optional[web.AppRunner] = None
        
        # Metrics
        self.metrics = {
            'events_processed': 0,
            'changepoints_detected': 0,
            'halts_triggered': 0,
            'recoveries_completed': 0,
            'last_event_time': None,
            'current_risk_state': RiskState.NORMAL.value,
            'service_start_time': datetime.utcnow()
        }
        
        logger.info("Risk Manager service initialized", 
                   service_name=settings.service_name,
                   version=settings.service_version)
    
    async def start(self) -> None:
        """Start the Risk Manager service."""
        logger.info("Starting Risk Manager service")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start health check server
            await self._start_health_server()
            
            # Start background tasks
            self.running = True
            
            # Create tasks
            tasks = [
                asyncio.create_task(self._consume_spread_data()),
                asyncio.create_task(self._update_status_periodically()),
                asyncio.create_task(self._update_metrics_periodically())
            ]
            
            logger.info("Risk Manager service started successfully")
            
            # Wait for tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error("Failed to start Risk Manager service", error=str(e), exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the Risk Manager service."""
        logger.info("Stopping Risk Manager service")
        
        self.running = False
        
        # Close connections
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Stop health server
        if self.health_runner:
            await self.health_runner.cleanup()
        
        logger.info("Risk Manager service stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize all service components."""
        # Initialize BOCD
        if self.settings.use_adaptive_bocd:
            self.bocd = AdaptiveBOCDWrapper(
                base_hazard_rate=self.settings.bocd_hazard_rate,
                max_run_length=self.settings.bocd_max_run_length,
                min_observations=self.settings.bocd_min_observations
            )
        else:
            self.bocd = BOCDWrapper(
                hazard_rate=self.settings.bocd_hazard_rate,
                max_run_length=self.settings.bocd_max_run_length,
                min_observations=self.settings.bocd_min_observations
            )
        
        # Initialize circuit breaker
        circuit_config = CircuitBreakerConfig(
            warning_threshold=self.settings.warning_threshold,
            halt_threshold=self.settings.halt_threshold,
            critical_threshold=self.settings.critical_threshold,
            min_halt_duration=timedelta(minutes=self.settings.min_halt_duration_minutes),
            max_halt_duration=timedelta(hours=self.settings.max_halt_duration_hours),
            recovery_observation_period=timedelta(minutes=self.settings.recovery_observation_period_minutes),
            max_consecutive_warnings=self.settings.max_consecutive_warnings,
            max_consecutive_high_prob_events=self.settings.max_consecutive_high_prob_events,
            recovery_probability_threshold=self.settings.recovery_probability_threshold
        )
        
        self.circuit_breaker = CircuitBreaker(
            config=circuit_config,
            halt_callback=self._handle_halt_event,
            recovery_callback=self._handle_recovery_event
        )
        
        # Initialize Kafka
        self.kafka_consumer = KafkaConsumer(
            self.settings.kafka_spread_topic,
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            group_id=self.settings.kafka_consumer_group,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks='all',
            retries=3
        )
        
        # Initialize Redis
        self.redis_client = redis.from_url(self.settings.redis_url)
        
        logger.info("All components initialized successfully")
    
    async def _consume_spread_data(self) -> None:
        """Consume spread data from Kafka and process with BOCD."""
        logger.info("Starting spread data consumption",
                   topic=self.settings.kafka_spread_topic)
        
        while self.running:
            try:
                # Poll for messages
                message_pack = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        await self._process_spread_message(message.value)
                
            except KafkaError as e:
                logger.error("Kafka error in spread data consumption", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
                
            except Exception as e:
                logger.error("Error processing spread data", error=str(e), exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_spread_message(self, message: Dict[str, Any]) -> None:
        """Process a single spread data message."""
        try:
            # Extract spread value and timestamp
            spread_value = message.get('spread_value')
            timestamp_str = message.get('timestamp')
            
            if spread_value is None or timestamp_str is None:
                logger.warning("Invalid spread message format", message=message)
                return
            
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Update BOCD
            changepoint_prob, changepoint_event = self.bocd.update(spread_value, timestamp)
            
            # Update metrics
            self.metrics['events_processed'] += 1
            self.metrics['last_event_time'] = timestamp
            
            # Log significant events
            if changepoint_prob > 0.1:
                logger.debug("Changepoint probability updated",
                           spread_value=spread_value,
                           probability=changepoint_prob,
                           timestamp=timestamp)
            
            # Process with circuit breaker if we have a significant event
            if changepoint_event:
                halt_event = self.circuit_breaker.process_changepoint_event(changepoint_event)
                
                if halt_event:
                    self.metrics['halts_triggered'] += 1
                
                # Update risk state metric
                self.metrics['current_risk_state'] = self.circuit_breaker.current_state.value
                
                # Log changepoint detection
                if changepoint_prob > self.settings.warning_threshold:
                    self.metrics['changepoints_detected'] += 1
                    
                    trade_logger.info("Changepoint detected",
                                    probability=changepoint_prob,
                                    confidence=changepoint_event.confidence_level,
                                    spread_value=spread_value,
                                    risk_state=self.circuit_breaker.current_state.value)
            
        except Exception as e:
            logger.error("Error processing spread message", error=str(e), message=message)
    
    def _handle_halt_event(self, halt_event: HaltEvent) -> None:
        """Handle trading halt event."""
        logger.critical("Trading halt triggered",
                       reason=halt_event.reason.value,
                       severity=halt_event.severity,
                       probability=halt_event.probability)
        
        # Publish halt message to Kafka
        try:
            halt_message = halt_event.to_kafka_message()
            
            self.kafka_producer.send(
                self.settings.kafka_halt_topic,
                value=halt_message
            ).get(timeout=10)  # Wait for confirmation
            
            logger.info("Halt message published to Kafka",
                       topic=self.settings.kafka_halt_topic)
            
            # Store in Redis for persistence
            asyncio.create_task(self._store_halt_event(halt_event))
            
        except Exception as e:
            logger.error("Failed to publish halt message", error=str(e))
    
    def _handle_recovery_event(self, recovery_event: RecoveryEvent) -> None:
        """Handle trading recovery event."""
        logger.info("Trading recovery triggered",
                   previous_reason=recovery_event.previous_halt_reason.value,
                   conditions=recovery_event.recovery_conditions_met)
        
        self.metrics['recoveries_completed'] += 1
        
        # Publish recovery message to Kafka
        try:
            recovery_message = recovery_event.to_kafka_message()
            
            self.kafka_producer.send(
                self.settings.kafka_recovery_topic,
                value=recovery_message
            ).get(timeout=10)
            
            logger.info("Recovery message published to Kafka",
                       topic=self.settings.kafka_recovery_topic)
            
            # Store in Redis
            asyncio.create_task(self._store_recovery_event(recovery_event))
            
        except Exception as e:
            logger.error("Failed to publish recovery message", error=str(e))
    
    async def _store_halt_event(self, halt_event: HaltEvent) -> None:
        """Store halt event in Redis."""
        try:
            key = f"risk:halt:{halt_event.timestamp.isoformat()}"
            value = json.dumps(halt_event.to_kafka_message())
            
            await self.redis_client.setex(key, 86400 * 7, value)  # Store for 7 days
            
            # Update latest halt
            await self.redis_client.set("risk:latest_halt", value)
            
        except Exception as e:
            logger.error("Failed to store halt event in Redis", error=str(e))
    
    async def _store_recovery_event(self, recovery_event: RecoveryEvent) -> None:
        """Store recovery event in Redis."""
        try:
            key = f"risk:recovery:{recovery_event.timestamp.isoformat()}"
            value = json.dumps(recovery_event.to_kafka_message())
            
            await self.redis_client.setex(key, 86400 * 7, value)  # Store for 7 days
            
            # Update latest recovery
            await self.redis_client.set("risk:latest_recovery", value)
            
        except Exception as e:
            logger.error("Failed to store recovery event in Redis", error=str(e))
    
    async def _update_status_periodically(self) -> None:
        """Update service status in Redis periodically."""
        while self.running:
            try:
                status = {
                    'service_name': self.settings.service_name,
                    'version': self.settings.service_version,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'running',
                    'circuit_breaker_status': self.circuit_breaker.get_status() if self.circuit_breaker else {},
                    'bocd_statistics': self.bocd.get_statistics() if self.bocd else {},
                    'metrics': self.metrics
                }
                
                await self.redis_client.setex(
                    f"service:status:{self.settings.service_name}",
                    self.settings.status_update_interval_seconds * 2,
                    json.dumps(status)
                )
                
                logger.debug("Status updated in Redis")
                
            except Exception as e:
                logger.error("Failed to update status in Redis", error=str(e))
            
            await asyncio.sleep(self.settings.status_update_interval_seconds)
    
    async def _update_metrics_periodically(self) -> None:
        """Update metrics in Redis periodically."""
        while self.running:
            try:
                # Performance metrics
                perf_metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'events_processed_rate': self.metrics['events_processed'] / 
                                           max(1, (datetime.utcnow() - self.metrics['service_start_time']).total_seconds()),
                    'current_risk_state': self.metrics['current_risk_state'],
                    'total_changepoints': self.metrics['changepoints_detected'],
                    'total_halts': self.metrics['halts_triggered'],
                    'total_recoveries': self.metrics['recoveries_completed']
                }
                
                perf_logger.info("Performance metrics", **perf_metrics)
                
                # Store in Redis
                await self.redis_client.setex(
                    f"metrics:{self.settings.service_name}",
                    self.settings.metrics_update_interval_seconds * 2,
                    json.dumps(perf_metrics)
                )
                
            except Exception as e:
                logger.error("Failed to update metrics", error=str(e))
            
            await asyncio.sleep(self.settings.metrics_update_interval_seconds)
    
    async def _start_health_server(self) -> None:
        """Start health check HTTP server."""
        self.health_app = web.Application()
        
        # Health check endpoint
        self.health_app.router.add_get('/health', self._health_check)
        self.health_app.router.add_get('/status', self._status_check)
        self.health_app.router.add_get('/metrics', self._metrics_check)
        
        # Manual control endpoints
        self.health_app.router.add_post('/halt', self._manual_halt)
        self.health_app.router.add_post('/recover', self._manual_recover)
        
        self.health_runner = web.AppRunner(self.health_app)
        await self.health_runner.setup()
        
        site = web.TCPSite(self.health_runner, '0.0.0.0', self.settings.health_check_port)
        await site.start()
        
        logger.info("Health check server started", port=self.settings.health_check_port)
    
    async def _health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        try:
            # Check component health
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'service': self.settings.service_name,
                'version': self.settings.service_version,
                'components': {
                    'bocd': self.bocd is not None,
                    'circuit_breaker': self.circuit_breaker is not None,
                    'kafka_consumer': self.kafka_consumer is not None,
                    'kafka_producer': self.kafka_producer is not None,
                    'redis': self.redis_client is not None
                }
            }
            
            # Check if any component is unhealthy
            if not all(health_status['components'].values()):
                health_status['status'] = 'unhealthy'
                return web.json_response(health_status, status=503)
            
            return web.json_response(health_status)
            
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
    
    async def _status_check(self, request: web.Request) -> web.Response:
        """Detailed status endpoint."""
        try:
            status = {
                'service_info': {
                    'name': self.settings.service_name,
                    'version': self.settings.service_version,
                    'uptime_seconds': (datetime.utcnow() - self.metrics['service_start_time']).total_seconds()
                },
                'metrics': self.metrics,
                'circuit_breaker': self.circuit_breaker.get_status() if self.circuit_breaker else {},
                'bocd_statistics': self.bocd.get_statistics() if self.bocd else {}
            }
            
            return web.json_response(status)
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _metrics_check(self, request: web.Request) -> web.Response:
        """Prometheus-style metrics endpoint."""
        try:
            metrics_text = f"""# HELP risk_events_processed_total Total events processed
# TYPE risk_events_processed_total counter
risk_events_processed_total {self.metrics['events_processed']}

# HELP risk_changepoints_detected_total Total changepoints detected
# TYPE risk_changepoints_detected_total counter
risk_changepoints_detected_total {self.metrics['changepoints_detected']}

# HELP risk_halts_triggered_total Total halts triggered
# TYPE risk_halts_triggered_total counter
risk_halts_triggered_total {self.metrics['halts_triggered']}

# HELP risk_recoveries_completed_total Total recoveries completed
# TYPE risk_recoveries_completed_total counter
risk_recoveries_completed_total {self.metrics['recoveries_completed']}

# HELP risk_current_state Current risk management state
# TYPE risk_current_state gauge
risk_current_state{{state="{self.metrics['current_risk_state']}"}} 1
"""
            
            return web.Response(text=metrics_text, content_type='text/plain')
            
        except Exception as e:
            return web.Response(text=f"# Error: {str(e)}", status=500)
    
    async def _manual_halt(self, request: web.Request) -> web.Response:
        """Manual halt endpoint."""
        try:
            data = await request.json()
            reason = data.get('reason', 'Manual halt via API')
            
            if self.circuit_breaker:
                halt_event = self.circuit_breaker.manual_halt(reason)
                return web.json_response({
                    'status': 'success',
                    'message': 'Manual halt triggered',
                    'halt_event': halt_event.to_kafka_message()
                })
            else:
                return web.json_response({'error': 'Circuit breaker not initialized'}, status=500)
                
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _manual_recover(self, request: web.Request) -> web.Response:
        """Manual recovery endpoint."""
        try:
            if self.circuit_breaker:
                recovery_event = self.circuit_breaker.manual_recovery()
                
                if recovery_event:
                    return web.json_response({
                        'status': 'success',
                        'message': 'Manual recovery triggered',
                        'recovery_event': recovery_event.to_kafka_message()
                    })
                else:
                    return web.json_response({
                        'status': 'failed',
                        'message': 'Cannot recover - not in halt state'
                    }, status=400)
            else:
                return web.json_response({'error': 'Circuit breaker not initialized'}, status=500)
                
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)


async def main():
    """Main function to run the Risk Manager service."""
    settings = RiskManagerSettings()
    service = RiskManagerService(settings)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(service.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Service failed", error=str(e), exc_info=True)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
