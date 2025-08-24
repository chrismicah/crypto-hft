"""
Main Orchestrator Service for EVOP Framework.

This service manages the evolutionary operation of trading strategies,
coordinating champion-challenger execution and performance monitoring.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional
import logging

import aiohttp
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
import redis.asyncio as redis

from common.logger import get_logger
from common.metrics import ServiceMetrics
from .models import StrategyParameters, EVOPConfiguration
from .champion_challenger import ChampionChallengerManager


class OrchestratorService:
    """
    Main orchestrator service for the EVOP framework.
    
    This service:
    - Manages champion-challenger strategy execution
    - Monitors performance from execution services
    - Triggers strategy promotions based on performance
    - Publishes strategy configurations to other services
    - Provides health and status endpoints
    """
    
    def __init__(self):
        """Initialize the orchestrator service."""
        self.logger = get_logger(__name__)
        
        # Configuration
        self.config = self._load_configuration()
        
        # Service state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Kafka integration
        self.kafka_producer = None
        self.kafka_consumer = None
        self.kafka_bootstrap_servers = None
        
        # Redis integration
        self.redis_client = None
        
        # HTTP server for health checks
        self.http_server = None
        self.http_runner = None
        
        # Metrics
        self.metrics = ServiceMetrics(
            service_name="orchestrator",
            registry=None  # Will use default registry
        )
        
        # Champion-Challenger Manager
        self.manager = ChampionChallengerManager(
            config=self.config,
            strategy_executor=self._execute_strategy,
            performance_tracker=self._track_performance
        )
        
        # Background tasks
        self.evaluation_task = None
        self.kafka_consumer_task = None
        self.health_check_task = None
        
    def _load_configuration(self) -> EVOPConfiguration:
        """Load EVOP configuration from environment variables."""
        import os
        
        return EVOPConfiguration(
            max_challengers=int(os.getenv('EVOP_MAX_CHALLENGERS', '3')),
            challenger_capital_fraction=float(os.getenv('EVOP_CHALLENGER_CAPITAL_FRACTION', '0.2')),
            min_evaluation_period_days=int(os.getenv('EVOP_MIN_EVALUATION_DAYS', '7')),
            max_evaluation_period_days=int(os.getenv('EVOP_MAX_EVALUATION_DAYS', '30')),
            min_trades_for_evaluation=int(os.getenv('EVOP_MIN_TRADES', '10')),
            required_confidence_level=float(os.getenv('EVOP_CONFIDENCE_LEVEL', '0.95')),
            min_sharpe_improvement=float(os.getenv('EVOP_MIN_SHARPE_IMPROVEMENT', '0.1')),
            min_calmar_improvement=float(os.getenv('EVOP_MIN_CALMAR_IMPROVEMENT', '0.1')),
            max_drawdown_tolerance=float(os.getenv('EVOP_MAX_DRAWDOWN_TOLERANCE', '0.15')),
            max_total_allocation=float(os.getenv('EVOP_MAX_TOTAL_ALLOCATION', '1.0')),
            emergency_stop_drawdown=float(os.getenv('EVOP_EMERGENCY_STOP_DRAWDOWN', '0.25')),
            parameter_mutation_rate=float(os.getenv('EVOP_MUTATION_RATE', '0.1')),
            parameter_mutation_std=float(os.getenv('EVOP_MUTATION_STD', '0.05')),
            evaluation_frequency_hours=int(os.getenv('EVOP_EVALUATION_FREQUENCY', '6')),
            challenger_restart_on_failure=os.getenv('EVOP_RESTART_ON_FAILURE', 'true').lower() == 'true'
        )
    
    async def start(self):
        """Start the orchestrator service."""
        try:
            self.logger.info("Starting Orchestrator Service...")
            
            # Initialize connections
            await self._initialize_kafka()
            await self._initialize_redis()
            
            # Start HTTP server
            await self._start_http_server()
            
            # Initialize champion strategy
            await self._initialize_champion()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            self.logger.info("Orchestrator Service started successfully")
            
            # Record startup metrics
            self.metrics.record_service_start()
            
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator service: {e}")
            await self.stop()
            raise
    
    async def _initialize_kafka(self):
        """Initialize Kafka producer and consumer."""
        import os
        
        self.kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
        try:
            # Initialize producer for publishing strategy configurations
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            
            # Initialize consumer for performance updates
            self.kafka_consumer = KafkaConsumer(
                'strategy-performance',
                'strategy-status',
                'trading-signals',
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id='orchestrator-service',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.logger.info("Kafka connections initialized successfully")
            
        except NoBrokersAvailable:
            self.logger.error("No Kafka brokers available")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection for service registry."""
        import os
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        try:
            self.redis_client = redis.from_url(redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            # Register service
            service_info = {
                'name': 'orchestrator-service',
                'status': 'starting',
                'started_at': datetime.now().isoformat(),
                'config': self.config.to_dict()
            }
            
            await self.redis_client.setex(
                'service:orchestrator',
                300,  # 5 minute TTL
                json.dumps(service_info)
            )
            
            self.logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _start_http_server(self):
        """Start HTTP server for health checks and API endpoints."""
        import os
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/health', self._health_check)
        app.router.add_get('/status', self._status_endpoint)
        app.router.add_get('/metrics', self._metrics_endpoint)
        app.router.add_post('/promote/{challenger_id}', self._manual_promotion)
        app.router.add_post('/emergency-stop', self._emergency_stop_endpoint)
        
        port = int(os.getenv('ORCHESTRATOR_PORT', '8005'))
        
        self.http_runner = web.AppRunner(app)
        await self.http_runner.setup()
        
        site = web.TCPSite(self.http_runner, '0.0.0.0', port)
        await site.start()
        
        self.logger.info(f"HTTP server started on port {port}")
    
    async def _initialize_champion(self):
        """Initialize the champion strategy with default parameters."""
        import os
        
        # Load champion parameters from environment or use defaults
        champion_params = StrategyParameters(
            entry_z_score=float(os.getenv('CHAMPION_ENTRY_Z_SCORE', '2.0')),
            exit_z_score=float(os.getenv('CHAMPION_EXIT_Z_SCORE', '0.5')),
            stop_loss_z_score=float(os.getenv('CHAMPION_STOP_LOSS_Z_SCORE', '4.0')),
            max_position_size=float(os.getenv('CHAMPION_MAX_POSITION_SIZE', '10000.0')),
            max_drawdown_percent=float(os.getenv('CHAMPION_MAX_DRAWDOWN_PERCENT', '10.0')),
            max_daily_loss=float(os.getenv('CHAMPION_MAX_DAILY_LOSS', '5000.0')),
            kelly_fraction=float(os.getenv('CHAMPION_KELLY_FRACTION', '0.25')),
        )
        
        initial_capital = Decimal(os.getenv('CHAMPION_INITIAL_CAPITAL', '100000.0'))
        
        # Initialize and start champion
        champion = await self.manager.initialize_champion(
            parameters=champion_params,
            initial_capital=initial_capital,
            name="Initial Champion"
        )
        
        await self.manager.start_champion()
        
        # Generate and start challengers
        await self.manager.generate_challengers()
        await self.manager.start_challengers()
        
        # Publish initial strategy configurations
        await self._publish_strategy_configurations()
        
        self.logger.info("Champion and challengers initialized successfully")
    
    async def _start_background_tasks(self):
        """Start background tasks for monitoring and evaluation."""
        # Performance evaluation task
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        
        # Kafka consumer task
        self.kafka_consumer_task = asyncio.create_task(self._kafka_consumer_loop())
        
        # Health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Background tasks started")
    
    async def _evaluation_loop(self):
        """Background task for periodic strategy evaluation."""
        while self.is_running:
            try:
                # Perform evaluations
                promotion_events = await self.manager.evaluate_promotions()
                
                if promotion_events:
                    self.logger.info(f"Processed {len(promotion_events)} promotions")
                    
                    # Publish updated configurations after promotions
                    await self._publish_strategy_configurations()
                    
                    # Record promotion metrics
                    for event in promotion_events:
                        self.metrics.record_custom_metric(
                            'strategy_promotions_total',
                            1,
                            labels={'reason': event.reason.value}
                        )
                
                # Wait for next evaluation cycle
                await asyncio.sleep(self.config.evaluation_frequency_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _kafka_consumer_loop(self):
        """Background task for consuming performance updates from Kafka."""
        while self.is_running:
            try:
                # Get messages from Kafka
                message_pack = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        await self._process_kafka_message(
                            topic_partition.topic,
                            message.value
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in Kafka consumer loop: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds before retry
    
    async def _process_kafka_message(self, topic: str, message: Dict[str, Any]):
        """Process a message from Kafka."""
        try:
            if topic == 'strategy-performance':
                # Update strategy performance
                strategy_id = message.get('strategy_id')
                performance_data = message.get('performance', {})
                
                if strategy_id:
                    success = await self.manager.update_performance(
                        strategy_id, performance_data
                    )
                    
                    if success:
                        self.logger.debug(f"Updated performance for strategy {strategy_id}")
                    
            elif topic == 'strategy-status':
                # Handle strategy status updates
                strategy_id = message.get('strategy_id')
                status = message.get('status')
                
                if strategy_id and status:
                    self.logger.info(f"Strategy {strategy_id} status: {status}")
                    
            elif topic == 'trading-signals':
                # Track trading signals for analysis
                signal_type = message.get('signal_type')
                if signal_type:
                    self.metrics.record_custom_metric(
                        'trading_signals_total',
                        1,
                        labels={'signal_type': signal_type}
                    )
                    
        except Exception as e:
            self.logger.error(f"Error processing Kafka message: {e}")
    
    async def _health_check_loop(self):
        """Background task for service health checks and registration."""
        while self.is_running:
            try:
                # Update service registration in Redis
                if self.redis_client:
                    service_info = {
                        'name': 'orchestrator-service',
                        'status': 'running',
                        'last_heartbeat': datetime.now().isoformat(),
                        'active_strategies': len([
                            s for s in [self.manager.champion] + list(self.manager.challengers.values())
                            if s and s.is_active()
                        ]),
                        'config': self.config.to_dict()
                    }
                    
                    await self.redis_client.setex(
                        'service:orchestrator',
                        300,  # 5 minute TTL
                        json.dumps(service_info)
                    )
                
                # Check for emergency conditions
                await self._check_emergency_conditions()
                
                # Wait for next health check
                await asyncio.sleep(30)  # Every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_emergency_conditions(self):
        """Check for emergency conditions that require stopping all strategies."""
        if not self.manager.champion:
            return
        
        # Check for excessive drawdown
        current_dd = self.manager.champion.performance.current_drawdown
        if current_dd > self.config.emergency_stop_drawdown:
            self.logger.critical(
                f"Emergency stop triggered: drawdown {current_dd:.2%} exceeds "
                f"threshold {self.config.emergency_stop_drawdown:.2%}"
            )
            
            await self.manager.emergency_stop("Excessive drawdown detected")
            
            # Record emergency stop metric
            self.metrics.record_custom_metric(
                'emergency_stops_total',
                1,
                labels={'reason': 'excessive_drawdown'}
            )
    
    async def _publish_strategy_configurations(self):
        """Publish current strategy configurations to Kafka."""
        try:
            configurations = {}
            
            # Add champion configuration
            if self.manager.champion:
                configurations['champion'] = {
                    'strategy_id': self.manager.champion.instance_id,
                    'parameters': self.manager.champion.parameters.to_dict(),
                    'allocated_capital': float(self.manager.champion.allocated_capital),
                    'is_champion': True
                }
            
            # Add challenger configurations
            for challenger in self.manager.challengers.values():
                configurations[f'challenger_{challenger.instance_id}'] = {
                    'strategy_id': challenger.instance_id,
                    'parameters': challenger.parameters.to_dict(),
                    'allocated_capital': float(challenger.allocated_capital),
                    'is_champion': False
                }
            
            # Publish to Kafka
            message = {
                'timestamp': datetime.now().isoformat(),
                'configurations': configurations,
                'total_strategies': len(configurations)
            }
            
            self.kafka_producer.send(
                'strategy-configurations',
                key='orchestrator',
                value=message
            )
            
            self.kafka_producer.flush()
            
            self.logger.debug(f"Published {len(configurations)} strategy configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to publish strategy configurations: {e}")
    
    async def _execute_strategy(self, strategy_instance):
        """
        Mock strategy execution function.
        
        In a real implementation, this would interface with the execution service
        to run the strategy with the specified parameters.
        """
        self.logger.info(f"Starting execution for strategy: {strategy_instance.name}")
        
        try:
            # Publish strategy configuration
            strategy_config = {
                'strategy_id': strategy_instance.instance_id,
                'parameters': strategy_instance.parameters.to_dict(),
                'allocated_capital': float(strategy_instance.allocated_capital),
                'is_champion': strategy_instance.is_champion
            }
            
            self.kafka_producer.send(
                'strategy-start',
                key=strategy_instance.instance_id,
                value=strategy_config
            )
            
            # In real implementation, this would coordinate with execution service
            # For now, we'll simulate with a long-running task
            while strategy_instance.status == strategy_instance.status.RUNNING:
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            self.logger.info(f"Strategy execution cancelled: {strategy_instance.name}")
        except Exception as e:
            self.logger.error(f"Error in strategy execution: {e}")
            strategy_instance.status = strategy_instance.status.FAILED
    
    async def _track_performance(self, strategy_instance):
        """Mock performance tracking function."""
        # This would interface with the performance tracking system
        pass
    
    # HTTP Endpoints
    
    async def _health_check(self, request):
        """Health check endpoint."""
        from aiohttp import web
        
        status = {
            'status': 'healthy' if self.is_running else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (
                (datetime.now() - self.start_time).total_seconds()
                if hasattr(self, 'start_time') else 0
            ),
            'active_strategies': len([
                s for s in [self.manager.champion] + list(self.manager.challengers.values())
                if s and s.is_active()
            ])
        }
        
        return web.json_response(status)
    
    async def _status_endpoint(self, request):
        """Detailed status endpoint."""
        from aiohttp import web
        
        status = self.manager.get_status_summary()
        return web.json_response(status)
    
    async def _metrics_endpoint(self, request):
        """Prometheus metrics endpoint."""
        from aiohttp import web
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        metrics_data = generate_latest()
        return web.Response(body=metrics_data, content_type=CONTENT_TYPE_LATEST)
    
    async def _manual_promotion(self, request):
        """Manual promotion endpoint."""
        from aiohttp import web
        
        challenger_id = request.match_info['challenger_id']
        
        if challenger_id not in self.manager.challengers:
            return web.json_response(
                {'error': 'Challenger not found'},
                status=404
            )
        
        try:
            challenger = self.manager.challengers[challenger_id]
            
            # Create manual promotion comparison result
            from .performance_comparator import ComparisonResult
            from .models import PromotionReason
            
            comparison = ComparisonResult(
                should_promote=True,
                confidence_score=1.0,
                reason=PromotionReason.MANUAL_OVERRIDE,
                details={'manual': True}
            )
            
            promotion_event = await self.manager._promote_challenger(challenger, comparison)
            
            if promotion_event:
                await self._publish_strategy_configurations()
                return web.json_response({
                    'success': True,
                    'promotion_event': promotion_event.to_dict()
                })
            else:
                return web.json_response(
                    {'error': 'Promotion failed'},
                    status=500
                )
                
        except Exception as e:
            self.logger.error(f"Manual promotion failed: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def _emergency_stop_endpoint(self, request):
        """Emergency stop endpoint."""
        from aiohttp import web
        
        try:
            data = await request.json()
            reason = data.get('reason', 'Manual emergency stop via API')
            
            success = await self.manager.emergency_stop(reason)
            
            return web.json_response({
                'success': success,
                'message': 'Emergency stop executed',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def stop(self):
        """Stop the orchestrator service."""
        self.logger.info("Stopping Orchestrator Service...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop all strategies
        if self.manager:
            await self.manager.emergency_stop("Service shutdown")
        
        # Cancel background tasks
        tasks = [self.evaluation_task, self.kafka_consumer_task, self.health_check_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close connections
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.http_runner:
            await self.http_runner.cleanup()
        
        self.logger.info("Orchestrator Service stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for the orchestrator service."""
    service = OrchestratorService()
    service.setup_signal_handlers()
    
    try:
        service.start_time = datetime.now()
        await service.start()
        
        # Wait for shutdown signal
        await service.shutdown_event.wait()
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Service failed: {e}")
        return 1
    finally:
        await service.stop()
    
    return 0


if __name__ == "__main__":
    import os
    
    # Configure logging
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the service
    sys.exit(asyncio.run(main()))
