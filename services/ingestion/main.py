"""Main ingestion service for real-time market data."""

import asyncio
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import structlog

from .config import settings
from .websocket_client import BinanceWebSocketClient, WebSocketMessage, parse_order_book_update
from .kafka_producer import IngestionKafkaProducer
from .service_registry import ServiceRegistry, HealthStatus
from common.logger import configure_logging, get_logger, get_trade_logger, get_performance_logger

# Configure logging
configure_logging(
    service_name=settings.service_name,
    log_level=settings.log_level,
    log_format=settings.log_format
)

logger = get_logger(__name__)
trade_logger = get_trade_logger(settings.service_name)
perf_logger = get_performance_logger(settings.service_name)

# Prometheus metrics
MESSAGE_COUNTER = Counter('websocket_messages_total', 'Total WebSocket messages received', ['symbol'])
ERROR_COUNTER = Counter('errors_total', 'Total errors', ['component', 'error_type'])
ORDER_BOOK_UPDATE_HISTOGRAM = Histogram('order_book_update_duration_seconds', 'Time spent processing order book updates')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections', ['type'])
KAFKA_MESSAGES_SENT = Counter('kafka_messages_sent_total', 'Total Kafka messages sent', ['topic'])


class IngestionService:
    """Main ingestion service class."""
    
    def __init__(self):
        self.websocket_client: Optional[BinanceWebSocketClient] = None
        self.kafka_producer: Optional[IngestionKafkaProducer] = None
        self.service_registry: Optional[ServiceRegistry] = None
        self.health_status = HealthStatus()
        self.running = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the ingestion service."""
        startup_start = datetime.utcnow()
        
        logger.info("Starting Ingestion Service", 
                   symbols=settings.symbols_list,
                   testnet=settings.binance_testnet)
        
        try:
            # Start Prometheus metrics server
            start_http_server(settings.prometheus_port)
            logger.info("Prometheus metrics server started", port=settings.prometheus_port)
            
            # Initialize components
            await self._initialize_components()
            
            self.running = True
            self.health_status.websocket_connected = True
            self.health_status.kafka_connected = True
            self.health_status.redis_connected = True
            
            startup_time = (datetime.utcnow() - startup_start).total_seconds()
            perf_logger.log_service_startup(startup_time)
            
            logger.info("Ingestion Service started successfully")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Keep the application running
            await self._run_forever()
            
        except Exception as e:
            logger.error("Error starting ingestion service", error=str(e))
            ERROR_COUNTER.labels(component='startup', error_type=type(e).__name__).inc()
            raise
    
    async def stop(self) -> None:
        """Stop the ingestion service."""
        logger.info("Stopping Ingestion Service")
        
        self.running = False
        
        # Stop heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect components
        if self.websocket_client:
            await self.websocket_client.disconnect()
        
        if self.kafka_producer:
            await self.kafka_producer.disconnect()
        
        if self.service_registry:
            await self.service_registry.disconnect()
        
        uptime = (datetime.utcnow() - self.health_status.start_time).total_seconds()
        perf_logger.log_service_shutdown(uptime)
        
        logger.info("Ingestion Service stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize all service components."""
        # Initialize Kafka producer
        self.kafka_producer = IngestionKafkaProducer()
        kafka_connected = await self.kafka_producer.connect()
        
        if not kafka_connected:
            raise Exception("Failed to connect to Kafka")
        
        # Initialize WebSocket client
        self.websocket_client = BinanceWebSocketClient(
            symbols=settings.symbols_list,
            on_message_callback=self._on_websocket_message
        )
        
        websocket_connected = await self.websocket_client.connect()
        
        if not websocket_connected:
            raise Exception("Failed to connect to WebSocket")
        
        # Initialize service registry
        self.service_registry = ServiceRegistry()
        registry_connected = await self.service_registry.connect()
        
        if not registry_connected:
            logger.warning("Failed to connect to service registry, continuing without it")
        
        logger.info("All components initialized successfully")
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Start WebSocket listener
        asyncio.create_task(self.websocket_client.listen())
        
        # Start heartbeat task if service registry is available
        if self.service_registry and self.service_registry.connected:
            self.heartbeat_task = asyncio.create_task(
                self.service_registry.start_heartbeat_task(self._get_health_data)
            )
    
    async def _run_forever(self) -> None:
        """Keep the application running."""
        try:
            while self.running:
                # Update health status
                await self._update_health_status()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error("Error in main loop", error=str(e))
        finally:
            await self.stop()
    
    async def _update_health_status(self) -> None:
        """Update the health status."""
        try:
            if self.websocket_client:
                stats = self.websocket_client.get_stats()
                
                self.health_status.websocket_connected = stats.get('connected', False)
                self.health_status.message_count = stats.get('message_count', 0)
                self.health_status.error_count = stats.get('error_count', 0)
                
                if stats.get('last_message_time'):
                    self.health_status.last_message_time = datetime.fromisoformat(
                        stats['last_message_time']
                    )
            
            if self.kafka_producer:
                kafka_stats = self.kafka_producer.get_stats()
                self.health_status.kafka_connected = kafka_stats.get('connected', False)
                self.health_status.error_count += kafka_stats.get('error_count', 0)
            
            # Update Prometheus metrics
            ACTIVE_CONNECTIONS.labels(type='websocket').set(
                1 if self.health_status.websocket_connected else 0
            )
            ACTIVE_CONNECTIONS.labels(type='kafka').set(
                1 if self.health_status.kafka_connected else 0
            )
            
        except Exception as e:
            logger.error("Error updating health status", error=str(e))
    
    async def _on_websocket_message(self, ws_message: WebSocketMessage) -> None:
        """Handle incoming WebSocket messages."""
        start_time = datetime.utcnow()
        
        try:
            # Parse order book update
            order_book_update = parse_order_book_update(ws_message)
            
            if order_book_update:
                # Publish to Kafka
                await self.kafka_producer.publish_order_book_update(order_book_update)
                
                # Extract price for trade tick
                if order_book_update.bids and order_book_update.asks:
                    # Use mid-price as trade tick
                    best_bid = max(order_book_update.bids, key=lambda x: x.price)
                    best_ask = min(order_book_update.asks, key=lambda x: x.price)
                    mid_price = (best_bid.price + best_ask.price) / 2
                    
                    # Publish trade tick for pairs trading
                    await self.kafka_producer.publish_trade_tick(
                        symbol=order_book_update.symbol,
                        price=mid_price,
                        quantity=1.0,  # Synthetic quantity for mid-price
                        timestamp=order_book_update.timestamp
                    )
                
                # Update metrics
                MESSAGE_COUNTER.labels(symbol=order_book_update.symbol).inc()
                KAFKA_MESSAGES_SENT.labels(topic=settings.kafka_topic_order_book).inc()
                KAFKA_MESSAGES_SENT.labels(topic=settings.kafka_topic_trade_ticks).inc()
                
                # Log trade event
                trade_logger.log_signal_generated(
                    pair_id=order_book_update.symbol,
                    signal_type="order_book_update",
                    signal_strength="normal",
                    confidence=1.0
                )
            
            # Record processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            ORDER_BOOK_UPDATE_HISTOGRAM.observe(processing_time)
            
        except Exception as e:
            logger.error("Error processing WebSocket message", error=str(e))
            ERROR_COUNTER.labels(component='websocket', error_type=type(e).__name__).inc()
    
    def _get_health_data(self) -> Dict[str, Any]:
        """Get current health data for service registry."""
        return self.health_status.to_dict()


# Global service instance
ingestion_service = IngestionService()


async def health_check_handler():
    """Simple health check endpoint."""
    from aiohttp import web, web_response
    
    async def health(request):
        health_data = ingestion_service._get_health_data()
        status = 200 if health_data.get('status') == 'healthy' else 503
        
        return web_response.json_response(health_data, status=status)
    
    app = web.Application()
    app.router.add_get('/health', health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', settings.health_check_port)
    await site.start()
    
    logger.info("Health check server started", port=settings.health_check_port)


async def main():
    """Main function."""
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info("Received signal", signal=signum)
        asyncio.create_task(ingestion_service.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start health check server
        await health_check_handler()
        
        # Start the ingestion service
        await ingestion_service.start()
        
    except Exception as e:
        logger.error("Unhandled exception in main", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Unhandled exception", error=str(e))
        sys.exit(1)
