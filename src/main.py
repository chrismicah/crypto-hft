"""Main entry point for the Crypto HFT Arbitrage Bot."""

import asyncio
import signal
import sys
from typing import List
import structlog
from prometheus_client import start_http_server, Counter, Histogram, Gauge

from .config import settings
from .order_book_manager import OrderBookManager
from .models import HealthStatus

# Configure structured logging
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
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
MESSAGE_COUNTER = Counter('websocket_messages_total', 'Total WebSocket messages received', ['symbol'])
ERROR_COUNTER = Counter('errors_total', 'Total errors', ['component', 'error_type'])
ORDER_BOOK_UPDATE_HISTOGRAM = Histogram('order_book_update_duration_seconds', 'Time spent processing order book updates')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections', ['type'])


class CryptoHFTBot:
    """Main application class for the Crypto HFT Arbitrage Bot."""
    
    def __init__(self):
        self.order_book_manager: OrderBookManager = None
        self.running = False
        self.health_status = HealthStatus()
        
    async def start(self):
        """Start the bot."""
        logger.info("Starting Crypto HFT Arbitrage Bot", 
                   symbols=settings.symbols_list,
                   testnet=settings.binance_testnet)
        
        try:
            # Start Prometheus metrics server
            start_http_server(settings.prometheus_port)
            logger.info("Prometheus metrics server started", port=settings.prometheus_port)
            
            # Initialize order book manager
            self.order_book_manager = OrderBookManager(settings.symbols_list)
            
            # Start order book manager
            success = await self.order_book_manager.start()
            if not success:
                logger.error("Failed to start order book manager")
                return False
            
            self.running = True
            self.health_status.websocket_connected = True
            self.health_status.kafka_connected = True
            
            logger.info("Crypto HFT Arbitrage Bot started successfully")
            
            # Keep the application running
            await self._run_forever()
            
        except Exception as e:
            logger.error("Error starting bot", error=str(e))
            return False
    
    async def stop(self):
        """Stop the bot."""
        logger.info("Stopping Crypto HFT Arbitrage Bot")
        
        self.running = False
        
        if self.order_book_manager:
            await self.order_book_manager.stop()
        
        logger.info("Crypto HFT Arbitrage Bot stopped")
    
    async def _run_forever(self):
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
    
    async def _update_health_status(self):
        """Update the health status."""
        try:
            if self.order_book_manager:
                stats = self.order_book_manager.get_stats()
                
                self.health_status.websocket_connected = stats.get('websocket_stats', {}).get('connected', False)
                self.health_status.kafka_connected = stats.get('kafka_connected', False)
                self.health_status.message_count = stats.get('websocket_stats', {}).get('message_count', 0)
                self.health_status.error_count = sum(stats.get('error_counts', {}).values())
                
                # Update Prometheus metrics
                ACTIVE_CONNECTIONS.labels(type='websocket').set(1 if self.health_status.websocket_connected else 0)
                ACTIVE_CONNECTIONS.labels(type='kafka').set(1 if self.health_status.kafka_connected else 0)
                
        except Exception as e:
            logger.error("Error updating health status", error=str(e))
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self.health_status


# Global bot instance
bot = CryptoHFTBot()


async def main():
    """Main function."""
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info("Received signal", signal=signum)
        asyncio.create_task(bot.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the bot
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Unhandled exception", error=str(e))
        sys.exit(1)
