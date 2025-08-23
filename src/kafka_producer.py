"""Kafka producer for publishing order book updates."""

import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import structlog

from .models import KafkaMessage, OrderBook, OrderBookUpdate
from .config import settings

logger = structlog.get_logger(__name__)


class OrderBookKafkaProducer:
    """Kafka producer for order book updates."""
    
    def __init__(self):
        self.producer: Optional[KafkaProducer] = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to Kafka cluster."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                client_id=settings.kafka_client_id,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                # Performance optimizations for HFT
                batch_size=16384,
                linger_ms=1,  # Low latency
                compression_type='snappy',
                acks=1,  # Balance between performance and reliability
                retries=3,
                retry_backoff_ms=100,
                # Buffer settings
                buffer_memory=33554432,  # 32MB
                max_block_ms=5000,
            )
            
            # Test connection
            metadata = self.producer.bootstrap_connected()
            if metadata:
                self.connected = True
                logger.info("Connected to Kafka", 
                           servers=settings.kafka_bootstrap_servers)
                return True
            else:
                logger.error("Failed to connect to Kafka")
                return False
                
        except Exception as e:
            logger.error("Error connecting to Kafka", error=str(e))
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Kafka."""
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
                self.connected = False
                logger.info("Disconnected from Kafka")
            except Exception as e:
                logger.error("Error disconnecting from Kafka", error=str(e))
    
    async def publish_order_book_snapshot(self, order_book: OrderBook) -> bool:
        """Publish a complete order book snapshot."""
        if not self.connected or not self.producer:
            logger.warning("Kafka producer not connected")
            return False
        
        try:
            message = KafkaMessage(
                topic=settings.kafka_topic_order_book,
                key=f"{order_book.symbol}_snapshot",
                value={
                    "type": "snapshot",
                    "symbol": order_book.symbol,
                    "bids": [[str(bid.price), str(bid.quantity)] for bid in order_book.bids],
                    "asks": [[str(ask.price), str(ask.quantity)] for ask in order_book.asks],
                    "last_update_id": order_book.last_update_id,
                    "timestamp": order_book.timestamp.isoformat(),
                    "checksum": order_book.calculate_checksum()
                }
            )
            
            future = self.producer.send(
                message.topic,
                key=message.key,
                value=message.value
            )
            
            # For HFT, we don't wait for confirmation to maintain low latency
            # But we can add a callback for monitoring
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            return True
            
        except Exception as e:
            logger.error("Error publishing order book snapshot", 
                        symbol=order_book.symbol, error=str(e))
            return False
    
    async def publish_order_book_update(self, update: OrderBookUpdate) -> bool:
        """Publish an incremental order book update."""
        if not self.connected or not self.producer:
            logger.warning("Kafka producer not connected")
            return False
        
        try:
            message = KafkaMessage(
                topic=settings.kafka_topic_order_book,
                key=f"{update.symbol}_update",
                value={
                    "type": "update",
                    "symbol": update.symbol,
                    "bids": [[str(bid.price), str(bid.quantity)] for bid in update.bids],
                    "asks": [[str(ask.price), str(ask.quantity)] for ask in update.asks],
                    "first_update_id": update.first_update_id,
                    "final_update_id": update.final_update_id,
                    "timestamp": update.timestamp.isoformat()
                }
            )
            
            future = self.producer.send(
                message.topic,
                key=message.key,
                value=message.value
            )
            
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            return True
            
        except Exception as e:
            logger.error("Error publishing order book update", 
                        symbol=update.symbol, error=str(e))
            return False
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message send."""
        logger.debug("Message sent successfully",
                    topic=record_metadata.topic,
                    partition=record_metadata.partition,
                    offset=record_metadata.offset)
    
    def _on_send_error(self, exception):
        """Callback for failed message send."""
        logger.error("Failed to send message", error=str(exception))
    
    def flush(self):
        """Flush any pending messages."""
        if self.producer:
            self.producer.flush()
    
    @property
    def is_connected(self) -> bool:
        """Check if producer is connected."""
        return self.connected and self.producer is not None
