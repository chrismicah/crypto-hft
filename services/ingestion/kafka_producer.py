"""Kafka producer for publishing market data from ingestion service."""

import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
from datetime import datetime
import asyncio

from .config import settings
from .websocket_client import OrderBookUpdate, WebSocketMessage
from common.logger import get_logger, get_performance_logger

logger = get_logger(__name__)
perf_logger = get_performance_logger("ingestion-service")


class KafkaMessage:
    """Kafka message wrapper."""
    
    def __init__(self, topic: str, key: str, value: Dict[str, Any]):
        self.topic = topic
        self.key = key
        self.value = value


class IngestionKafkaProducer:
    """Kafka producer for ingestion service."""
    
    def __init__(self):
        self.producer: Optional[KafkaProducer] = None
        self.connected = False
        self.message_count = 0
        self.error_count = 0
        
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
    
    async def publish_order_book_update(self, update: OrderBookUpdate) -> bool:
        """Publish an order book update."""
        if not self.connected or not self.producer:
            logger.warning("Kafka producer not connected")
            return False
        
        start_time = datetime.utcnow()
        
        try:
            message = KafkaMessage(
                topic=settings.kafka_topic_order_book,
                key=f"{update.symbol}_update",
                value={
                    "type": "order_book_update",
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
            
            self.message_count += 1
            
            # Log performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            perf_logger.log_processing_time("kafka_publish_order_book", processing_time * 1000)
            
            return True
            
        except Exception as e:
            logger.error("Error publishing order book update", 
                        symbol=update.symbol, error=str(e))
            self.error_count += 1
            return False
    
    async def publish_trade_tick(self, symbol: str, price: float, quantity: float, timestamp: datetime) -> bool:
        """Publish a trade tick for pairs trading calculations."""
        if not self.connected or not self.producer:
            logger.warning("Kafka producer not connected")
            return False
        
        start_time = datetime.utcnow()
        
        try:
            message = KafkaMessage(
                topic=settings.kafka_topic_trade_ticks,
                key=f"{symbol}_tick",
                value={
                    "type": "trade_tick",
                    "symbol": symbol,
                    "price": price,
                    "quantity": quantity,
                    "timestamp": timestamp.isoformat()
                }
            )
            
            future = self.producer.send(
                message.topic,
                key=message.key,
                value=message.value
            )
            
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            self.message_count += 1
            
            # Log performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            perf_logger.log_processing_time("kafka_publish_trade_tick", processing_time * 1000)
            
            return True
            
        except Exception as e:
            logger.error("Error publishing trade tick", 
                        symbol=symbol, error=str(e))
            self.error_count += 1
            return False
    
    async def publish_health_status(self, health_data: Dict[str, Any]) -> bool:
        """Publish service health status."""
        if not self.connected or not self.producer:
            return False
        
        try:
            message = KafkaMessage(
                topic="health-status",
                key=f"{settings.service_name}_health",
                value={
                    "service_name": settings.service_name,
                    "service_version": settings.service_version,
                    "health_data": health_data,
                    "timestamp": datetime.utcnow().isoformat()
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
            logger.error("Error publishing health status", error=str(e))
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
        self.error_count += 1
    
    def flush(self):
        """Flush any pending messages."""
        if self.producer:
            self.producer.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            "connected": self.connected,
            "message_count": self.message_count,
            "error_count": self.error_count
        }
    
    @property
    def is_connected(self) -> bool:
        """Check if producer is connected."""
        return self.connected and self.producer is not None
