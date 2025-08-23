"""Tests for Kafka producer."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.kafka_producer import OrderBookKafkaProducer
from src.models import OrderBook, OrderBookUpdate, PriceLevel


class TestOrderBookKafkaProducer:
    """Test OrderBookKafkaProducer."""
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful Kafka connection."""
        producer = OrderBookKafkaProducer()
        
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.bootstrap_connected.return_value = True
            mock_producer_class.return_value = mock_producer
            
            result = await producer.connect()
            
            assert result is True
            assert producer.connected is True
            assert producer.producer == mock_producer
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test Kafka connection failure."""
        producer = OrderBookKafkaProducer()
        
        with patch('kafka.KafkaProducer', side_effect=Exception("Connection failed")):
            result = await producer.connect()
            
            assert result is False
            assert producer.connected is False
    
    @pytest.mark.asyncio
    async def test_publish_order_book_snapshot(self):
        """Test publishing order book snapshot."""
        producer = OrderBookKafkaProducer()
        producer.connected = True
        
        mock_producer = Mock()
        mock_future = Mock()
        mock_producer.send.return_value = mock_future
        producer.producer = mock_producer
        
        # Create test order book
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        result = await producer.publish_order_book_snapshot(order_book)
        
        assert result is True
        mock_producer.send.assert_called_once()
        
        # Check the call arguments
        call_args = mock_producer.send.call_args
        assert call_args[1]['key'] == "BTCUSDT_snapshot"
        assert call_args[1]['value']['type'] == "snapshot"
        assert call_args[1]['value']['symbol'] == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_publish_order_book_update(self):
        """Test publishing order book update."""
        producer = OrderBookKafkaProducer()
        producer.connected = True
        
        mock_producer = Mock()
        mock_future = Mock()
        mock_producer.send.return_value = mock_future
        producer.producer = mock_producer
        
        # Create test update
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12345,
            final_update_id=12346,
            bids=bids,
            asks=asks
        )
        
        result = await producer.publish_order_book_update(update)
        
        assert result is True
        mock_producer.send.assert_called_once()
        
        # Check the call arguments
        call_args = mock_producer.send.call_args
        assert call_args[1]['key'] == "BTCUSDT_update"
        assert call_args[1]['value']['type'] == "update"
        assert call_args[1]['value']['symbol'] == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_publish_when_not_connected(self):
        """Test publishing when not connected."""
        producer = OrderBookKafkaProducer()
        producer.connected = False
        
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        result = await producer.publish_order_book_snapshot(order_book)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting from Kafka."""
        producer = OrderBookKafkaProducer()
        producer.connected = True
        
        mock_producer = Mock()
        producer.producer = mock_producer
        
        await producer.disconnect()
        
        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()
        assert producer.connected is False
    
    def test_is_connected_property(self):
        """Test is_connected property."""
        producer = OrderBookKafkaProducer()
        
        # Initially not connected
        assert producer.is_connected is False
        
        # Set connected
        producer.connected = True
        producer.producer = Mock()
        assert producer.is_connected is True
        
        # No producer
        producer.producer = None
        assert producer.is_connected is False
