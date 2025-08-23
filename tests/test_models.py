"""Tests for data models."""

import pytest
from decimal import Decimal
from datetime import datetime

from src.models import PriceLevel, OrderBook, OrderBookUpdate, KafkaMessage, WebSocketMessage


class TestPriceLevel:
    """Test PriceLevel model."""
    
    def test_price_level_creation(self):
        """Test creating a price level."""
        level = PriceLevel(price="100.50", quantity="1.25")
        assert level.price == Decimal("100.50")
        assert level.quantity == Decimal("1.25")
    
    def test_price_level_decimal_conversion(self):
        """Test automatic conversion to Decimal."""
        level = PriceLevel(price=100.50, quantity=1.25)
        assert isinstance(level.price, Decimal)
        assert isinstance(level.quantity, Decimal)


class TestOrderBook:
    """Test OrderBook model."""
    
    def test_order_book_creation(self):
        """Test creating an order book."""
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        assert order_book.symbol == "BTCUSDT"
        assert len(order_book.bids) == 1
        assert len(order_book.asks) == 1
        assert order_book.last_update_id == 12345
    
    def test_best_bid_ask(self):
        """Test getting best bid and ask."""
        bids = [
            PriceLevel(price="100.00", quantity="1.0"),
            PriceLevel(price="99.50", quantity="2.0"),
        ]
        asks = [
            PriceLevel(price="101.00", quantity="1.0"),
            PriceLevel(price="101.50", quantity="2.0"),
        ]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        assert best_bid.price == Decimal("100.00")
        assert best_ask.price == Decimal("101.00")
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        spread = order_book.get_spread()
        assert spread == Decimal("1.00")
    
    def test_checksum_calculation(self):
        """Test checksum calculation."""
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        checksum = order_book.calculate_checksum()
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hash length


class TestOrderBookUpdate:
    """Test OrderBookUpdate model."""
    
    def test_order_book_update_creation(self):
        """Test creating an order book update."""
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12345,
            final_update_id=12346,
            bids=bids,
            asks=asks
        )
        
        assert update.symbol == "BTCUSDT"
        assert update.first_update_id == 12345
        assert update.final_update_id == 12346
        assert len(update.bids) == 1
        assert len(update.asks) == 1


class TestKafkaMessage:
    """Test KafkaMessage model."""
    
    def test_kafka_message_creation(self):
        """Test creating a Kafka message."""
        message = KafkaMessage(
            topic="test_topic",
            key="test_key",
            value={"test": "data"}
        )
        
        assert message.topic == "test_topic"
        assert message.key == "test_key"
        assert message.value == {"test": "data"}
    
    def test_kafka_message_to_json(self):
        """Test converting Kafka message to JSON."""
        message = KafkaMessage(
            topic="test_topic",
            key="test_key",
            value={"test": "data"}
        )
        
        json_str = message.to_json()
        assert isinstance(json_str, str)
        assert "test_topic" in json_str
        assert "test_key" in json_str


class TestWebSocketMessage:
    """Test WebSocketMessage model."""
    
    def test_websocket_message_creation(self):
        """Test creating a WebSocket message."""
        message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={"test": "data"}
        )
        
        assert message.stream == "btcusdt@depth20@100ms"
        assert message.data == {"test": "data"}
    
    def test_from_binance_depth(self):
        """Test creating from Binance depth message."""
        binance_message = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12345,
                "u": 12346,
                "b": [["100.00", "1.0"]],
                "a": [["101.00", "1.0"]]
            }
        }
        
        message = WebSocketMessage.from_binance_depth(binance_message)
        assert message.stream == "btcusdt@depth20@100ms"
        assert message.data["s"] == "BTCUSDT"
