"""Tests for order book manager."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from decimal import Decimal

from src.order_book_manager import OrderBookManager
from src.models import OrderBook, OrderBookUpdate, PriceLevel


class TestOrderBookManager:
    """Test OrderBookManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        manager = OrderBookManager(symbols)
        
        assert manager.symbols == symbols
        assert len(manager.order_books) == 0
        assert len(manager.last_update_ids) == 0
        assert len(manager.update_counts) == 2
        assert len(manager.error_counts) == 2
    
    def test_validate_update_sequence_valid(self):
        """Test valid update sequence validation."""
        manager = OrderBookManager(["BTCUSDT"])
        manager.last_update_ids["BTCUSDT"] = 12345
        
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12346,
            final_update_id=12347,
            bids=[],
            asks=[]
        )
        
        result = manager._validate_update_sequence("BTCUSDT", update)
        assert result is True
    
    def test_validate_update_sequence_invalid(self):
        """Test invalid update sequence validation."""
        manager = OrderBookManager(["BTCUSDT"])
        manager.last_update_ids["BTCUSDT"] = 12345
        
        # Gap in sequence
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12348,  # Should be 12346
            final_update_id=12349,
            bids=[],
            asks=[]
        )
        
        result = manager._validate_update_sequence("BTCUSDT", update)
        assert result is False
    
    def test_apply_update_add_levels(self):
        """Test applying update that adds price levels."""
        manager = OrderBookManager(["BTCUSDT"])
        
        # Initial order book
        initial_bids = [PriceLevel(price="100.00", quantity="1.0")]
        initial_asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=initial_bids,
            asks=initial_asks,
            last_update_id=12345
        )
        
        # Update with new levels
        update_bids = [PriceLevel(price="99.50", quantity="2.0")]
        update_asks = [PriceLevel(price="101.50", quantity="2.0")]
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12346,
            final_update_id=12346,
            bids=update_bids,
            asks=update_asks
        )
        
        updated_book = manager._apply_update(order_book, update)
        
        assert len(updated_book.bids) == 2
        assert len(updated_book.asks) == 2
        assert updated_book.last_update_id == 12346
    
    def test_apply_update_remove_levels(self):
        """Test applying update that removes price levels."""
        manager = OrderBookManager(["BTCUSDT"])
        
        # Initial order book with multiple levels
        initial_bids = [
            PriceLevel(price="100.00", quantity="1.0"),
            PriceLevel(price="99.50", quantity="2.0")
        ]
        initial_asks = [
            PriceLevel(price="101.00", quantity="1.0"),
            PriceLevel(price="101.50", quantity="2.0")
        ]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=initial_bids,
            asks=initial_asks,
            last_update_id=12345
        )
        
        # Update that removes levels (quantity = 0)
        update_bids = [PriceLevel(price="99.50", quantity="0.0")]
        update_asks = [PriceLevel(price="101.50", quantity="0.0")]
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12346,
            final_update_id=12346,
            bids=update_bids,
            asks=update_asks
        )
        
        updated_book = manager._apply_update(order_book, update)
        
        assert len(updated_book.bids) == 1
        assert len(updated_book.asks) == 1
        assert updated_book.bids[0].price == Decimal("100.00")
        assert updated_book.asks[0].price == Decimal("101.00")
    
    def test_validate_order_book_valid(self):
        """Test validating a valid order book."""
        manager = OrderBookManager(["BTCUSDT"])
        
        bids = [
            PriceLevel(price="100.00", quantity="1.0"),
            PriceLevel(price="99.50", quantity="2.0")
        ]
        asks = [
            PriceLevel(price="101.00", quantity="1.0"),
            PriceLevel(price="101.50", quantity="2.0")
        ]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        result = manager._validate_order_book(order_book)
        assert result is True
    
    def test_validate_order_book_crossed(self):
        """Test validating a crossed order book (invalid)."""
        manager = OrderBookManager(["BTCUSDT"])
        
        # Crossed book: best bid >= best ask
        bids = [PriceLevel(price="101.00", quantity="1.0")]
        asks = [PriceLevel(price="100.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        result = manager._validate_order_book(order_book)
        assert result is False
    
    def test_validate_order_book_negative_quantity(self):
        """Test validating order book with negative quantities."""
        manager = OrderBookManager(["BTCUSDT"])
        
        bids = [PriceLevel(price="100.00", quantity="-1.0")]  # Negative quantity
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        result = manager._validate_order_book(order_book)
        assert result is False
    
    def test_should_publish_snapshot(self):
        """Test snapshot publishing logic."""
        manager = OrderBookManager(["BTCUSDT"])
        
        # No previous snapshot - should publish
        result = manager._should_publish_snapshot("BTCUSDT")
        assert result is True
        
        # Recent snapshot - should not publish
        manager.last_snapshot_times["BTCUSDT"] = datetime.utcnow()
        result = manager._should_publish_snapshot("BTCUSDT")
        assert result is False
    
    def test_get_order_book(self):
        """Test getting order book for symbol."""
        manager = OrderBookManager(["BTCUSDT"])
        
        # No order book initially
        result = manager.get_order_book("BTCUSDT")
        assert result is None
        
        # Add order book
        bids = [PriceLevel(price="100.00", quantity="1.0")]
        asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        manager.order_books["BTCUSDT"] = order_book
        
        result = manager.get_order_book("BTCUSDT")
        assert result == order_book
    
    def test_get_stats(self):
        """Test getting manager statistics."""
        manager = OrderBookManager(["BTCUSDT", "ETHUSDT"])
        
        stats = manager.get_stats()
        
        assert "symbols" in stats
        assert "order_books_count" in stats
        assert "update_counts" in stats
        assert "error_counts" in stats
        assert "websocket_stats" in stats
        assert "kafka_connected" in stats
        
        assert stats["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert stats["order_books_count"] == 0  # No order books yet
