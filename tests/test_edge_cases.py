"""Tests for edge cases and error conditions."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime

from src.websocket_client import BinanceWebSocketClient, parse_order_book_update
from src.kafka_producer import OrderBookKafkaProducer
from src.order_book_manager import OrderBookManager
from src.models import WebSocketMessage, OrderBook, OrderBookUpdate, PriceLevel


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_parse_empty_websocket_message(self):
        """Test parsing empty WebSocket message."""
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={}
        )
        
        update = parse_order_book_update(ws_message)
        assert update is not None
        assert update.symbol == "BTCUSDT"
        assert len(update.bids) == 0
        assert len(update.asks) == 0
        assert update.first_update_id == 0
        assert update.final_update_id == 0
    
    @pytest.mark.unit
    def test_parse_malformed_price_data(self):
        """Test parsing WebSocket message with malformed price data."""
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12345,
                "u": 12346,
                "b": [["invalid_price", "1.0"], ["100.00"]],  # Malformed data
                "a": [["101.00", "invalid_quantity"], []]      # Malformed data
            }
        )
        
        update = parse_order_book_update(ws_message)
        assert update is not None
        assert len(update.bids) == 0  # Invalid entries filtered out
        assert len(update.asks) == 0  # Invalid entries filtered out
    
    @pytest.mark.unit
    def test_order_book_with_zero_quantities(self):
        """Test order book handling with zero quantities."""
        manager = OrderBookManager(['BTCUSDT'])
        
        # Initial order book
        initial_bids = [PriceLevel(price="100.00", quantity="1.0")]
        initial_asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=initial_bids,
            asks=initial_asks,
            last_update_id=12345
        )
        
        # Update with zero quantities (should remove levels)
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12346,
            final_update_id=12346,
            bids=[PriceLevel(price="100.00", quantity="0.0")],
            asks=[PriceLevel(price="101.00", quantity="0.0")]
        )
        
        updated_book = manager._apply_update(order_book, update)
        
        assert len(updated_book.bids) == 0
        assert len(updated_book.asks) == 0
    
    @pytest.mark.unit
    def test_order_book_with_extreme_values(self):
        """Test order book with extreme price/quantity values."""
        # Very large numbers
        large_bids = [PriceLevel(price="999999999.99999999", quantity="999999999.99999999")]
        large_asks = [PriceLevel(price="1000000000.00000000", quantity="999999999.99999999")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=large_bids,
            asks=large_asks,
            last_update_id=12345
        )
        
        # Should handle large numbers correctly
        assert order_book.get_best_bid().price == Decimal("999999999.99999999")
        assert order_book.get_best_ask().price == Decimal("1000000000.00000000")
        assert order_book.get_spread() == Decimal("0.00000001")
    
    @pytest.mark.unit
    def test_order_book_with_very_small_values(self):
        """Test order book with very small price/quantity values."""
        # Very small numbers
        small_bids = [PriceLevel(price="0.00000001", quantity="0.00000001")]
        small_asks = [PriceLevel(price="0.00000002", quantity="0.00000001")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=small_bids,
            asks=small_asks,
            last_update_id=12345
        )
        
        # Should handle small numbers correctly
        assert order_book.get_best_bid().price == Decimal("0.00000001")
        assert order_book.get_best_ask().price == Decimal("0.00000002")
        assert order_book.get_spread() == Decimal("0.00000001")
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_kafka_producer_connection_failure_recovery(self):
        """Test Kafka producer recovery from connection failures."""
        producer = OrderBookKafkaProducer()
        
        # Simulate connection failure
        with patch('kafka.KafkaProducer', side_effect=Exception("Connection failed")):
            result = await producer.connect()
            assert result is False
            assert producer.connected is False
        
        # Simulate successful reconnection
        with patch('kafka.KafkaProducer') as mock_producer_class:
            mock_producer = Mock()
            mock_producer.bootstrap_connected.return_value = True
            mock_producer_class.return_value = mock_producer
            
            result = await producer.connect()
            assert result is True
            assert producer.connected is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_websocket_message_callback_exception(self):
        """Test WebSocket client handling callback exceptions."""
        # Create callback that raises exception
        def failing_callback(message):
            raise Exception("Callback failed")
        
        client = BinanceWebSocketClient(['BTCUSDT'], failing_callback)
        
        # Should not crash when callback fails
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={"test": "data"}
        )
        
        # This should not raise an exception
        await client._process_message(ws_message)
        
        # Error count should be incremented
        assert client.error_count > 0
    
    @pytest.mark.unit
    def test_order_book_manager_invalid_symbol(self):
        """Test order book manager with invalid symbol."""
        manager = OrderBookManager(['BTCUSDT'])
        
        # Try to get order book for non-existent symbol
        result = manager.get_order_book('INVALID')
        assert result is None
        
        # Try to process update for non-existent symbol
        update = OrderBookUpdate(
            symbol="INVALID",
            first_update_id=1,
            final_update_id=1,
            bids=[],
            asks=[]
        )
        
        # Should handle gracefully (log warning but not crash)
        asyncio.run(manager._process_order_book_update(update))
    
    @pytest.mark.unit
    def test_order_book_checksum_with_empty_book(self):
        """Test checksum calculation with empty order book."""
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            last_update_id=12345
        )
        
        checksum = order_book.calculate_checksum()
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hash length
    
    @pytest.mark.unit
    def test_order_book_spread_with_empty_sides(self):
        """Test spread calculation with empty bid or ask sides."""
        # Empty bids
        order_book1 = OrderBook(
            symbol="BTCUSDT",
            bids=[],
            asks=[PriceLevel(price="101.00", quantity="1.0")],
            last_update_id=12345
        )
        assert order_book1.get_spread() is None
        
        # Empty asks
        order_book2 = OrderBook(
            symbol="BTCUSDT",
            bids=[PriceLevel(price="100.00", quantity="1.0")],
            asks=[],
            last_update_id=12345
        )
        assert order_book2.get_spread() is None
        
        # Both empty
        order_book3 = OrderBook(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            last_update_id=12345
        )
        assert order_book3.get_spread() is None
    
    @pytest.mark.unit
    def test_websocket_client_reconnection_max_attempts(self):
        """Test WebSocket client max reconnection attempts."""
        callback = Mock()
        client = BinanceWebSocketClient(['BTCUSDT'], callback)
        client.max_reconnect_attempts = 3
        client.reconnect_attempts = 3  # Already at max
        
        # Should not attempt reconnection
        result = asyncio.run(client._reconnect())
        assert result is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_order_book_manager_sequence_gap_recovery(self):
        """Test order book manager recovery from sequence gaps."""
        manager = OrderBookManager(['BTCUSDT'])
        manager.last_update_ids['BTCUSDT'] = 12345
        
        # Mock the initialization method to simulate recovery
        with patch.object(manager, '_initialize_order_book') as mock_init:
            mock_init.return_value = True
            
            # Create update with gap in sequence
            update = OrderBookUpdate(
                symbol="BTCUSDT",
                first_update_id=12350,  # Gap: should be 12346
                final_update_id=12350,
                bids=[],
                asks=[]
            )
            
            await manager._process_order_book_update(update)
            
            # Should trigger re-initialization
            mock_init.assert_called_once_with('BTCUSDT')
    
    @pytest.mark.unit
    def test_price_level_decimal_precision(self):
        """Test PriceLevel decimal precision handling."""
        # Test with string input (preferred for precision)
        level1 = PriceLevel(price="100.123456789", quantity="1.987654321")
        assert level1.price == Decimal("100.123456789")
        assert level1.quantity == Decimal("1.987654321")
        
        # Test with float input (should convert to Decimal)
        level2 = PriceLevel(price=100.123456789, quantity=1.987654321)
        assert isinstance(level2.price, Decimal)
        assert isinstance(level2.quantity, Decimal)
        
        # Note: Float precision may not be exact due to floating point representation
        # This is why string input is preferred for financial data
