"""Integration tests for the crypto HFT bot."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.order_book_manager import OrderBookManager
from src.websocket_client import BinanceWebSocketClient, parse_order_book_update
from src.kafka_producer import OrderBookKafkaProducer
from src.models import WebSocketMessage, OrderBook, PriceLevel


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_to_kafka_flow(self):
        """Test the complete flow from WebSocket to Kafka."""
        # Mock Kafka producer
        mock_kafka_producer = AsyncMock(spec=OrderBookKafkaProducer)
        mock_kafka_producer.connect.return_value = True
        mock_kafka_producer.is_connected = True
        mock_kafka_producer.publish_order_book_snapshot.return_value = True
        mock_kafka_producer.publish_order_book_update.return_value = True
        
        # Mock REST API response for initial snapshot
        mock_snapshot_data = {
            'lastUpdateId': 12345,
            'bids': [['100.00', '1.0'], ['99.50', '2.0']],
            'asks': [['101.00', '1.0'], ['101.50', '2.0']]
        }
        
        with patch('src.order_book_manager.OrderBookKafkaProducer', return_value=mock_kafka_producer), \
             patch('aiohttp.ClientSession.get') as mock_get:
            
            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_snapshot_data
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Create manager
            manager = OrderBookManager(['BTCUSDT'])
            
            # Initialize order book
            success = await manager._initialize_order_book('BTCUSDT')
            assert success is True
            
            # Verify initial snapshot was published
            mock_kafka_producer.publish_order_book_snapshot.assert_called_once()
            
            # Simulate WebSocket update
            ws_message = WebSocketMessage(
                stream="btcusdt@depth20@100ms",
                data={
                    "e": "depthUpdate",
                    "E": 1234567890,
                    "s": "BTCUSDT",
                    "U": 12346,  # first_update_id = last_update_id + 1
                    "u": 12346,  # final_update_id
                    "b": [["99.75", "1.5"]],  # New bid
                    "a": [["101.25", "1.5"]]   # New ask
                }
            )
            
            # Process the update
            await manager._on_websocket_message(ws_message)
            
            # Verify update was published
            mock_kafka_producer.publish_order_book_update.assert_called_once()
            
            # Verify order book was updated
            order_book = manager.get_order_book('BTCUSDT')
            assert order_book is not None
            assert order_book.last_update_id == 12346
    
    @pytest.mark.integration
    def test_order_book_update_parsing_and_application(self):
        """Test parsing WebSocket messages and applying updates."""
        # Create initial order book
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
        
        # Create WebSocket message
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12346,
                "u": 12346,
                "b": [["99.75", "1.5"], ["99.50", "0.0"]],  # Add new bid, remove existing
                "a": [["101.25", "1.5"]]  # Add new ask
            }
        )
        
        # Parse update
        update = parse_order_book_update(ws_message)
        assert update is not None
        assert update.symbol == "BTCUSDT"
        assert len(update.bids) == 2  # One new, one removal
        assert len(update.asks) == 1
        
        # Apply update
        manager = OrderBookManager(['BTCUSDT'])
        updated_book = manager._apply_update(order_book, update)
        
        # Verify results
        assert updated_book.last_update_id == 12346
        assert len(updated_book.bids) == 2  # 100.00 and 99.75 (99.50 removed)
        assert len(updated_book.asks) == 3  # Original 2 + new 101.25
        
        # Verify bid prices (should be sorted descending)
        bid_prices = [bid.price for bid in updated_book.bids]
        assert bid_prices == sorted(bid_prices, reverse=True)
        
        # Verify ask prices (should be sorted ascending)
        ask_prices = [ask.price for ask in updated_book.asks]
        assert ask_prices == sorted(ask_prices)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconnection_logic(self):
        """Test WebSocket reconnection logic."""
        callback = AsyncMock()
        client = BinanceWebSocketClient(['BTCUSDT'], callback)
        
        # Mock websockets.connect to fail first, then succeed
        connect_calls = 0
        async def mock_connect(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            if connect_calls == 1:
                raise Exception("Connection failed")
            return AsyncMock()
        
        with patch('websockets.connect', side_effect=mock_connect):
            # First connection attempt should fail
            result = await client.connect()
            assert result is False
            assert client.connected is False
            
            # Reconnection should succeed
            result = await client._reconnect()
            assert result is True
            assert client.connected is True
            assert connect_calls == 2
    
    @pytest.mark.integration
    def test_order_book_validation_comprehensive(self):
        """Test comprehensive order book validation."""
        manager = OrderBookManager(['BTCUSDT'])
        
        # Valid order book
        valid_bids = [
            PriceLevel(price="100.00", quantity="1.0"),
            PriceLevel(price="99.50", quantity="2.0")
        ]
        valid_asks = [
            PriceLevel(price="101.00", quantity="1.0"),
            PriceLevel(price="101.50", quantity="2.0")
        ]
        valid_book = OrderBook(
            symbol="BTCUSDT",
            bids=valid_bids,
            asks=valid_asks,
            last_update_id=12345
        )
        assert manager._validate_order_book(valid_book) is True
        
        # Invalid: unsorted bids
        invalid_bids = [
            PriceLevel(price="99.50", quantity="2.0"),
            PriceLevel(price="100.00", quantity="1.0")  # Should be first
        ]
        invalid_book1 = OrderBook(
            symbol="BTCUSDT",
            bids=invalid_bids,
            asks=valid_asks,
            last_update_id=12345
        )
        assert manager._validate_order_book(invalid_book1) is False
        
        # Invalid: unsorted asks
        invalid_asks = [
            PriceLevel(price="101.50", quantity="2.0"),
            PriceLevel(price="101.00", quantity="1.0")  # Should be first
        ]
        invalid_book2 = OrderBook(
            symbol="BTCUSDT",
            bids=valid_bids,
            asks=invalid_asks,
            last_update_id=12345
        )
        assert manager._validate_order_book(invalid_book2) is False
        
        # Invalid: crossed book
        crossed_bids = [PriceLevel(price="102.00", quantity="1.0")]  # Higher than best ask
        crossed_book = OrderBook(
            symbol="BTCUSDT",
            bids=crossed_bids,
            asks=valid_asks,
            last_update_id=12345
        )
        assert manager._validate_order_book(crossed_book) is False
    
    @pytest.mark.integration
    def test_checksum_calculation_consistency(self):
        """Test that checksum calculation is consistent."""
        bids = [
            PriceLevel(price="100.00", quantity="1.0"),
            PriceLevel(price="99.50", quantity="2.0")
        ]
        asks = [
            PriceLevel(price="101.00", quantity="1.0"),
            PriceLevel(price="101.50", quantity="2.0")
        ]
        
        # Create two identical order books
        book1 = OrderBook(
            symbol="BTCUSDT",
            bids=bids.copy(),
            asks=asks.copy(),
            last_update_id=12345
        )
        
        book2 = OrderBook(
            symbol="BTCUSDT",
            bids=bids.copy(),
            asks=asks.copy(),
            last_update_id=12345
        )
        
        # Checksums should be identical
        checksum1 = book1.calculate_checksum()
        checksum2 = book2.calculate_checksum()
        assert checksum1 == checksum2
        
        # Modify one book slightly
        book2.bids[0].quantity = "1.1"
        checksum3 = book2.calculate_checksum()
        assert checksum1 != checksum3
