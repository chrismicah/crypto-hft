"""Tests for WebSocket client."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json

from src.websocket_client import BinanceWebSocketClient, parse_order_book_update
from src.models import WebSocketMessage, PriceLevel


class TestBinanceWebSocketClient:
    """Test BinanceWebSocketClient."""
    
    def test_build_stream_url(self):
        """Test building stream URL."""
        callback = Mock()
        client = BinanceWebSocketClient(["BTCUSDT", "ETHUSDT"], callback)
        
        url = client._build_stream_url()
        assert "btcusdt@depth20@100ms" in url
        assert "ethusdt@depth20@100ms" in url
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        callback = Mock()
        client = BinanceWebSocketClient(["BTCUSDT"], callback)
        
        with patch('websockets.connect') as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            result = await client.connect()
            
            assert result is True
            assert client.connected is True
            assert client.websocket == mock_websocket
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        callback = Mock()
        client = BinanceWebSocketClient(["BTCUSDT"], callback)
        
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            result = await client.connect()
            
            assert result is False
            assert client.connected is False
    
    def test_get_stats(self):
        """Test getting connection statistics."""
        callback = Mock()
        client = BinanceWebSocketClient(["BTCUSDT"], callback)
        
        stats = client.get_stats()
        
        assert "connected" in stats
        assert "message_count" in stats
        assert "error_count" in stats
        assert "symbols" in stats
        assert stats["symbols"] == ["btcusdt"]


class TestParseOrderBookUpdate:
    """Test order book update parsing."""
    
    def test_parse_valid_update(self):
        """Test parsing a valid order book update."""
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12345,
                "u": 12346,
                "b": [["100.00", "1.0"], ["99.50", "2.0"]],
                "a": [["101.00", "1.0"], ["101.50", "2.0"]]
            }
        )
        
        update = parse_order_book_update(ws_message)
        
        assert update is not None
        assert update.symbol == "BTCUSDT"
        assert update.first_update_id == 12345
        assert update.final_update_id == 12346
        assert len(update.bids) == 2
        assert len(update.asks) == 2
        assert update.bids[0].price == 100.00
        assert update.bids[0].quantity == 1.0
    
    def test_parse_update_with_zero_quantity(self):
        """Test parsing update with zero quantities (removals)."""
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12345,
                "u": 12346,
                "b": [["100.00", "0.0"], ["99.50", "2.0"]],  # First bid should be filtered out
                "a": [["101.00", "1.0"], ["101.50", "0.0"]]   # Second ask should be filtered out
            }
        )
        
        update = parse_order_book_update(ws_message)
        
        assert update is not None
        assert len(update.bids) == 1  # Only non-zero quantity bid
        assert len(update.asks) == 1  # Only non-zero quantity ask
        assert update.bids[0].price == 99.50
        assert update.asks[0].price == 101.00
    
    def test_parse_invalid_stream(self):
        """Test parsing with invalid stream format."""
        ws_message = WebSocketMessage(
            stream="invalid_stream",
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12345,
                "u": 12346,
                "b": [["100.00", "1.0"]],
                "a": [["101.00", "1.0"]]
            }
        )
        
        update = parse_order_book_update(ws_message)
        
        assert update is None
    
    def test_parse_malformed_data(self):
        """Test parsing with malformed data."""
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={
                "invalid": "data"
            }
        )
        
        update = parse_order_book_update(ws_message)
        
        assert update is not None  # Should create update with empty bids/asks
        assert update.symbol == "BTCUSDT"
        assert len(update.bids) == 0
        assert len(update.asks) == 0
