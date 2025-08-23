"""Tests for main application."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from src.main import CryptoHFTBot
from src.models import HealthStatus


class TestCryptoHFTBot:
    """Test CryptoHFTBot main application."""
    
    def test_initialization(self):
        """Test bot initialization."""
        bot = CryptoHFTBot()
        
        assert bot.order_book_manager is None
        assert bot.running is False
        assert isinstance(bot.health_status, HealthStatus)
        assert bot.health_status.status == "healthy"
    
    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful bot startup."""
        bot = CryptoHFTBot()
        
        with patch('src.main.start_http_server') as mock_prometheus, \
             patch('src.main.OrderBookManager') as mock_manager_class, \
             patch.object(bot, '_run_forever') as mock_run_forever:
            
            mock_manager = AsyncMock()
            mock_manager.start.return_value = True
            mock_manager_class.return_value = mock_manager
            
            result = await bot.start()
            
            mock_prometheus.assert_called_once()
            mock_manager.start.assert_called_once()
            assert bot.running is True
            assert bot.health_status.websocket_connected is True
            assert bot.health_status.kafka_connected is True
    
    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test bot startup failure."""
        bot = CryptoHFTBot()
        
        with patch('src.main.start_http_server'), \
             patch('src.main.OrderBookManager') as mock_manager_class:
            
            mock_manager = AsyncMock()
            mock_manager.start.return_value = False  # Simulate failure
            mock_manager_class.return_value = mock_manager
            
            result = await bot.start()
            
            assert result is False
            assert bot.running is False
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """Test bot shutdown."""
        bot = CryptoHFTBot()
        bot.running = True
        
        mock_manager = AsyncMock()
        bot.order_book_manager = mock_manager
        
        await bot.stop()
        
        assert bot.running is False
        mock_manager.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_health_status(self):
        """Test health status updates."""
        bot = CryptoHFTBot()
        
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {
            'websocket_stats': {
                'connected': True,
                'message_count': 100
            },
            'kafka_connected': True,
            'error_counts': {'BTCUSDT': 2, 'ETHUSDT': 1}
        }
        bot.order_book_manager = mock_manager
        
        await bot._update_health_status()
        
        assert bot.health_status.websocket_connected is True
        assert bot.health_status.kafka_connected is True
        assert bot.health_status.message_count == 100
        assert bot.health_status.error_count == 3  # Sum of error counts
    
    def test_get_health_status(self):
        """Test getting health status."""
        bot = CryptoHFTBot()
        
        health_status = bot.get_health_status()
        
        assert isinstance(health_status, HealthStatus)
        assert health_status.status == "healthy"
