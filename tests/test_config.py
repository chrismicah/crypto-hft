"""Tests for configuration management."""

import pytest
import os
from unittest.mock import patch

from src.config import Settings


class TestSettings:
    """Test Settings configuration."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_SECRET_KEY': 'test_secret_key'
        }):
            settings = Settings()
            
            assert settings.binance_api_key == 'test_api_key'
            assert settings.binance_secret_key == 'test_secret_key'
            assert settings.binance_testnet is True
            assert settings.kafka_bootstrap_servers == 'localhost:9092'
            assert settings.kafka_topic_order_book == 'order_book_updates'
            assert settings.kafka_client_id == 'crypto_hft_bot'
            assert settings.symbols == ['BTCUSDT', 'ETHUSDT']
            assert settings.order_book_depth == 20
            assert settings.log_level == 'INFO'
            assert settings.log_format == 'json'
            assert settings.prometheus_port == 8000
            assert settings.health_check_port == 8001
    
    def test_environment_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_SECRET_KEY': 'test_secret_key',
            'BINANCE_TESTNET': 'false',
            'KAFKA_BOOTSTRAP_SERVERS': 'kafka1:9092,kafka2:9092',
            'SYMBOLS': 'BTCUSDT,ETHUSDT,ADAUSDT',
            'ORDER_BOOK_DEPTH': '50',
            'LOG_LEVEL': 'DEBUG',
            'PROMETHEUS_PORT': '9000'
        }):
            settings = Settings()
            
            assert settings.binance_testnet is False
            assert settings.kafka_bootstrap_servers == 'kafka1:9092,kafka2:9092'
            assert settings.symbols == ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            assert settings.order_book_depth == 50
            assert settings.log_level == 'DEBUG'
            assert settings.prometheus_port == 9000
    
    def test_binance_ws_base_url_testnet(self):
        """Test WebSocket URL for testnet."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_SECRET_KEY': 'test_secret_key',
            'BINANCE_TESTNET': 'true'
        }):
            settings = Settings()
            assert settings.binance_ws_base_url == 'wss://testnet.binance.vision/ws'
    
    def test_binance_ws_base_url_mainnet(self):
        """Test WebSocket URL for mainnet."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_SECRET_KEY': 'test_secret_key',
            'BINANCE_TESTNET': 'false'
        }):
            settings = Settings()
            assert settings.binance_ws_base_url == 'wss://stream.binance.com:9443/ws'
    
    def test_binance_api_base_url_testnet(self):
        """Test API URL for testnet."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_SECRET_KEY': 'test_secret_key',
            'BINANCE_TESTNET': 'true'
        }):
            settings = Settings()
            assert settings.binance_api_base_url == 'https://testnet.binance.vision/api'
    
    def test_binance_api_base_url_mainnet(self):
        """Test API URL for mainnet."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_api_key',
            'BINANCE_SECRET_KEY': 'test_secret_key',
            'BINANCE_TESTNET': 'false'
        }):
            settings = Settings()
            assert settings.binance_api_base_url == 'https://api.binance.com/api'
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # Pydantic validation error
                Settings()
