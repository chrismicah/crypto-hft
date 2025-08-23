"""Pytest configuration and fixtures."""

import pytest
import asyncio
import os
from unittest.mock import patch

# Set test environment variables
os.environ.update({
    'BINANCE_API_KEY': 'test_api_key',
    'BINANCE_SECRET_KEY': 'test_secret_key',
    'BINANCE_TESTNET': 'true',
    'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
    'LOG_LEVEL': 'DEBUG'
})


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """Fixture to provide clean environment variables for testing."""
    test_env = {
        'BINANCE_API_KEY': 'test_api_key',
        'BINANCE_SECRET_KEY': 'test_secret_key',
        'BINANCE_TESTNET': 'true',
        'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
        'KAFKA_TOPIC_ORDER_BOOK': 'test_order_book_updates',
        'KAFKA_CLIENT_ID': 'test_crypto_hft_bot',
        'SYMBOLS': 'BTCUSDT,ETHUSDT',
        'ORDER_BOOK_DEPTH': '20',
        'LOG_LEVEL': 'DEBUG',
        'LOG_FORMAT': 'json',
        'PROMETHEUS_PORT': '8000',
        'HEALTH_CHECK_PORT': '8001'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def sample_binance_depth_message():
    """Sample Binance depth update message."""
    return {
        "stream": "btcusdt@depth20@100ms",
        "data": {
            "e": "depthUpdate",
            "E": 1234567890,
            "s": "BTCUSDT",
            "U": 12345,
            "u": 12346,
            "b": [
                ["100.00", "1.0"],
                ["99.50", "2.0"],
                ["99.00", "0.0"]  # Removal
            ],
            "a": [
                ["101.00", "1.0"],
                ["101.50", "2.0"],
                ["102.00", "1.5"]
            ]
        }
    }


@pytest.fixture
def sample_order_book_snapshot():
    """Sample order book snapshot from REST API."""
    return {
        "lastUpdateId": 12345,
        "bids": [
            ["100.00", "1.0"],
            ["99.50", "2.0"],
            ["99.00", "1.5"],
            ["98.50", "3.0"]
        ],
        "asks": [
            ["101.00", "1.0"],
            ["101.50", "2.0"],
            ["102.00", "1.5"],
            ["102.50", "3.0"]
        ]
    }


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    from unittest.mock import AsyncMock
    
    producer = AsyncMock()
    producer.connect.return_value = True
    producer.disconnect.return_value = None
    producer.publish_order_book_snapshot.return_value = True
    producer.publish_order_book_update.return_value = True
    producer.is_connected = True
    producer.flush.return_value = None
    
    return producer


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for testing."""
    from unittest.mock import AsyncMock
    
    websocket = AsyncMock()
    websocket.close.return_value = None
    
    return websocket


# Configure asyncio for tests
@pytest.fixture(autouse=True)
def configure_asyncio():
    """Configure asyncio for testing."""
    # Set event loop policy for Windows compatibility
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
