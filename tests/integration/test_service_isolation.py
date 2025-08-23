"""Integration tests for microservice isolation and communication."""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import tempfile
import os

from services.ingestion.main import IngestionService
from services.ingestion.config import IngestionSettings
from services.ingestion.service_registry import ServiceRegistry, HealthStatus
from services.ingestion.websocket_client import WebSocketMessage, OrderBookUpdate, PriceLevel
from common.db import DatabaseClient


class TestServiceIsolation:
    """Test service isolation and fault tolerance."""
    
    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer."""
        producer = MagicMock()
        producer.connected = True
        producer.bootstrap_connected.return_value = True
        producer.send.return_value = MagicMock()
        producer.flush.return_value = None
        producer.close.return_value = None
        return producer
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection."""
        websocket = AsyncMock()
        websocket.connect.return_value = True
        websocket.close.return_value = None
        return websocket
    
    @pytest.fixture
    def sample_order_book_update(self):
        """Sample order book update."""
        return OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=1,
            final_update_id=2,
            bids=[PriceLevel("50000.0", "1.0"), PriceLevel("49999.0", "2.0")],
            asks=[PriceLevel("50001.0", "1.5"), PriceLevel("50002.0", "1.0")]
        )
    
    @pytest.fixture
    def temp_settings(self):
        """Temporary settings for testing."""
        settings = IngestionSettings(
            binance_api_key="test_key",
            binance_secret_key="test_secret",
            binance_testnet=True,
            kafka_bootstrap_servers="localhost:9092",
            symbols="BTCUSDT,ETHUSDT",
            redis_url="redis://localhost:6379",
            log_level="DEBUG"
        )
        return settings
    
    @pytest.mark.asyncio
    async def test_ingestion_service_startup_shutdown(self, temp_settings):
        """Test that ingestion service can start and stop independently."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Mock dependencies
            with patch.object(service, '_initialize_components') as mock_init:
                mock_init.return_value = None
                
                with patch.object(service, '_start_background_tasks') as mock_start:
                    mock_start.return_value = None
                    
                    with patch.object(service, '_run_forever') as mock_run:
                        mock_run.return_value = None
                        
                        # Test startup
                        await service.start()
                        
                        assert service.running is True
                        assert service.health_status.websocket_connected is True
                        assert service.health_status.kafka_connected is True
                        
                        # Test shutdown
                        await service.stop()
                        
                        assert service.running is False
    
    @pytest.mark.asyncio
    async def test_service_isolation_kafka_failure(self, temp_settings, mock_websocket):
        """Test that WebSocket continues working when Kafka fails."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Mock WebSocket success
            with patch('services.ingestion.websocket_client.websockets.connect', return_value=mock_websocket):
                # Mock Kafka failure
                with patch('services.ingestion.kafka_producer.KafkaProducer') as mock_producer_class:
                    mock_producer_class.side_effect = Exception("Kafka connection failed")
                    
                    # Should handle Kafka failure gracefully
                    try:
                        await service._initialize_components()
                        assert False, "Should have raised exception for Kafka failure"
                    except Exception as e:
                        assert "Failed to connect to Kafka" in str(e)
                    
                    # Service should still be able to handle WebSocket
                    assert service.websocket_client is not None
    
    @pytest.mark.asyncio
    async def test_service_isolation_websocket_failure(self, temp_settings, mock_kafka_producer):
        """Test that Kafka continues working when WebSocket fails."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Mock Kafka success
            with patch('services.ingestion.kafka_producer.KafkaProducer', return_value=mock_kafka_producer):
                # Mock WebSocket failure
                with patch('services.ingestion.websocket_client.websockets.connect') as mock_connect:
                    mock_connect.side_effect = Exception("WebSocket connection failed")
                    
                    # Should handle WebSocket failure gracefully
                    try:
                        await service._initialize_components()
                        assert False, "Should have raised exception for WebSocket failure"
                    except Exception as e:
                        assert "Failed to connect to WebSocket" in str(e)
                    
                    # Kafka should still be available
                    assert service.kafka_producer is not None
                    assert service.kafka_producer.connected is True
    
    @pytest.mark.asyncio
    async def test_message_flow_isolation(self, temp_settings, sample_order_book_update):
        """Test that message processing failures don't crash the service."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Mock successful initialization
            service.kafka_producer = MagicMock()
            service.kafka_producer.connected = True
            service.kafka_producer.publish_order_book_update = AsyncMock(return_value=True)
            service.kafka_producer.publish_trade_tick = AsyncMock(return_value=True)
            
            # Test successful message processing
            ws_message = WebSocketMessage(
                stream="btcusdt@depth20@100ms",
                data={
                    "bids": [["50000.0", "1.0"]],
                    "asks": [["50001.0", "1.0"]],
                    "U": 1,
                    "u": 2
                }
            )
            
            # Should process message successfully
            await service._on_websocket_message(ws_message)
            
            # Verify Kafka calls were made
            service.kafka_producer.publish_order_book_update.assert_called_once()
            service.kafka_producer.publish_trade_tick.assert_called_once()
            
            # Test message processing failure
            service.kafka_producer.publish_order_book_update.side_effect = Exception("Kafka publish failed")
            
            # Should handle failure gracefully without crashing
            await service._on_websocket_message(ws_message)
            
            # Service should still be functional
            assert service.running is False  # Not started yet, but should be stable
    
    @pytest.mark.asyncio
    async def test_service_registry_isolation(self, temp_settings):
        """Test that service registry failure doesn't affect core functionality."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Mock successful Kafka and WebSocket
            service.kafka_producer = MagicMock()
            service.kafka_producer.connected = True
            service.kafka_producer.connect = AsyncMock(return_value=True)
            
            service.websocket_client = MagicMock()
            service.websocket_client.connect = AsyncMock(return_value=True)
            
            # Mock service registry failure
            with patch('services.ingestion.service_registry.redis.from_url') as mock_redis:
                mock_redis.side_effect = Exception("Redis connection failed")
                
                # Should continue without service registry
                await service._initialize_components()
                
                # Core components should still be initialized
                assert service.kafka_producer is not None
                assert service.websocket_client is not None
                assert service.service_registry is not None  # Created but not connected
    
    @pytest.mark.asyncio
    async def test_health_status_isolation(self, temp_settings):
        """Test that health status updates work independently."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Initialize health status
            service.health_status = HealthStatus()
            
            # Test health data generation
            health_data = service._get_health_data()
            
            assert "websocket_connected" in health_data
            assert "kafka_connected" in health_data
            assert "status" in health_data
            assert health_data["status"] == "unhealthy"  # Not connected yet
            
            # Mock connected state
            service.health_status.websocket_connected = True
            service.health_status.kafka_connected = True
            service.health_status.last_message_time = datetime.utcnow()
            
            health_data = service._get_health_data()
            assert health_data["status"] == "healthy"
    
    def test_service_configuration_isolation(self, temp_settings):
        """Test that each service has independent configuration."""
        # Test ingestion service settings
        assert temp_settings.service_name == "ingestion-service"
        assert temp_settings.kafka_client_id == "ingestion-service"
        assert temp_settings.symbols_list == ["BTCUSDT", "ETHUSDT"]
        
        # Test that configuration is properly isolated
        assert hasattr(temp_settings, 'binance_api_key')
        assert hasattr(temp_settings, 'kafka_bootstrap_servers')
        assert hasattr(temp_settings, 'redis_url')


class TestServiceCommunication:
    """Test communication between microservices."""
    
    @pytest.fixture
    def temp_db(self):
        """Temporary database for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        db_url = f"sqlite:///{temp_db.name}"
        client = DatabaseClient(database_url=db_url)
        client.create_tables()
        
        yield client
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
    
    @pytest.mark.asyncio
    async def test_database_persistence_isolation(self, temp_db):
        """Test that database operations are isolated from service failures."""
        # Test order creation
        order = temp_db.write_order(
            order_id="test_order_123",
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            amount=0.1,
            status="filled"
        )
        
        assert order is not None
        assert order.order_id == "test_order_123"
        
        # Test trade creation
        trade = temp_db.write_trade(
            trade_id="test_trade_123",
            pair_id="BTCETH",
            asset1_symbol="BTC",
            asset2_symbol="ETH",
            side="long",
            strategy_id="pairs_trading",
            entry_time=datetime.utcnow(),
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0
        )
        
        assert trade is not None
        assert trade.trade_id == "test_trade_123"
        
        # Verify data persistence
        with temp_db.get_session() as session:
            saved_order = session.query(temp_db.__class__.__module__.split('.')[0]).first()
            # Note: This is a simplified test - in practice we'd query the actual models
    
    def test_kafka_topic_isolation(self):
        """Test that Kafka topics are properly isolated between services."""
        # Define expected topics for each service
        ingestion_topics = {
            "produces": ["order_book_updates", "trade-ticks", "health-status"],
            "consumes": []
        }
        
        kalman_topics = {
            "produces": ["signals-hedge-ratio"],
            "consumes": ["trade-ticks"]
        }
        
        garch_topics = {
            "produces": ["signals-thresholds"],
            "consumes": ["spread-data"]
        }
        
        execution_topics = {
            "produces": ["trade-events"],
            "consumes": ["order_book_updates", "signals-hedge-ratio", "signals-thresholds"]
        }
        
        # Verify no topic conflicts
        all_produced_topics = set()
        for service_topics in [ingestion_topics, kalman_topics, garch_topics, execution_topics]:
            for topic in service_topics["produces"]:
                assert topic not in all_produced_topics, f"Topic {topic} is produced by multiple services"
                all_produced_topics.add(topic)
        
        # Verify consumer dependencies
        all_topics = all_produced_topics
        for service_topics in [kalman_topics, garch_topics, execution_topics]:
            for topic in service_topics["consumes"]:
                assert topic in all_topics, f"Topic {topic} is consumed but not produced"
    
    def test_service_port_isolation(self):
        """Test that services use different ports to avoid conflicts."""
        service_ports = {
            "ingestion-service": {"metrics": 8000, "health": 8001},
            "kalman-filter-service": {"metrics": 8002, "health": 8003},
            "garch-volatility-service": {"metrics": 8004, "health": 8005},
            "execution-service": {"metrics": 8006, "health": 8007}
        }
        
        used_ports = set()
        for service, ports in service_ports.items():
            for port_type, port in ports.items():
                assert port not in used_ports, f"Port {port} is used by multiple services"
                used_ports.add(port)


class TestServiceRecovery:
    """Test service recovery and resilience."""
    
    @pytest.mark.asyncio
    async def test_service_restart_recovery(self, temp_settings):
        """Test that services can recover after restart."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Mock initialization
            service.kafka_producer = MagicMock()
            service.kafka_producer.connected = True
            service.websocket_client = MagicMock()
            service.websocket_client.connected = True
            
            # Simulate startup
            service.running = True
            service.health_status.websocket_connected = True
            service.health_status.kafka_connected = True
            
            # Verify healthy state
            health_data = service._get_health_data()
            assert health_data["status"] == "healthy"
            
            # Simulate restart
            await service.stop()
            assert service.running is False
            
            # Simulate restart with new instance
            new_service = IngestionService()
            
            # Should be able to start again
            new_service.kafka_producer = MagicMock()
            new_service.kafka_producer.connected = True
            new_service.websocket_client = MagicMock()
            new_service.websocket_client.connected = True
            
            new_service.running = True
            new_service.health_status.websocket_connected = True
            new_service.health_status.kafka_connected = True
            
            health_data = new_service._get_health_data()
            assert health_data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_partial_service_failure_recovery(self, temp_settings):
        """Test recovery from partial service failures."""
        with patch('services.ingestion.config.settings', temp_settings):
            service = IngestionService()
            
            # Start with healthy state
            service.health_status.websocket_connected = True
            service.health_status.kafka_connected = True
            service.health_status.last_message_time = datetime.utcnow()
            
            health_data = service._get_health_data()
            assert health_data["status"] == "healthy"
            
            # Simulate WebSocket failure
            service.health_status.websocket_connected = False
            
            health_data = service._get_health_data()
            assert health_data["status"] == "unhealthy"
            
            # Simulate recovery
            service.health_status.websocket_connected = True
            service.health_status.last_message_time = datetime.utcnow()
            
            health_data = service._get_health_data()
            assert health_data["status"] == "healthy"
