"""Unit tests for metrics framework."""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from prometheus_client import CollectorRegistry, REGISTRY

from common.metrics import (
    ServiceMetrics, TradingMetrics, MarketDataMetrics, RiskMetrics,
    MetricConfig
)


class TestServiceMetrics:
    """Test cases for ServiceMetrics base class."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return CollectorRegistry()
    
    @pytest.fixture
    def service_metrics(self, registry):
        """Create ServiceMetrics instance."""
        return ServiceMetrics(
            service_name="test-service",
            service_version="1.0.0",
            registry=registry,
            enable_system_metrics=False  # Disable for testing
        )
    
    def test_initialization(self, service_metrics, registry):
        """Test service metrics initialization."""
        assert service_metrics.service_name == "test-service"
        assert service_metrics.service_version == "1.0.0"
        assert service_metrics.registry == registry
        assert not service_metrics.enable_system_metrics
        
        # Check common labels
        assert service_metrics.common_labels['service'] == "test-service"
        assert service_metrics.common_labels['version'] == "1.0.0"
    
    def test_common_metrics_creation(self, service_metrics):
        """Test that common metrics are created."""
        # Check that metrics exist
        assert hasattr(service_metrics, 'service_info')
        assert hasattr(service_metrics, 'requests_total')
        assert hasattr(service_metrics, 'request_duration')
        assert hasattr(service_metrics, 'active_requests')
        assert hasattr(service_metrics, 'errors_total')
        assert hasattr(service_metrics, 'service_up')
        assert hasattr(service_metrics, 'last_activity')
    
    def test_track_request_success(self, service_metrics):
        """Test successful request tracking."""
        initial_requests = service_metrics.requests_total._value._value
        initial_active = service_metrics.active_requests._value._value
        
        with service_metrics.track_request("GET", "/test"):
            # Should increment active requests
            current_active = service_metrics.active_requests._value._value
            assert current_active == initial_active + 1
            
            # Simulate some work
            time.sleep(0.01)
        
        # Should increment total requests and decrement active
        final_requests = service_metrics.requests_total._value._value
        final_active = service_metrics.active_requests._value._value
        
        assert final_requests > initial_requests
        assert final_active == initial_active
    
    def test_track_request_error(self, service_metrics):
        """Test request tracking with error."""
        initial_errors = service_metrics.errors_total._value._value
        
        with pytest.raises(ValueError):
            with service_metrics.track_request("POST", "/error"):
                raise ValueError("Test error")
        
        # Should increment error count
        final_errors = service_metrics.errors_total._value._value
        assert final_errors > initial_errors
    
    def test_record_error(self, service_metrics):
        """Test error recording."""
        initial_count = service_metrics.errors_total._value._value
        
        service_metrics.record_error("TestError")
        
        final_count = service_metrics.errors_total._value._value
        assert final_count == initial_count + 1
    
    def test_set_health_status(self, service_metrics):
        """Test health status setting."""
        # Initially should be up (1)
        assert service_metrics.service_up._value._value == 1
        
        # Set to down
        service_metrics.set_health_status(False)
        assert service_metrics.service_up._value._value == 0
        
        # Set back to up
        service_metrics.set_health_status(True)
        assert service_metrics.service_up._value._value == 1
    
    def test_update_activity(self, service_metrics):
        """Test activity timestamp update."""
        initial_time = service_metrics.last_activity._value._value
        
        time.sleep(0.01)
        service_metrics.update_activity()
        
        final_time = service_metrics.last_activity._value._value
        assert final_time > initial_time
    
    def test_get_metrics(self, service_metrics):
        """Test metrics export."""
        metrics_output = service_metrics.get_metrics()
        
        assert isinstance(metrics_output, bytes)
        assert b'service_up' in metrics_output
        assert b'test-service' in metrics_output
    
    def test_system_metrics_disabled(self, registry):
        """Test that system metrics are not created when disabled."""
        metrics = ServiceMetrics(
            service_name="test",
            registry=registry,
            enable_system_metrics=False
        )
        
        # System metrics should not exist
        assert not hasattr(metrics, 'cpu_usage')
        assert not hasattr(metrics, 'memory_usage')


class TestTradingMetrics:
    """Test cases for TradingMetrics."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return CollectorRegistry()
    
    @pytest.fixture
    def trading_metrics(self, registry):
        """Create TradingMetrics instance."""
        return TradingMetrics(
            service_name="trading-service",
            registry=registry,
            enable_system_metrics=False
        )
    
    def test_trading_metrics_initialization(self, trading_metrics):
        """Test trading metrics initialization."""
        # Check trading-specific metrics exist
        assert hasattr(trading_metrics, 'realized_pnl')
        assert hasattr(trading_metrics, 'unrealized_pnl')
        assert hasattr(trading_metrics, 'portfolio_value')
        assert hasattr(trading_metrics, 'position_size')
        assert hasattr(trading_metrics, 'trades_total')
        assert hasattr(trading_metrics, 'win_rate')
        assert hasattr(trading_metrics, 'sharpe_ratio')
    
    def test_update_pnl(self, trading_metrics):
        """Test P&L update."""
        trading_metrics.update_pnl("BTCUSDT", 1000.0, 500.0)
        
        # Check that metrics were updated
        # Check that metrics were updated by collecting samples
        realized_samples = list(trading_metrics.realized_pnl.collect())[0].samples
        unrealized_samples = list(trading_metrics.unrealized_pnl.collect())[0].samples
        
        realized_pnl = next((s.value for s in realized_samples 
                            if s.labels.get('symbol') == 'BTCUSDT'), 0)
        unrealized_pnl = next((s.value for s in unrealized_samples 
                              if s.labels.get('symbol') == 'BTCUSDT'), 0)
        
        assert realized_pnl == 1000.0
        assert unrealized_pnl == 500.0
    
    def test_update_portfolio_value(self, trading_metrics):
        """Test portfolio value update."""
        trading_metrics.update_portfolio_value(100000.0)
        
        portfolio_value = trading_metrics.portfolio_value.labels(
            service="trading-service"
        )._value._value
        
        assert portfolio_value == 100000.0
    
    def test_update_position(self, trading_metrics):
        """Test position update."""
        trading_metrics.update_position("BTCUSDT", 1.5, "LONG")
        
        position_size = trading_metrics.position_size.labels(
            service="trading-service",
            symbol="BTCUSDT",
            side="LONG"
        )._value._value
        
        assert position_size == 1.5
    
    def test_record_trade(self, trading_metrics):
        """Test trade recording."""
        # Get initial values by checking the metric samples
        initial_samples = list(trading_metrics.trades_total.collect())[0].samples
        initial_count = len(initial_samples)
        
        trading_metrics.record_trade(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            pnl=1000.0
        )
        
        # Check that metrics were recorded
        trade_samples = list(trading_metrics.trades_total.collect())[0].samples
        volume_samples = list(trading_metrics.trade_volume.collect())[0].samples
        fee_samples = list(trading_metrics.trade_fees.collect())[0].samples
        
        # Should have more samples after recording trade
        assert len(trade_samples) > initial_count
        
        # Check that volume and fees were recorded
        assert len(volume_samples) > 0
        assert len(fee_samples) > 0
        
        # Check specific values
        btc_volume = next((s.value for s in volume_samples 
                          if s.labels.get('symbol') == 'BTCUSDT'), 0)
        btc_fees = next((s.value for s in fee_samples 
                        if s.labels.get('symbol') == 'BTCUSDT'), 0)
        
        assert btc_volume == 50000.0  # quantity * price
        assert btc_fees == 25.0
    
    def test_update_performance_metrics(self, trading_metrics):
        """Test performance metrics update."""
        trading_metrics.update_performance_metrics(
            symbol="BTCUSDT",
            win_rate=65.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0,
            current_drawdown=2.0
        )
        
        win_rate = trading_metrics.win_rate.labels(
            service="trading-service", symbol="BTCUSDT"
        )._value._value
        
        sharpe_ratio = trading_metrics.sharpe_ratio.labels(
            service="trading-service", symbol="BTCUSDT"
        )._value._value
        
        assert win_rate == 65.0
        assert sharpe_ratio == 1.5


class TestMarketDataMetrics:
    """Test cases for MarketDataMetrics."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return CollectorRegistry()
    
    @pytest.fixture
    def market_metrics(self, registry):
        """Create MarketDataMetrics instance."""
        return MarketDataMetrics(
            service_name="market-service",
            registry=registry,
            enable_system_metrics=False
        )
    
    def test_market_metrics_initialization(self, market_metrics):
        """Test market data metrics initialization."""
        # Check market-specific metrics exist
        assert hasattr(market_metrics, 'messages_received')
        assert hasattr(market_metrics, 'messages_processed')
        assert hasattr(market_metrics, 'data_staleness')
        assert hasattr(market_metrics, 'spread_value')
        assert hasattr(market_metrics, 'connection_status')
    
    def test_record_message(self, market_metrics):
        """Test message recording."""
        initial_received = market_metrics.messages_received._value._value
        initial_processed = market_metrics.messages_processed._value._value
        
        market_metrics.record_message("binance", "orderbook", "BTCUSDT")
        
        final_received = market_metrics.messages_received._value._value
        final_processed = market_metrics.messages_processed._value._value
        
        assert final_received > initial_received
        assert final_processed > initial_processed
    
    def test_update_spread(self, market_metrics):
        """Test spread value update."""
        market_metrics.update_spread("BTCUSDT-ETHUSDT", 0.05)
        
        spread_value = market_metrics.spread_value.labels(
            service="market-service",
            symbol_pair="BTCUSDT-ETHUSDT"
        )._value._value
        
        assert spread_value == 0.05
    
    def test_update_volatility_forecast(self, market_metrics):
        """Test volatility forecast update."""
        market_metrics.update_volatility_forecast("BTCUSDT-ETHUSDT", "1h", 0.15)
        
        volatility = market_metrics.volatility_forecast.labels(
            service="market-service",
            symbol_pair="BTCUSDT-ETHUSDT",
            horizon="1h"
        )._value._value
        
        assert volatility == 0.15
    
    def test_set_connection_status(self, market_metrics):
        """Test connection status setting."""
        # Set connected
        market_metrics.set_connection_status("binance", True)
        
        status = market_metrics.connection_status.labels(
            service="market-service",
            source="binance"
        )._value._value
        
        assert status == 1
        
        # Set disconnected
        initial_reconnects = market_metrics.reconnection_attempts._value._value
        market_metrics.set_connection_status("binance", False)
        
        status = market_metrics.connection_status.labels(
            service="market-service",
            source="binance"
        )._value._value
        
        final_reconnects = market_metrics.reconnection_attempts._value._value
        
        assert status == 0
        assert final_reconnects > initial_reconnects


class TestRiskMetrics:
    """Test cases for RiskMetrics."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return CollectorRegistry()
    
    @pytest.fixture
    def risk_metrics(self, registry):
        """Create RiskMetrics instance."""
        return RiskMetrics(
            service_name="risk-service",
            registry=registry,
            enable_system_metrics=False
        )
    
    def test_risk_metrics_initialization(self, risk_metrics):
        """Test risk metrics initialization."""
        # Check risk-specific metrics exist
        assert hasattr(risk_metrics, 'risk_state')
        assert hasattr(risk_metrics, 'trading_halted')
        assert hasattr(risk_metrics, 'changepoint_probability')
        assert hasattr(risk_metrics, 'risk_violations')
        assert hasattr(risk_metrics, 'orders_rejected')
    
    def test_update_risk_state(self, risk_metrics):
        """Test risk state update."""
        risk_metrics.update_risk_state(2)  # HALT state
        
        risk_state = risk_metrics.risk_state.labels(
            service="risk-service"
        )._value._value
        
        assert risk_state == 2
    
    def test_set_trading_halt(self, risk_metrics):
        """Test trading halt status."""
        risk_metrics.set_trading_halt(True)
        
        halt_status = risk_metrics.trading_halted.labels(
            service="risk-service"
        )._value._value
        
        assert halt_status == 1
        
        risk_metrics.set_trading_halt(False)
        
        halt_status = risk_metrics.trading_halted.labels(
            service="risk-service"
        )._value._value
        
        assert halt_status == 0
    
    def test_update_changepoint_probability(self, risk_metrics):
        """Test changepoint probability update."""
        risk_metrics.update_changepoint_probability("BTCUSDT", 0.85)
        
        cp_prob = risk_metrics.changepoint_probability.labels(
            service="risk-service",
            symbol="BTCUSDT"
        )._value._value
        
        assert cp_prob == 0.85
    
    def test_record_changepoint(self, risk_metrics):
        """Test changepoint recording."""
        initial_count = risk_metrics.changepoints_detected._value._value
        
        risk_metrics.record_changepoint("HIGH")
        
        final_count = risk_metrics.changepoints_detected._value._value
        assert final_count > initial_count
    
    def test_record_risk_violation(self, risk_metrics):
        """Test risk violation recording."""
        initial_count = risk_metrics.risk_violations._value._value
        
        risk_metrics.record_risk_violation("POSITION_SIZE_EXCEEDED")
        
        final_count = risk_metrics.risk_violations._value._value
        assert final_count > initial_count
    
    def test_record_order_rejection(self, risk_metrics):
        """Test order rejection recording."""
        initial_count = risk_metrics.orders_rejected._value._value
        
        risk_metrics.record_order_rejection("DAILY_LOSS_LIMIT")
        
        final_count = risk_metrics.orders_rejected._value._value
        assert final_count > initial_count


class TestMetricConfig:
    """Test cases for MetricConfig."""
    
    def test_metric_config_creation(self):
        """Test metric configuration creation."""
        config = MetricConfig(
            name="test_metric",
            help_text="Test metric for testing",
            labels=["service", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 5.0]
        )
        
        assert config.name == "test_metric"
        assert config.help_text == "Test metric for testing"
        assert config.labels == ["service", "endpoint"]
        assert config.buckets == [0.1, 0.5, 1.0, 5.0]
    
    def test_metric_config_defaults(self):
        """Test metric configuration with defaults."""
        config = MetricConfig(
            name="simple_metric",
            help_text="Simple test metric"
        )
        
        assert config.labels == []
        assert config.buckets is None


class TestMetricsIntegration:
    """Integration tests for metrics framework."""
    
    def test_multiple_services_same_registry(self):
        """Test multiple services using the same registry."""
        registry = CollectorRegistry()
        
        service1 = ServiceMetrics("service1", registry=registry, enable_system_metrics=False)
        service2 = ServiceMetrics("service2", registry=registry, enable_system_metrics=False)
        
        # Both services should be able to export metrics
        metrics1 = service1.get_metrics()
        metrics2 = service2.get_metrics()
        
        assert b'service1' in metrics1
        assert b'service2' in metrics2
        
        # Combined metrics should contain both services
        combined_metrics = registry.generate_latest()
        assert b'service1' in combined_metrics
        assert b'service2' in combined_metrics
    
    def test_concurrent_metric_updates(self):
        """Test concurrent metric updates."""
        registry = CollectorRegistry()
        metrics = ServiceMetrics("concurrent-test", registry=registry, enable_system_metrics=False)
        
        def update_metrics():
            for i in range(100):
                metrics.record_error("TestError")
                metrics.update_activity()
                time.sleep(0.001)
        
        # Run concurrent updates
        threads = [threading.Thread(target=update_metrics) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have recorded 500 errors total (100 * 5 threads)
        final_errors = metrics.errors_total._value._value
        assert final_errors == 500
    
    @patch('common.metrics.start_http_server')
    def test_metrics_server_start(self, mock_start_server):
        """Test metrics server startup."""
        from common.metrics import start_metrics_server
        
        start_metrics_server(port=8000)
        mock_start_server.assert_called_once_with(8000, registry=None)
