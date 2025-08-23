"""Common metrics framework for Prometheus instrumentation across all services."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server
)
from prometheus_client.core import REGISTRY

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    help_text: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


class ServiceMetrics:
    """
    Base metrics class for HFT services with common patterns.
    Provides standardized metrics collection for all microservices.
    """
    
    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        registry: Optional[CollectorRegistry] = None,
        enable_system_metrics: bool = True
    ):
        """
        Initialize service metrics.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            registry: Prometheus registry (uses default if None)
            enable_system_metrics: Whether to collect system metrics
        """
        self.service_name = service_name
        self.service_version = service_version
        self.registry = registry or REGISTRY
        self.enable_system_metrics = enable_system_metrics
        
        # Common labels for all metrics
        self.common_labels = {
            'service': service_name,
            'version': service_version
        }
        
        # Initialize common metrics
        self._init_common_metrics()
        
        # System metrics collection
        if enable_system_metrics:
            self._init_system_metrics()
            self._start_system_metrics_collection()
        
        # Request tracking
        self._active_requests = {}
        self._request_lock = threading.Lock()
        
        logger.info("Service metrics initialized",
                   service=service_name,
                   version=service_version,
                   system_metrics=enable_system_metrics)
    
    def _init_common_metrics(self) -> None:
        """Initialize common metrics used by all services."""
        # Service info (use service-specific name to avoid conflicts)
        service_name_clean = self.service_name.replace('-', '_').replace('.', '_')
        self.service_info = Info(
            f'{service_name_clean}_service_info',
            'Service information',
            registry=self.registry
        )
        self.service_info.info(self.common_labels)
        
        # Request metrics (shared across services, use labels for differentiation)
        self.requests_total = Counter(
            'hft_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status', 'service'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'hft_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'service'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.active_requests = Gauge(
            'hft_active_requests',
            'Number of active requests',
            ['service'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'hft_errors_total',
            'Total number of errors',
            ['error_type', 'service'],
            registry=self.registry
        )
        
        # Health metrics
        self.service_up = Gauge(
            'hft_service_up',
            'Service health status (1 = up, 0 = down)',
            ['service'],
            registry=self.registry
        )
        self.service_up.labels(service=self.service_name).set(1)
        
        self.last_activity = Gauge(
            'hft_last_activity_timestamp',
            'Timestamp of last activity',
            ['service'],
            registry=self.registry
        )
        self.last_activity.labels(service=self.service_name).set(time.time())
    
    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        # CPU metrics
        self.cpu_usage = Gauge(
            'hft_cpu_usage_percent',
            'CPU usage percentage',
            ['service'],
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'hft_memory_usage_bytes',
            'Memory usage in bytes',
            ['service'],
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'hft_memory_usage_percent',
            'Memory usage percentage',
            ['service'],
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage = Gauge(
            'hft_disk_usage_percent',
            'Disk usage percentage',
            ['service'],
            registry=self.registry
        )
        
        # Network metrics
        self.network_bytes_sent = Counter(
            'hft_network_bytes_sent_total',
            'Total network bytes sent',
            ['service'],
            registry=self.registry
        )
        
        self.network_bytes_recv = Counter(
            'hft_network_bytes_received_total',
            'Total network bytes received',
            ['service'],
            registry=self.registry
        )
    
    def _start_system_metrics_collection(self) -> None:
        """Start background thread for system metrics collection."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.labels(service=self.service_name).set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    process = psutil.Process()
                    process_memory = process.memory_info()
                    
                    self.memory_usage.labels(service=self.service_name).set(process_memory.rss)
                    self.memory_usage_percent.labels(service=self.service_name).set(memory.percent)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.disk_usage.labels(service=self.service_name).set(disk.percent)
                    
                    # Network I/O
                    net_io = psutil.net_io_counters()
                    if hasattr(self, '_prev_net_sent'):
                        bytes_sent_delta = net_io.bytes_sent - self._prev_net_sent
                        bytes_recv_delta = net_io.bytes_recv - self._prev_net_recv
                        
                        self.network_bytes_sent.labels(service=self.service_name)._value._value += bytes_sent_delta
                        self.network_bytes_recv.labels(service=self.service_name)._value._value += bytes_recv_delta
                    
                    self._prev_net_sent = net_io.bytes_sent
                    self._prev_net_recv = net_io.bytes_recv
                    
                except Exception as e:
                    logger.error("Error collecting system metrics", error=str(e))
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager to track request metrics."""
        request_id = f"{method}_{endpoint}_{time.time()}"
        start_time = time.time()
        
        with self._request_lock:
            self._active_requests[request_id] = start_time
            self.active_requests.labels(service=self.service_name).inc()
        
        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            self.record_error(type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            self.requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status,
                service=self.service_name
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint,
                service=self.service_name
            ).observe(duration)
            
            with self._request_lock:
                self._active_requests.pop(request_id, None)
                self.active_requests.labels(service=self.service_name).dec()
            
            self.update_activity()
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.errors_total.labels(
            error_type=error_type,
            service=self.service_name
        ).inc()
        
        logger.debug("Error recorded", error_type=error_type, service=self.service_name)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity.labels(service=self.service_name).set(time.time())
    
    def set_health_status(self, healthy: bool) -> None:
        """Set service health status."""
        self.service_up.labels(service=self.service_name).set(1 if healthy else 0)
        
        if healthy:
            self.update_activity()
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)


class TradingMetrics(ServiceMetrics):
    """Specialized metrics for trading services."""
    
    def __init__(self, service_name: str, **kwargs):
        """Initialize trading metrics."""
        super().__init__(service_name, **kwargs)
        self._init_trading_metrics()
    
    def _init_trading_metrics(self) -> None:
        """Initialize trading-specific metrics."""
        # P&L metrics
        self.realized_pnl = Gauge(
            'realized_pnl_total',
            'Total realized P&L',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.unrealized_pnl = Gauge(
            'unrealized_pnl_total',
            'Total unrealized P&L',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value_total',
            'Total portfolio value',
            ['service'],
            registry=self.registry
        )
        
        # Position metrics
        self.position_size = Gauge(
            'position_size',
            'Current position size',
            ['service', 'symbol', 'side'],
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'active_positions_count',
            'Number of active positions',
            ['service'],
            registry=self.registry
        )
        
        # Trade metrics
        self.trades_total = Counter(
            'trades_total',
            'Total number of trades',
            ['service', 'symbol', 'side', 'result'],
            registry=self.registry
        )
        
        self.trade_volume = Counter(
            'trade_volume_total',
            'Total trade volume',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.trade_fees = Counter(
            'trade_fees_total',
            'Total trading fees',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        # Performance metrics
        self.win_rate = Gauge(
            'win_rate',
            'Win rate percentage',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'sharpe_ratio',
            'Sharpe ratio',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'max_drawdown_percent',
            'Maximum drawdown percentage',
            ['service'],
            registry=self.registry
        )
        
        self.current_drawdown = Gauge(
            'current_drawdown_percent',
            'Current drawdown percentage',
            ['service'],
            registry=self.registry
        )
    
    def update_pnl(self, symbol: str, realized_pnl: float, unrealized_pnl: float) -> None:
        """Update P&L metrics."""
        self.realized_pnl.labels(service=self.service_name, symbol=symbol).set(realized_pnl)
        self.unrealized_pnl.labels(service=self.service_name, symbol=symbol).set(unrealized_pnl)
        self.update_activity()
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value."""
        self.portfolio_value.labels(service=self.service_name).set(value)
        self.update_activity()
    
    def update_position(self, symbol: str, size: float, side: str) -> None:
        """Update position metrics."""
        self.position_size.labels(
            service=self.service_name,
            symbol=symbol,
            side=side
        ).set(abs(size))
        self.update_activity()
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        pnl: float
    ) -> None:
        """Record a trade."""
        result = "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
        
        self.trades_total.labels(
            service=self.service_name,
            symbol=symbol,
            side=side,
            result=result
        ).inc()
        
        self.trade_volume.labels(
            service=self.service_name,
            symbol=symbol
        ).inc(abs(quantity * price))
        
        self.trade_fees.labels(
            service=self.service_name,
            symbol=symbol
        ).inc(fee)
        
        self.update_activity()
    
    def update_performance_metrics(
        self,
        symbol: str,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        current_drawdown: float
    ) -> None:
        """Update performance metrics."""
        self.win_rate.labels(service=self.service_name, symbol=symbol).set(win_rate)
        self.sharpe_ratio.labels(service=self.service_name, symbol=symbol).set(sharpe_ratio)
        self.max_drawdown.labels(service=self.service_name).set(max_drawdown)
        self.current_drawdown.labels(service=self.service_name).set(current_drawdown)
        self.update_activity()


class MarketDataMetrics(ServiceMetrics):
    """Specialized metrics for market data services."""
    
    def __init__(self, service_name: str, **kwargs):
        """Initialize market data metrics."""
        super().__init__(service_name, **kwargs)
        self._init_market_data_metrics()
    
    def _init_market_data_metrics(self) -> None:
        """Initialize market data specific metrics."""
        # Data ingestion metrics
        self.messages_received = Counter(
            'messages_received_total',
            'Total messages received',
            ['service', 'source', 'message_type'],
            registry=self.registry
        )
        
        self.messages_processed = Counter(
            'messages_processed_total',
            'Total messages processed',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.message_processing_duration = Histogram(
            'message_processing_duration_seconds',
            'Message processing duration',
            ['service', 'message_type'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        # Market data quality metrics
        self.data_staleness = Gauge(
            'data_staleness_seconds',
            'Age of latest data in seconds',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.spread_value = Gauge(
            'spread_value',
            'Current spread value',
            ['service', 'symbol_pair'],
            registry=self.registry
        )
        
        self.volatility_forecast = Gauge(
            'volatility_forecast',
            'Volatility forecast value',
            ['service', 'symbol_pair', 'horizon'],
            registry=self.registry
        )
        
        # Connection metrics
        self.connection_status = Gauge(
            'connection_status',
            'Connection status (1 = connected, 0 = disconnected)',
            ['service', 'source'],
            registry=self.registry
        )
        
        self.reconnection_attempts = Counter(
            'reconnection_attempts_total',
            'Total reconnection attempts',
            ['service', 'source'],
            registry=self.registry
        )
    
    def record_message(self, source: str, message_type: str, symbol: str = None) -> None:
        """Record a received message."""
        self.messages_received.labels(
            service=self.service_name,
            source=source,
            message_type=message_type
        ).inc()
        
        if symbol:
            self.messages_processed.labels(
                service=self.service_name,
                symbol=symbol
            ).inc()
        
        self.update_activity()
    
    def update_spread(self, symbol_pair: str, spread_value: float) -> None:
        """Update spread value."""
        self.spread_value.labels(
            service=self.service_name,
            symbol_pair=symbol_pair
        ).set(spread_value)
        self.update_activity()
    
    def update_volatility_forecast(self, symbol_pair: str, horizon: str, forecast: float) -> None:
        """Update volatility forecast."""
        self.volatility_forecast.labels(
            service=self.service_name,
            symbol_pair=symbol_pair,
            horizon=horizon
        ).set(forecast)
        self.update_activity()
    
    def set_connection_status(self, source: str, connected: bool) -> None:
        """Set connection status."""
        self.connection_status.labels(
            service=self.service_name,
            source=source
        ).set(1 if connected else 0)
        
        if not connected:
            self.reconnection_attempts.labels(
                service=self.service_name,
                source=source
            ).inc()


class RiskMetrics(ServiceMetrics):
    """Specialized metrics for risk management services."""
    
    def __init__(self, service_name: str, **kwargs):
        """Initialize risk metrics."""
        super().__init__(service_name, **kwargs)
        self._init_risk_metrics()
    
    def _init_risk_metrics(self) -> None:
        """Initialize risk management specific metrics."""
        # Risk state metrics
        self.risk_state = Gauge(
            'risk_state',
            'Current risk state (0=NORMAL, 1=WARNING, 2=HALT, 3=RECOVERY)',
            ['service'],
            registry=self.registry
        )
        
        self.trading_halted = Gauge(
            'trading_halted',
            'Trading halt status (1 = halted, 0 = active)',
            ['service'],
            registry=self.registry
        )
        
        # Changepoint detection metrics
        self.changepoint_probability = Gauge(
            'changepoint_probability',
            'Current changepoint probability',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.changepoints_detected = Counter(
            'changepoints_detected_total',
            'Total changepoints detected',
            ['service', 'confidence_level'],
            registry=self.registry
        )
        
        # Risk violations
        self.risk_violations = Counter(
            'risk_violations_total',
            'Total risk violations',
            ['service', 'violation_type'],
            registry=self.registry
        )
        
        self.orders_rejected = Counter(
            'orders_rejected_total',
            'Total orders rejected by risk checks',
            ['service', 'rejection_reason'],
            registry=self.registry
        )
        
        # Position limits
        self.position_limit_utilization = Gauge(
            'position_limit_utilization_percent',
            'Position limit utilization percentage',
            ['service', 'symbol'],
            registry=self.registry
        )
        
        self.concentration_risk = Gauge(
            'concentration_risk_percent',
            'Concentration risk percentage',
            ['service', 'symbol'],
            registry=self.registry
        )
    
    def update_risk_state(self, state: int) -> None:
        """Update risk state (0=NORMAL, 1=WARNING, 2=HALT, 3=RECOVERY)."""
        self.risk_state.labels(service=self.service_name).set(state)
        self.update_activity()
    
    def set_trading_halt(self, halted: bool) -> None:
        """Set trading halt status."""
        self.trading_halted.labels(service=self.service_name).set(1 if halted else 0)
        self.update_activity()
    
    def update_changepoint_probability(self, symbol: str, probability: float) -> None:
        """Update changepoint probability."""
        self.changepoint_probability.labels(
            service=self.service_name,
            symbol=symbol
        ).set(probability)
        self.update_activity()
    
    def record_changepoint(self, confidence_level: str) -> None:
        """Record a detected changepoint."""
        self.changepoints_detected.labels(
            service=self.service_name,
            confidence_level=confidence_level
        ).inc()
        self.update_activity()
    
    def record_risk_violation(self, violation_type: str) -> None:
        """Record a risk violation."""
        self.risk_violations.labels(
            service=self.service_name,
            violation_type=violation_type
        ).inc()
        self.update_activity()
    
    def record_order_rejection(self, reason: str) -> None:
        """Record an order rejection."""
        self.orders_rejected.labels(
            service=self.service_name,
            rejection_reason=reason
        ).inc()
        self.update_activity()


def start_metrics_server(port: int = 8000, registry: Optional[CollectorRegistry] = None) -> None:
    """Start Prometheus metrics HTTP server."""
    try:
        start_http_server(port, registry=registry)
        logger.info("Metrics server started", port=port)
    except Exception as e:
        logger.error("Failed to start metrics server", port=port, error=str(e))
        raise
