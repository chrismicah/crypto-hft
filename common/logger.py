"""Unified structured logging configuration for all services."""

import structlog
import logging
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import json


def configure_logging(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    include_stdlib: bool = True
) -> None:
    """
    Configure structured logging for a service.
    
    Args:
        service_name: Name of the service for log identification
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' or 'console')
        log_file: Optional file path for log output
        include_stdlib: Whether to configure stdlib logging as well
    """
    # Set log level
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure processors based on format
    if log_format.lower() == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_service_name(service_name),
            add_timestamp,
            add_process_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console format for development
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_service_name(service_name),
            add_timestamp,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging if requested
    if include_stdlib:
        # Create handler
        if log_file:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter
        if log_format.lower() == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_obj)
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        
        # Suppress noisy third-party loggers
        logging.getLogger("kafka").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("ccxt").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)


def add_service_name(service_name: str):
    """Processor to add service name to all log entries."""
    def processor(logger, method_name, event_dict):
        event_dict["service"] = service_name
        return event_dict
    return processor


def add_timestamp(logger, method_name, event_dict):
    """Processor to add ISO timestamp to all log entries."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_process_info(logger, method_name, event_dict):
    """Processor to add process information to log entries."""
    event_dict["process_id"] = os.getpid()
    return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for stdlib logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process_id": os.getpid(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text',
                          'stack_info', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class TradeLogger:
    """Specialized logger for trading events."""
    
    def __init__(self, service_name: str):
        """Initialize trade logger."""
        self.logger = structlog.get_logger(f"{service_name}.trades")
    
    def log_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log order placement."""
        self.logger.info(
            "Order placed",
            event_type="order_placed",
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            **kwargs
        )
    
    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        fee: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log order fill."""
        self.logger.info(
            "Order filled",
            event_type="order_filled",
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            fee=fee,
            **kwargs
        )
    
    def log_order_cancelled(
        self,
        order_id: str,
        symbol: str,
        reason: str = "",
        **kwargs
    ) -> None:
        """Log order cancellation."""
        self.logger.info(
            "Order cancelled",
            event_type="order_cancelled",
            order_id=order_id,
            symbol=symbol,
            reason=reason,
            **kwargs
        )
    
    def log_trade_opened(
        self,
        trade_id: str,
        pair_id: str,
        side: str,
        entry_price: float,
        quantity: float,
        hedge_ratio: float,
        **kwargs
    ) -> None:
        """Log trade opening."""
        self.logger.info(
            "Trade opened",
            event_type="trade_opened",
            trade_id=trade_id,
            pair_id=pair_id,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            hedge_ratio=hedge_ratio,
            **kwargs
        )
    
    def log_trade_closed(
        self,
        trade_id: str,
        pair_id: str,
        exit_price: float,
        pnl: float,
        duration_seconds: float,
        **kwargs
    ) -> None:
        """Log trade closing."""
        self.logger.info(
            "Trade closed",
            event_type="trade_closed",
            trade_id=trade_id,
            pair_id=pair_id,
            exit_price=exit_price,
            pnl=pnl,
            duration_seconds=duration_seconds,
            **kwargs
        )
    
    def log_position_update(
        self,
        pair_id: str,
        unrealized_pnl: float,
        portfolio_value: float,
        **kwargs
    ) -> None:
        """Log position update."""
        self.logger.info(
            "Position updated",
            event_type="position_update",
            pair_id=pair_id,
            unrealized_pnl=unrealized_pnl,
            portfolio_value=portfolio_value,
            **kwargs
        )
    
    def log_signal_generated(
        self,
        pair_id: str,
        signal_type: str,
        signal_strength: str,
        confidence: float,
        z_score: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log signal generation."""
        self.logger.info(
            "Signal generated",
            event_type="signal_generated",
            pair_id=pair_id,
            signal_type=signal_type,
            signal_strength=signal_strength,
            confidence=confidence,
            z_score=z_score,
            **kwargs
        )
    
    def log_state_transition(
        self,
        pair_id: str,
        from_state: str,
        to_state: str,
        trigger: str,
        **kwargs
    ) -> None:
        """Log state machine transition."""
        self.logger.info(
            "State transition",
            event_type="state_transition",
            pair_id=pair_id,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            **kwargs
        )


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, service_name: str):
        """Initialize performance logger."""
        self.logger = structlog.get_logger(f"{service_name}.performance")
    
    def log_service_startup(
        self,
        startup_time_seconds: float,
        **kwargs
    ) -> None:
        """Log service startup."""
        self.logger.info(
            "Service started",
            event_type="service_startup",
            startup_time_seconds=startup_time_seconds,
            **kwargs
        )
    
    def log_service_shutdown(
        self,
        uptime_seconds: float,
        **kwargs
    ) -> None:
        """Log service shutdown."""
        self.logger.info(
            "Service shutdown",
            event_type="service_shutdown",
            uptime_seconds=uptime_seconds,
            **kwargs
        )
    
    def log_kafka_lag(
        self,
        topic: str,
        partition: int,
        lag: int,
        **kwargs
    ) -> None:
        """Log Kafka consumer lag."""
        self.logger.info(
            "Kafka consumer lag",
            event_type="kafka_lag",
            topic=topic,
            partition=partition,
            lag=lag,
            **kwargs
        )
    
    def log_processing_time(
        self,
        operation: str,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log operation processing time."""
        self.logger.info(
            "Processing time",
            event_type="processing_time",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_error_rate(
        self,
        error_count: int,
        total_count: int,
        error_rate: float,
        time_window_seconds: int,
        **kwargs
    ) -> None:
        """Log error rate metrics."""
        self.logger.info(
            "Error rate",
            event_type="error_rate",
            error_count=error_count,
            total_count=total_count,
            error_rate=error_rate,
            time_window_seconds=time_window_seconds,
            **kwargs
        )
    
    def log_memory_usage(
        self,
        memory_mb: float,
        memory_percent: float,
        **kwargs
    ) -> None:
        """Log memory usage."""
        self.logger.info(
            "Memory usage",
            event_type="memory_usage",
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            **kwargs
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def get_trade_logger(service_name: str) -> TradeLogger:
    """Get a trade logger instance."""
    return TradeLogger(service_name)


def get_performance_logger(service_name: str) -> PerformanceLogger:
    """Get a performance logger instance."""
    return PerformanceLogger(service_name)
