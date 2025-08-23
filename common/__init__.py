"""Common utilities and shared components for the crypto trading bot."""

from .logger import (
    configure_logging,
    get_logger,
    get_trade_logger,
    get_performance_logger,
    TradeLogger,
    PerformanceLogger
)

__all__ = [
    'configure_logging',
    'get_logger',
    'get_trade_logger',
    'get_performance_logger',
    'TradeLogger',
    'PerformanceLogger'
]

__version__ = '1.0.0'
