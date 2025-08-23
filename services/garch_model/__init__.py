"""GARCH volatility forecasting service for dynamic threshold calculation."""

from .model import (
    RollingGARCHModel,
    MultiPairGARCHManager,
    GARCHForecast,
    GARCHModelState
)
from .thresholds import (
    DynamicThresholdCalculator,
    MultiPairThresholdManager,
    ThresholdSignal,
    AdaptiveThresholds
)
from .main import GARCHVolatilityService

__all__ = [
    'RollingGARCHModel',
    'MultiPairGARCHManager',
    'GARCHForecast',
    'GARCHModelState',
    'DynamicThresholdCalculator',
    'MultiPairThresholdManager',
    'ThresholdSignal',
    'AdaptiveThresholds',
    'GARCHVolatilityService'
]

__version__ = '1.0.0'
