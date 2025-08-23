"""Kalman Filter service for dynamic hedge ratio calculation."""

from .filter import DynamicHedgeRatioKalman, PairTradingKalmanFilter, KalmanState
from .state import StateManager
from .main import KalmanFilterService

__all__ = [
    'DynamicHedgeRatioKalman',
    'PairTradingKalmanFilter', 
    'KalmanState',
    'StateManager',
    'KalmanFilterService'
]

__version__ = '1.0.0'
