"""Core execution service for trading strategy orchestration."""

from .exchange import (
    BinanceTestnetExchange,
    ExchangeManager,
    OrderResult,
    PositionInfo,
    BalanceInfo
)
from .signals import (
    SignalGenerator,
    SpreadCalculator,
    MarketData,
    TradingSignal,
    SignalType,
    SignalStrength
)
from .state_machine import (
    StrategyStateMachine,
    OrderSizer,
    TradingState,
    PositionSide,
    PositionInfo as StateMachinePositionInfo,
    StateTransitionEvent
)
from .main import ExecutionService, DataAggregator

__all__ = [
    'BinanceTestnetExchange',
    'ExchangeManager',
    'OrderResult',
    'PositionInfo',
    'BalanceInfo',
    'SignalGenerator',
    'SpreadCalculator',
    'MarketData',
    'TradingSignal',
    'SignalType',
    'SignalStrength',
    'StrategyStateMachine',
    'OrderSizer',
    'TradingState',
    'PositionSide',
    'StateMachinePositionInfo',
    'StateTransitionEvent',
    'ExecutionService',
    'DataAggregator'
]

__version__ = '1.0.0'
