"""Data models for backtesting engine including slippage, latency, and fee models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class EventType(Enum):
    """Types of events in the backtesting engine."""
    MARKET_DATA = "market_data"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    SIGNAL_GENERATED = "signal_generated"
    POSITION_UPDATE = "position_update"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PriceLevel:
    """Order book price level."""
    price: float
    quantity: float
    
    def __post_init__(self):
        self.price = float(self.price)
        self.quantity = float(self.quantity)


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    sequence_id: int = 0
    
    def get_best_bid(self) -> Optional[PriceLevel]:
        """Get best bid price level."""
        return max(self.bids, key=lambda x: x.price) if self.bids else None
    
    def get_best_ask(self) -> Optional[PriceLevel]:
        """Get best ask price level."""
        return min(self.asks, key=lambda x: x.price) if self.asks else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_depth(self, side: OrderSide, levels: int = 5) -> List[PriceLevel]:
        """Get order book depth for specified side."""
        if side == OrderSide.BUY:
            return sorted(self.bids, key=lambda x: x.price, reverse=True)[:levels]
        else:
            return sorted(self.asks, key=lambda x: x.price)[:levels]


@dataclass
class Trade:
    """Individual trade/tick data."""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: OrderSide
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        self.price = float(self.price)
        self.quantity = float(self.quantity)


@dataclass
class Order:
    """Order representation in backtesting."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    fees: float = 0.0
    
    def __post_init__(self):
        self.quantity = float(self.quantity)
        self.filled_quantity = float(self.filled_quantity)
        if self.price is not None:
            self.price = float(self.price)
        if self.stop_price is not None:
            self.stop_price = float(self.stop_price)
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return abs(self.remaining_quantity) < 1e-8
    
    def fill(self, quantity: float, price: float, fee: float = 0.0) -> None:
        """Fill order with specified quantity and price."""
        fill_qty = min(quantity, self.remaining_quantity)
        
        # Update average fill price
        total_filled_value = self.average_fill_price * self.filled_quantity
        new_fill_value = price * fill_qty
        self.filled_quantity += fill_qty
        
        if self.filled_quantity > 0:
            self.average_fill_price = (total_filled_value + new_fill_value) / self.filled_quantity
        
        self.fees += fee
        self.updated_time = datetime.utcnow()
        
        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class Fill:
    """Order fill representation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime
    fill_id: Optional[str] = None
    
    def __post_init__(self):
        self.quantity = float(self.quantity)
        self.price = float(self.price)
        self.fee = float(self.fee)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def __post_init__(self):
        self.quantity = float(self.quantity)
        self.average_price = float(self.average_price)
        self.unrealized_pnl = float(self.unrealized_pnl)
        self.realized_pnl = float(self.realized_pnl)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.quantity) < 1e-8
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        if not self.is_flat:
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
    
    def add_fill(self, fill: Fill) -> None:
        """Add a fill to the position."""
        fill_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        
        if self.is_flat:
            # Opening new position
            self.quantity = fill_qty
            self.average_price = fill.price
        elif (self.quantity > 0 and fill_qty > 0) or (self.quantity < 0 and fill_qty < 0):
            # Adding to existing position
            total_cost = self.average_price * abs(self.quantity) + fill.price * abs(fill_qty)
            self.quantity += fill_qty
            if not self.is_flat:
                self.average_price = total_cost / abs(self.quantity)
        else:
            # Reducing or closing position
            if abs(fill_qty) >= abs(self.quantity):
                # Closing and potentially reversing position
                closing_qty = self.quantity
                self.realized_pnl += (fill.price - self.average_price) * abs(closing_qty)
                
                remaining_qty = fill_qty + closing_qty
                if abs(remaining_qty) > 1e-8:
                    # Reversing position
                    self.quantity = remaining_qty
                    self.average_price = fill.price
                else:
                    # Flat position
                    self.quantity = 0.0
                    self.average_price = 0.0
            else:
                # Partial close
                self.realized_pnl += (fill.price - self.average_price) * abs(fill_qty)
                self.quantity += fill_qty


@dataclass
class BacktestEvent:
    """Base event class for backtesting."""
    timestamp: datetime
    event_type: EventType
    symbol: str
    data: Dict = field(default_factory=dict)


@dataclass
class MarketDataEvent(BacktestEvent):
    """Market data event."""
    order_book: Optional[OrderBook] = None
    trade: Optional[Trade] = None
    
    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA


@dataclass
class OrderEvent(BacktestEvent):
    """Order-related event."""
    order: Order = None
    
    def __post_init__(self):
        if self.order.status == OrderStatus.PENDING:
            self.event_type = EventType.ORDER_PLACED
        elif self.order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            self.event_type = EventType.ORDER_FILLED
        elif self.order.status == OrderStatus.CANCELLED:
            self.event_type = EventType.ORDER_CANCELLED


# Abstract base classes for models

class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self,
        order: Order,
        order_book: OrderBook,
        market_impact_factor: float = 1.0
    ) -> float:
        """Calculate slippage for an order."""
        pass


class LatencyModel(ABC):
    """Abstract base class for latency models."""
    
    @abstractmethod
    def get_order_latency(self, order: Order) -> timedelta:
        """Get latency for order placement."""
        pass
    
    @abstractmethod
    def get_market_data_latency(self) -> timedelta:
        """Get latency for market data."""
        pass


class FeeModel(ABC):
    """Abstract base class for fee models."""
    
    @abstractmethod
    def calculate_fee(self, fill: Fill) -> float:
        """Calculate trading fee for a fill."""
        pass


# Concrete implementations

class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size and spread."""
    
    def __init__(
        self,
        base_slippage_bps: float = 0.5,
        size_impact_factor: float = 0.1,
        max_slippage_bps: float = 10.0
    ):
        """
        Initialize linear slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points
            size_impact_factor: Factor for size-based impact
            max_slippage_bps: Maximum slippage in basis points
        """
        self.base_slippage_bps = base_slippage_bps
        self.size_impact_factor = size_impact_factor
        self.max_slippage_bps = max_slippage_bps
    
    def calculate_slippage(
        self,
        order: Order,
        order_book: OrderBook,
        market_impact_factor: float = 1.0
    ) -> float:
        """Calculate slippage for an order."""
        if not order_book.bids or not order_book.asks:
            return self.max_slippage_bps / 10000  # Return max slippage if no book
        
        mid_price = order_book.get_mid_price()
        if not mid_price:
            return self.max_slippage_bps / 10000
        
        spread = order_book.get_spread()
        if not spread:
            spread = mid_price * 0.001  # Default 0.1% spread
        
        # Calculate size impact based on order size relative to top level
        if order.side == OrderSide.BUY:
            top_level = order_book.get_best_ask()
        else:
            top_level = order_book.get_best_bid()
        
        if not top_level:
            return self.max_slippage_bps / 10000
        
        size_ratio = order.quantity / top_level.quantity
        size_impact = self.size_impact_factor * size_ratio
        
        # Total slippage
        spread_bps = (spread / mid_price) * 10000
        total_slippage_bps = (
            self.base_slippage_bps +
            spread_bps * 0.5 +  # Half spread crossing
            size_impact * market_impact_factor
        )
        
        # Cap at maximum slippage
        total_slippage_bps = min(total_slippage_bps, self.max_slippage_bps)
        
        return total_slippage_bps / 10000


class ConstantLatencyModel(LatencyModel):
    """Constant latency model."""
    
    def __init__(
        self,
        order_latency_ms: float = 50.0,
        market_data_latency_ms: float = 10.0
    ):
        """
        Initialize constant latency model.
        
        Args:
            order_latency_ms: Order placement latency in milliseconds
            market_data_latency_ms: Market data latency in milliseconds
        """
        self.order_latency = timedelta(milliseconds=order_latency_ms)
        self.market_data_latency = timedelta(milliseconds=market_data_latency_ms)
    
    def get_order_latency(self, order: Order) -> timedelta:
        """Get latency for order placement."""
        return self.order_latency
    
    def get_market_data_latency(self) -> timedelta:
        """Get latency for market data."""
        return self.market_data_latency


class NormalDistributionLatencyModel(LatencyModel):
    """Latency model with normal distribution."""
    
    def __init__(
        self,
        order_latency_mean_ms: float = 50.0,
        order_latency_std_ms: float = 10.0,
        market_data_latency_mean_ms: float = 10.0,
        market_data_latency_std_ms: float = 2.0,
        min_latency_ms: float = 1.0,
        max_latency_ms: float = 500.0
    ):
        """
        Initialize normal distribution latency model.
        
        Args:
            order_latency_mean_ms: Mean order latency in milliseconds
            order_latency_std_ms: Standard deviation of order latency
            market_data_latency_mean_ms: Mean market data latency
            market_data_latency_std_ms: Standard deviation of market data latency
            min_latency_ms: Minimum latency
            max_latency_ms: Maximum latency
        """
        self.order_mean = order_latency_mean_ms
        self.order_std = order_latency_std_ms
        self.market_data_mean = market_data_latency_mean_ms
        self.market_data_std = market_data_latency_std_ms
        self.min_latency = min_latency_ms
        self.max_latency = max_latency_ms
    
    def get_order_latency(self, order: Order) -> timedelta:
        """Get latency for order placement."""
        latency_ms = np.random.normal(self.order_mean, self.order_std)
        latency_ms = np.clip(latency_ms, self.min_latency, self.max_latency)
        return timedelta(milliseconds=latency_ms)
    
    def get_market_data_latency(self) -> timedelta:
        """Get latency for market data."""
        latency_ms = np.random.normal(self.market_data_mean, self.market_data_std)
        latency_ms = np.clip(latency_ms, self.min_latency, self.max_latency)
        return timedelta(milliseconds=latency_ms)


class TieredFeeModel(FeeModel):
    """Tiered fee model based on trading volume."""
    
    def __init__(
        self,
        maker_fee_bps: float = 1.0,
        taker_fee_bps: float = 1.5,
        volume_tiers: Optional[Dict[float, Tuple[float, float]]] = None
    ):
        """
        Initialize tiered fee model.
        
        Args:
            maker_fee_bps: Base maker fee in basis points
            taker_fee_bps: Base taker fee in basis points
            volume_tiers: Dict of volume thresholds to (maker_fee, taker_fee) tuples
        """
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.volume_tiers = volume_tiers or {}
        self.total_volume = 0.0
    
    def calculate_fee(self, fill: Fill) -> float:
        """Calculate trading fee for a fill."""
        # Update total volume
        self.total_volume += fill.quantity * fill.price
        
        # Determine fee tier
        maker_fee = self.maker_fee_bps
        taker_fee = self.taker_fee_bps
        
        for volume_threshold, (tier_maker, tier_taker) in sorted(self.volume_tiers.items()):
            if self.total_volume >= volume_threshold:
                maker_fee = tier_maker
                taker_fee = tier_taker
            else:
                break
        
        # For backtesting, assume all orders are taker orders (conservative)
        fee_rate = taker_fee / 10000
        return fill.quantity * fill.price * fee_rate


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    start_time: datetime
    end_time: datetime
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    slippage_model: SlippageModel = field(default_factory=LinearSlippageModel)
    latency_model: LatencyModel = field(default_factory=ConstantLatencyModel)
    fee_model: FeeModel = field(default_factory=TieredFeeModel)
    market_impact_factor: float = 1.0
    enable_short_selling: bool = True
    max_position_size: Optional[float] = None
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    def __post_init__(self):
        self.initial_capital = float(self.initial_capital)
        self.market_impact_factor = float(self.market_impact_factor)
        self.risk_free_rate = float(self.risk_free_rate)
