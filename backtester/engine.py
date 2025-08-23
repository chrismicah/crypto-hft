"""Core backtesting engine with event-driven simulation."""

import asyncio
import heapq
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import uuid
import numpy as np

from .models import (
    BacktestEvent, MarketDataEvent, OrderEvent, EventType,
    Order, OrderSide, OrderType, OrderStatus, Fill, Position,
    BacktestConfig, SlippageModel, LatencyModel, FeeModel,
    OrderBook, Trade
)
from .data_loader import DataLoader
from .metrics import PerformanceCalculator, TradeMetrics
from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScheduledEvent:
    """Event scheduled for future execution."""
    timestamp: datetime
    event: BacktestEvent
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __lt__(self, other):
        """For heap ordering."""
        return self.timestamp < other.timestamp


class OrderManager:
    """Manages orders and fills in the backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize order manager."""
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.open_orders: Dict[str, Set[str]] = defaultdict(set)  # symbol -> order_ids
        self.fills: List[Fill] = []
        self.next_order_id = 1
        
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Create a new order."""
        order_id = f"ORDER_{self.next_order_id:06d}"
        self.next_order_id += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_time=datetime.utcnow()
        )
        
        self.orders[order_id] = order
        logger.debug("Created order", order_id=order_id, symbol=symbol, side=side.value)
        
        return order
    
    def submit_order(self, order: Order) -> None:
        """Submit order to the market."""
        order.status = OrderStatus.OPEN
        order.updated_time = datetime.utcnow()
        self.open_orders[order.symbol].add(order.order_id)
        
        logger.info("Order submitted", 
                   order_id=order.order_id,
                   symbol=order.symbol,
                   side=order.side.value,
                   quantity=order.quantity,
                   price=order.price)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status != OrderStatus.OPEN:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_time = datetime.utcnow()
        self.open_orders[order.symbol].discard(order_id)
        
        logger.info("Order cancelled", order_id=order_id)
        return True
    
    def fill_order(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        timestamp: datetime,
        fee: float = 0.0
    ) -> Optional[Fill]:
        """Fill an order (partially or completely)."""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
            return None
        
        # Create fill
        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            fee=fee,
            timestamp=timestamp,
            fill_id=f"FILL_{len(self.fills) + 1:06d}"
        )
        
        # Update order
        order.fill(fill_quantity, fill_price, fee)
        
        # Remove from open orders if completely filled
        if order.is_filled:
            self.open_orders[order.symbol].discard(order_id)
        
        self.fills.append(fill)
        
        logger.info("Order filled",
                   order_id=order_id,
                   fill_quantity=fill_quantity,
                   fill_price=fill_price,
                   remaining=order.remaining_quantity)
        
        return fill
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders for a symbol or all symbols."""
        if symbol:
            order_ids = self.open_orders.get(symbol, set())
            return [self.orders[oid] for oid in order_ids if oid in self.orders]
        else:
            all_open_orders = []
            for order_ids in self.open_orders.values():
                all_open_orders.extend([self.orders[oid] for oid in order_ids if oid in self.orders])
            return all_open_orders


class PositionManager:
    """Manages positions in the backtesting engine."""
    
    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, Position] = {}
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_position(self, fill: Fill) -> None:
        """Update position based on a fill."""
        position = self.get_position(fill.symbol)
        position.add_fill(fill)
        
        logger.debug("Position updated",
                    symbol=fill.symbol,
                    quantity=position.quantity,
                    average_price=position.average_price,
                    unrealized_pnl=position.unrealized_pnl)
    
    def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L for a position."""
        if symbol in self.positions:
            self.positions[symbol].update_unrealized_pnl(current_price)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()


class BacktestEngine:
    """Main backtesting engine with event-driven simulation."""
    
    def __init__(
        self,
        config: BacktestConfig,
        data_loader: DataLoader,
        strategy_callback: Callable[['BacktestEngine', BacktestEvent], None]
    ):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
            data_loader: Data loader for historical data
            strategy_callback: Strategy function to call for each event
        """
        self.config = config
        self.data_loader = data_loader
        self.strategy_callback = strategy_callback
        
        # Core components
        self.order_manager = OrderManager(config)
        self.position_manager = PositionManager()
        self.performance_calculator = PerformanceCalculator(config)
        
        # Event management
        self.event_queue: List[ScheduledEvent] = []
        self.current_time: Optional[datetime] = None
        self.market_data: Dict[str, OrderBook] = {}
        self.last_prices: Dict[str, float] = {}
        
        # State tracking
        self.is_running = False
        self.total_events_processed = 0
        self.strategy_errors = 0
        
        logger.info("Backtesting engine initialized",
                   symbols=config.symbols,
                   start_time=config.start_time,
                   end_time=config.end_time,
                   initial_capital=config.initial_capital)
    
    def schedule_event(self, event: BacktestEvent, delay: timedelta = timedelta(0)) -> None:
        """Schedule an event for future execution."""
        if not self.current_time:
            execution_time = event.timestamp + delay
        else:
            execution_time = self.current_time + delay
        
        scheduled_event = ScheduledEvent(
            timestamp=execution_time,
            event=event
        )
        
        heapq.heappush(self.event_queue, scheduled_event)
        
        logger.debug("Event scheduled",
                    event_type=event.event_type.value,
                    execution_time=execution_time,
                    delay_ms=delay.total_seconds() * 1000)
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """Place an order (returns order ID)."""
        # Create order
        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # Apply latency
        latency = self.config.latency_model.get_order_latency(order)
        
        # Schedule order submission
        order_event = OrderEvent(
            timestamp=self.current_time,
            symbol=symbol,
            order=order
        )
        
        self.schedule_event(order_event, delay=latency)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.order_manager.cancel_order(order_id)
    
    def get_position(self, symbol: str) -> Position:
        """Get current position for a symbol."""
        return self.position_manager.get_position(symbol)
    
    def get_market_data(self, symbol: str) -> Optional[OrderBook]:
        """Get current market data for a symbol."""
        return self.market_data.get(symbol)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for a symbol."""
        return self.last_prices.get(symbol)
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        cash = self.performance_calculator.cash
        position_value = 0.0
        
        for symbol, position in self.position_manager.get_all_positions().items():
            if not position.is_flat and symbol in self.last_prices:
                position_value += position.quantity * self.last_prices[symbol]
        
        return cash + position_value
    
    async def run(self) -> None:
        """Run the backtest."""
        logger.info("Starting backtest",
                   start_time=self.config.start_time,
                   end_time=self.config.end_time)
        
        self.is_running = True
        start_time = datetime.utcnow()
        
        try:
            # Load and schedule market data events
            await self._load_market_data()
            
            # Process events
            await self._process_events()
            
            # Calculate final metrics
            final_metrics = self.performance_calculator.calculate_metrics()
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info("Backtest completed",
                       total_events=self.total_events_processed,
                       strategy_errors=self.strategy_errors,
                       execution_time_seconds=execution_time,
                       final_return=f"{final_metrics.total_return:.2%}",
                       sharpe_ratio=f"{final_metrics.sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error("Backtest failed", error=str(e), exc_info=True)
            raise
        finally:
            self.is_running = False
    
    async def _load_market_data(self) -> None:
        """Load market data and schedule events."""
        logger.info("Loading market data", symbols=self.config.symbols)
        
        # Load data for all symbols
        events = list(self.data_loader.load_multiple_symbols(
            symbols=self.config.symbols,
            start_time=self.config.start_time,
            end_time=self.config.end_time
        ))
        
        logger.info("Loaded market data", total_events=len(events))
        
        # Schedule all market data events
        for event in events:
            # Apply market data latency
            latency = self.config.latency_model.get_market_data_latency()
            self.schedule_event(event, delay=latency)
    
    async def _process_events(self) -> None:
        """Process all events in chronological order."""
        logger.info("Processing events", total_events=len(self.event_queue))
        
        while self.event_queue and self.is_running:
            # Get next event
            scheduled_event = heapq.heappop(self.event_queue)
            event = scheduled_event.event
            
            # Update current time
            self.current_time = scheduled_event.timestamp
            
            # Skip events outside our time range
            if (self.current_time < self.config.start_time or 
                self.current_time > self.config.end_time):
                continue
            
            try:
                await self._process_event(event)
                self.total_events_processed += 1
                
                # Log progress periodically
                if self.total_events_processed % 10000 == 0:
                    logger.info("Processing progress",
                               events_processed=self.total_events_processed,
                               current_time=self.current_time,
                               portfolio_value=self.get_portfolio_value())
                
            except Exception as e:
                logger.error("Error processing event",
                           event_type=event.event_type.value,
                           timestamp=event.timestamp,
                           error=str(e))
                self.strategy_errors += 1
                
                # Continue processing unless too many errors
                if self.strategy_errors > 100:
                    logger.error("Too many strategy errors, stopping backtest")
                    break
    
    async def _process_event(self, event: BacktestEvent) -> None:
        """Process a single event."""
        if event.event_type == EventType.MARKET_DATA:
            await self._process_market_data_event(event)
        elif event.event_type == EventType.ORDER_PLACED:
            await self._process_order_event(event)
        elif event.event_type == EventType.ORDER_FILLED:
            await self._process_fill_event(event)
        
        # Call strategy callback
        try:
            if asyncio.iscoroutinefunction(self.strategy_callback):
                await self.strategy_callback(self, event)
            else:
                self.strategy_callback(self, event)
        except Exception as e:
            logger.error("Strategy callback error", error=str(e))
            self.strategy_errors += 1
    
    async def _process_market_data_event(self, event: MarketDataEvent) -> None:
        """Process market data event."""
        symbol = event.symbol
        
        # Update market data
        if event.order_book:
            self.market_data[symbol] = event.order_book
            
            # Update last price
            mid_price = event.order_book.get_mid_price()
            if mid_price:
                self.last_prices[symbol] = mid_price
        
        elif event.trade:
            self.last_prices[symbol] = event.trade.price
        
        # Update unrealized P&L for positions
        if symbol in self.last_prices:
            self.position_manager.update_unrealized_pnl(symbol, self.last_prices[symbol])
        
        # Update portfolio value
        self.performance_calculator.update_portfolio_value(
            timestamp=event.timestamp,
            market_prices=self.last_prices
        )
        
        # Check for order fills
        await self._check_order_fills(symbol)
    
    async def _process_order_event(self, event: OrderEvent) -> None:
        """Process order placement event."""
        order = event.order
        
        # Submit order to market
        self.order_manager.submit_order(order)
        
        # For market orders, try to fill immediately
        if order.order_type == OrderType.MARKET:
            await self._try_fill_market_order(order)
    
    async def _process_fill_event(self, event: OrderEvent) -> None:
        """Process order fill event."""
        # This would be called for fills that were scheduled with latency
        pass
    
    async def _check_order_fills(self, symbol: str) -> None:
        """Check if any open orders can be filled."""
        if symbol not in self.market_data:
            return
        
        order_book = self.market_data[symbol]
        open_orders = self.order_manager.get_open_orders(symbol)
        
        for order in open_orders:
            if order.order_type == OrderType.LIMIT:
                await self._try_fill_limit_order(order, order_book)
            elif order.order_type == OrderType.STOP:
                await self._try_fill_stop_order(order, order_book)
    
    async def _try_fill_market_order(self, order: Order) -> None:
        """Try to fill a market order."""
        if order.symbol not in self.market_data:
            return
        
        order_book = self.market_data[order.symbol]
        
        # Determine fill price
        if order.side == OrderSide.BUY:
            best_ask = order_book.get_best_ask()
            if not best_ask:
                return
            fill_price = best_ask.price
        else:
            best_bid = order_book.get_best_bid()
            if not best_bid:
                return
            fill_price = best_bid.price
        
        # Apply slippage
        slippage = self.config.slippage_model.calculate_slippage(
            order, order_book, self.config.market_impact_factor
        )
        
        if order.side == OrderSide.BUY:
            fill_price *= (1 + slippage)
        else:
            fill_price *= (1 - slippage)
        
        # Calculate fee
        temp_fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining_quantity,
            price=fill_price,
            fee=0.0,
            timestamp=self.current_time
        )
        
        fee = self.config.fee_model.calculate_fee(temp_fill)
        
        # Fill the order
        fill = self.order_manager.fill_order(
            order_id=order.order_id,
            fill_quantity=order.remaining_quantity,
            fill_price=fill_price,
            timestamp=self.current_time,
            fee=fee
        )
        
        if fill:
            # Update position
            self.position_manager.update_position(fill)
            
            # Add to performance calculator
            self.performance_calculator.add_fill(fill)
    
    async def _try_fill_limit_order(self, order: Order, order_book: OrderBook) -> None:
        """Try to fill a limit order."""
        if not order.price:
            return
        
        fill_price = None
        
        if order.side == OrderSide.BUY:
            best_ask = order_book.get_best_ask()
            if best_ask and best_ask.price <= order.price:
                fill_price = min(order.price, best_ask.price)
        else:
            best_bid = order_book.get_best_bid()
            if best_bid and best_bid.price >= order.price:
                fill_price = max(order.price, best_bid.price)
        
        if fill_price:
            # Calculate fee
            temp_fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.remaining_quantity,
                price=fill_price,
                fee=0.0,
                timestamp=self.current_time
            )
            
            fee = self.config.fee_model.calculate_fee(temp_fill)
            
            # Fill the order
            fill = self.order_manager.fill_order(
                order_id=order.order_id,
                fill_quantity=order.remaining_quantity,
                fill_price=fill_price,
                timestamp=self.current_time,
                fee=fee
            )
            
            if fill:
                # Update position
                self.position_manager.update_position(fill)
                
                # Add to performance calculator
                self.performance_calculator.add_fill(fill)
    
    async def _try_fill_stop_order(self, order: Order, order_book: OrderBook) -> None:
        """Try to fill a stop order."""
        if not order.stop_price:
            return
        
        mid_price = order_book.get_mid_price()
        if not mid_price:
            return
        
        # Check if stop is triggered
        triggered = False
        
        if order.side == OrderSide.BUY and mid_price >= order.stop_price:
            triggered = True
        elif order.side == OrderSide.SELL and mid_price <= order.stop_price:
            triggered = True
        
        if triggered:
            # Convert to market order
            order.order_type = OrderType.MARKET
            order.stop_price = None
            await self._try_fill_market_order(order)
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtesting results."""
        metrics = self.performance_calculator.calculate_metrics()
        
        return {
            'metrics': metrics.to_dict(),
            'portfolio_timeseries': self.performance_calculator.get_portfolio_timeseries(),
            'trade_analysis': self.performance_calculator.get_trade_analysis(),
            'fills': [
                {
                    'fill_id': fill.fill_id,
                    'order_id': fill.order_id,
                    'symbol': fill.symbol,
                    'side': fill.side.value,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'fee': fill.fee,
                    'timestamp': fill.timestamp.isoformat()
                }
                for fill in self.order_manager.fills
            ],
            'orders': [
                {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'order_type': order.order_type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'status': order.status.value,
                    'filled_quantity': order.filled_quantity,
                    'average_fill_price': order.average_fill_price,
                    'fees': order.fees,
                    'created_time': order.created_time.isoformat() if order.created_time else None,
                    'updated_time': order.updated_time.isoformat() if order.updated_time else None
                }
                for order in self.order_manager.orders.values()
            ],
            'final_positions': {
                symbol: {
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl
                }
                for symbol, position in self.position_manager.get_all_positions().items()
                if not position.is_flat
            }
        }
