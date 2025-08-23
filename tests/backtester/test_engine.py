"""Unit tests for backtesting engine."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from backtester.engine import BacktestEngine, OrderManager, PositionManager
from backtester.models import (
    BacktestConfig, Order, OrderSide, OrderType, OrderStatus,
    LinearSlippageModel, ConstantLatencyModel, TieredFeeModel,
    BacktestEvent, MarketDataEvent, EventType, OrderBook, PriceLevel
)
from backtester.data_loader import DataLoader, SyntheticDataSource


class TestOrderManager:
    """Test cases for OrderManager."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return BacktestConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            initial_capital=100000.0,
            symbols=['BTCUSDT']
        )
    
    @pytest.fixture
    def order_manager(self, config):
        """OrderManager instance."""
        return OrderManager(config)
    
    def test_create_order(self, order_manager):
        """Test creating an order."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        assert order.order_id == "ORDER_000001"
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1.0
        assert order.price == 50000.0
        assert order.status == OrderStatus.PENDING
        
        # Should be stored in orders dict
        assert order.order_id in order_manager.orders
    
    def test_submit_order(self, order_manager):
        """Test submitting an order."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        order_manager.submit_order(order)
        
        assert order.status == OrderStatus.OPEN
        assert order.order_id in order_manager.open_orders["BTCUSDT"]
    
    def test_cancel_order(self, order_manager):
        """Test cancelling an order."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        order_manager.submit_order(order)
        
        # Cancel the order
        success = order_manager.cancel_order(order.order_id)
        
        assert success
        assert order.status == OrderStatus.CANCELLED
        assert order.order_id not in order_manager.open_orders["BTCUSDT"]
    
    def test_cancel_nonexistent_order(self, order_manager):
        """Test cancelling a non-existent order."""
        success = order_manager.cancel_order("NONEXISTENT")
        assert not success
    
    def test_fill_order(self, order_manager):
        """Test filling an order."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0
        )
        
        order_manager.submit_order(order)
        
        # Partial fill
        fill = order_manager.fill_order(
            order_id=order.order_id,
            fill_quantity=1.0,
            fill_price=50000.0,
            timestamp=datetime.utcnow(),
            fee=25.0
        )
        
        assert fill is not None
        assert fill.quantity == 1.0
        assert fill.price == 50000.0
        assert fill.fee == 25.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 1.0
        assert order.remaining_quantity == 1.0
        
        # Complete fill
        fill2 = order_manager.fill_order(
            order_id=order.order_id,
            fill_quantity=1.0,
            fill_price=50100.0,
            timestamp=datetime.utcnow(),
            fee=25.0
        )
        
        assert fill2 is not None
        assert order.status == OrderStatus.FILLED
        assert order.is_filled
        assert order.order_id not in order_manager.open_orders["BTCUSDT"]
    
    def test_get_open_orders(self, order_manager):
        """Test getting open orders."""
        # Create and submit orders
        order1 = order_manager.create_order("BTCUSDT", OrderSide.BUY, OrderType.LIMIT, 1.0, 50000.0)
        order2 = order_manager.create_order("ETHUSDT", OrderSide.SELL, OrderType.LIMIT, 1.0, 3000.0)
        
        order_manager.submit_order(order1)
        order_manager.submit_order(order2)
        
        # Get all open orders
        all_open = order_manager.get_open_orders()
        assert len(all_open) == 2
        
        # Get open orders for specific symbol
        btc_open = order_manager.get_open_orders("BTCUSDT")
        assert len(btc_open) == 1
        assert btc_open[0].symbol == "BTCUSDT"


class TestPositionManager:
    """Test cases for PositionManager."""
    
    @pytest.fixture
    def position_manager(self):
        """PositionManager instance."""
        return PositionManager()
    
    def test_get_position(self, position_manager):
        """Test getting a position."""
        position = position_manager.get_position("BTCUSDT")
        
        assert position.symbol == "BTCUSDT"
        assert position.is_flat
        assert "BTCUSDT" in position_manager.positions
    
    def test_update_position_with_fill(self, position_manager):
        """Test updating position with a fill."""
        from backtester.models import Fill
        
        fill = Fill(
            order_id="ORDER_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            timestamp=datetime.utcnow()
        )
        
        position_manager.update_position(fill)
        
        position = position_manager.get_position("BTCUSDT")
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
    
    def test_update_unrealized_pnl(self, position_manager):
        """Test updating unrealized P&L."""
        from backtester.models import Fill
        
        # Create position
        fill = Fill(
            order_id="ORDER_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            timestamp=datetime.utcnow()
        )
        
        position_manager.update_position(fill)
        
        # Update P&L
        position_manager.update_unrealized_pnl("BTCUSDT", 51000.0)
        
        position = position_manager.get_position("BTCUSDT")
        assert position.unrealized_pnl == 1000.0  # (51000 - 50000) * 1
    
    def test_get_all_positions(self, position_manager):
        """Test getting all positions."""
        from backtester.models import Fill
        
        # Create positions for multiple symbols
        fill1 = Fill("ORDER_001", "BTCUSDT", OrderSide.BUY, 1.0, 50000.0, 25.0, datetime.utcnow())
        fill2 = Fill("ORDER_002", "ETHUSDT", OrderSide.SELL, 1.0, 3000.0, 15.0, datetime.utcnow())
        
        position_manager.update_position(fill1)
        position_manager.update_position(fill2)
        
        all_positions = position_manager.get_all_positions()
        
        assert len(all_positions) == 2
        assert "BTCUSDT" in all_positions
        assert "ETHUSDT" in all_positions


class TestBacktestEngine:
    """Test cases for BacktestEngine."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return BacktestConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1, 1, 0),  # 1 hour
            initial_capital=100000.0,
            symbols=['BTCUSDT'],
            slippage_model=LinearSlippageModel(base_slippage_bps=1.0),
            latency_model=ConstantLatencyModel(order_latency_ms=10.0),
            fee_model=TieredFeeModel(taker_fee_bps=1.0)
        )
    
    @pytest.fixture
    def data_loader(self):
        """Test data loader."""
        loader = DataLoader()
        synthetic_source = SyntheticDataSource(
            symbols=['BTCUSDT'],
            start_price=50000.0,
            volatility=0.01,
            tick_frequency_ms=60000  # 1 minute
        )
        loader.add_data_source("synthetic", synthetic_source, is_default=True)
        return loader
    
    @pytest.fixture
    def strategy_callback(self):
        """Mock strategy callback."""
        return AsyncMock()
    
    @pytest.fixture
    def engine(self, config, data_loader, strategy_callback):
        """BacktestEngine instance."""
        return BacktestEngine(config, data_loader, strategy_callback)
    
    def test_engine_initialization(self, engine, config):
        """Test engine initialization."""
        assert engine.config == config
        assert engine.order_manager is not None
        assert engine.position_manager is not None
        assert engine.performance_calculator is not None
        assert not engine.is_running
        assert engine.total_events_processed == 0
    
    def test_schedule_event(self, engine):
        """Test scheduling an event."""
        event = BacktestEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            event_type=EventType.MARKET_DATA,
            symbol="BTCUSDT"
        )
        
        engine.current_time = datetime(2024, 1, 1, 9, 59)
        engine.schedule_event(event, delay=timedelta(minutes=1))
        
        assert len(engine.event_queue) == 1
        scheduled_event = engine.event_queue[0]
        assert scheduled_event.timestamp == datetime(2024, 1, 1, 10, 0)
    
    def test_place_order(self, engine):
        """Test placing an order."""
        engine.current_time = datetime(2024, 1, 1, 10, 0)
        
        order_id = engine.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        assert order_id.startswith("ORDER_")
        assert order_id in engine.order_manager.orders
        
        # Should have scheduled an order event
        assert len(engine.event_queue) == 1
    
    def test_get_position(self, engine):
        """Test getting a position."""
        position = engine.get_position("BTCUSDT")
        
        assert position.symbol == "BTCUSDT"
        assert position.is_flat
    
    def test_get_portfolio_value(self, engine):
        """Test getting portfolio value."""
        # Initially should equal initial capital
        portfolio_value = engine.get_portfolio_value()
        assert portfolio_value == engine.config.initial_capital
        
        # Add a position and update
        from backtester.models import Fill
        fill = Fill("ORDER_001", "BTCUSDT", OrderSide.BUY, 1.0, 50000.0, 25.0, datetime.utcnow())
        engine.performance_calculator.add_fill(fill)
        engine.position_manager.update_position(fill)
        engine.last_prices["BTCUSDT"] = 51000.0
        
        portfolio_value = engine.get_portfolio_value()
        expected_value = engine.performance_calculator.cash + 1.0 * 51000.0
        assert portfolio_value == expected_value
    
    @pytest.mark.asyncio
    async def test_process_market_data_event(self, engine):
        """Test processing market data event."""
        # Create order book
        bids = [PriceLevel("50000.0", "1.0")]
        asks = [PriceLevel("50010.0", "1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime(2024, 1, 1, 10, 0),
            bids=bids,
            asks=asks
        )
        
        event = MarketDataEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="BTCUSDT",
            order_book=order_book
        )
        
        engine.current_time = datetime(2024, 1, 1, 10, 0)
        
        await engine._process_market_data_event(event)
        
        # Should update market data and last prices
        assert "BTCUSDT" in engine.market_data
        assert "BTCUSDT" in engine.last_prices
        assert engine.last_prices["BTCUSDT"] == 50005.0  # Mid price
    
    @pytest.mark.asyncio
    async def test_try_fill_market_order(self, engine):
        """Test filling a market order."""
        # Set up market data
        bids = [PriceLevel("50000.0", "2.0")]
        asks = [PriceLevel("50010.0", "2.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime(2024, 1, 1, 10, 0),
            bids=bids,
            asks=asks
        )
        
        engine.market_data["BTCUSDT"] = order_book
        engine.current_time = datetime(2024, 1, 1, 10, 0)
        
        # Create and submit market order
        order = engine.order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        engine.order_manager.submit_order(order)
        
        # Try to fill the order
        await engine._try_fill_market_order(order)
        
        # Order should be filled
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order.average_fill_price > 50010.0  # Should include slippage
        
        # Should have created a fill
        assert len(engine.order_manager.fills) == 1
    
    @pytest.mark.asyncio
    async def test_try_fill_limit_order(self, engine):
        """Test filling a limit order."""
        # Set up market data
        bids = [PriceLevel("50000.0", "2.0")]
        asks = [PriceLevel("49990.0", "2.0")]  # Ask below limit price
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime(2024, 1, 1, 10, 0),
            bids=bids,
            asks=asks
        )
        
        engine.current_time = datetime(2024, 1, 1, 10, 0)
        
        # Create limit buy order above current ask
        order = engine.order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        engine.order_manager.submit_order(order)
        
        # Try to fill the order
        await engine._try_fill_limit_order(order, order_book)
        
        # Order should be filled at the better price
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order.average_fill_price == 49990.0  # Should get the ask price
    
    @pytest.mark.asyncio
    async def test_try_fill_limit_order_no_fill(self, engine):
        """Test limit order that shouldn't fill."""
        # Set up market data
        bids = [PriceLevel("50000.0", "2.0")]
        asks = [PriceLevel("50020.0", "2.0")]  # Ask above limit price
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime(2024, 1, 1, 10, 0),
            bids=bids,
            asks=asks
        )
        
        # Create limit buy order below current ask
        order = engine.order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50010.0
        )
        engine.order_manager.submit_order(order)
        
        # Try to fill the order
        await engine._try_fill_limit_order(order, order_book)
        
        # Order should not be filled
        assert order.status == OrderStatus.OPEN
        assert order.filled_quantity == 0.0
    
    def test_get_results(self, engine):
        """Test getting backtest results."""
        # Add some test data
        from backtester.models import Fill
        fill = Fill("ORDER_001", "BTCUSDT", OrderSide.BUY, 1.0, 50000.0, 25.0, datetime.utcnow())
        engine.order_manager.fills.append(fill)
        engine.performance_calculator.add_fill(fill)
        
        results = engine.get_results()
        
        assert 'metrics' in results
        assert 'portfolio_timeseries' in results
        assert 'trade_analysis' in results
        assert 'fills' in results
        assert 'orders' in results
        assert 'final_positions' in results
        
        # Should have one fill
        assert len(results['fills']) == 1
        assert results['fills'][0]['symbol'] == 'BTCUSDT'


class TestEventOrdering:
    """Test event ordering and processing."""
    
    @pytest.mark.asyncio
    async def test_event_ordering(self):
        """Test that events are processed in chronological order."""
        config = BacktestConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1, 1, 0),
            initial_capital=100000.0,
            symbols=['BTCUSDT']
        )
        
        data_loader = DataLoader()
        processed_events = []
        
        async def test_strategy(engine, event):
            processed_events.append(event.timestamp)
        
        engine = BacktestEngine(config, data_loader, test_strategy)
        
        # Schedule events out of order
        events = [
            BacktestEvent(datetime(2024, 1, 1, 0, 30), EventType.MARKET_DATA, "BTCUSDT"),
            BacktestEvent(datetime(2024, 1, 1, 0, 10), EventType.MARKET_DATA, "BTCUSDT"),
            BacktestEvent(datetime(2024, 1, 1, 0, 20), EventType.MARKET_DATA, "BTCUSDT"),
        ]
        
        for event in events:
            engine.schedule_event(event)
        
        # Process events
        await engine._process_events()
        
        # Should be processed in chronological order
        assert processed_events == [
            datetime(2024, 1, 1, 0, 10),
            datetime(2024, 1, 1, 0, 20),
            datetime(2024, 1, 1, 0, 30)
        ]
