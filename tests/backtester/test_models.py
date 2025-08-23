"""Unit tests for backtesting models."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from backtester.models import (
    Order, OrderSide, OrderType, OrderStatus, Fill, Position,
    PriceLevel, OrderBook, Trade, BacktestConfig,
    LinearSlippageModel, ConstantLatencyModel, NormalDistributionLatencyModel,
    TieredFeeModel
)


class TestOrder:
    """Test cases for Order model."""
    
    def test_order_creation(self):
        """Test creating an Order."""
        order = Order(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        assert order.order_id == "TEST_001"
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1.0
        assert order.price == 50000.0
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.remaining_quantity == 1.0
        assert not order.is_filled
    
    def test_order_fill(self):
        """Test filling an order."""
        order = Order(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0
        )
        
        # Partial fill
        order.fill(1.0, 50000.0, 5.0)
        
        assert order.filled_quantity == 1.0
        assert order.remaining_quantity == 1.0
        assert order.average_fill_price == 50000.0
        assert order.fees == 5.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert not order.is_filled
        
        # Complete fill
        order.fill(1.0, 50100.0, 5.0)
        
        assert order.filled_quantity == 2.0
        assert order.remaining_quantity == 0.0
        assert order.average_fill_price == 50050.0  # Average of 50000 and 50100
        assert order.fees == 10.0
        assert order.status == OrderStatus.FILLED
        assert order.is_filled


class TestPosition:
    """Test cases for Position model."""
    
    def test_position_creation(self):
        """Test creating a Position."""
        position = Position(symbol="BTCUSDT")
        
        assert position.symbol == "BTCUSDT"
        assert position.quantity == 0.0
        assert position.average_price == 0.0
        assert position.is_flat
        assert not position.is_long
        assert not position.is_short
    
    def test_position_long_entry(self):
        """Test entering a long position."""
        position = Position(symbol="BTCUSDT")
        
        fill = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        
        position.add_fill(fill)
        
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
        assert position.is_long
        assert not position.is_flat
        assert not position.is_short
    
    def test_position_short_entry(self):
        """Test entering a short position."""
        position = Position(symbol="BTCUSDT")
        
        fill = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        
        position.add_fill(fill)
        
        assert position.quantity == -1.0
        assert position.average_price == 50000.0
        assert position.is_short
        assert not position.is_flat
        assert not position.is_long
    
    def test_position_averaging(self):
        """Test position averaging."""
        position = Position(symbol="BTCUSDT")
        
        # First fill
        fill1 = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill1)
        
        # Second fill at higher price
        fill2 = Fill(
            order_id="TEST_002",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=52000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill2)
        
        assert position.quantity == 2.0
        assert position.average_price == 51000.0  # Average of 50000 and 52000
    
    def test_position_partial_close(self):
        """Test partially closing a position."""
        position = Position(symbol="BTCUSDT")
        
        # Open position
        fill1 = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=2.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill1)
        
        # Partial close
        fill2 = Fill(
            order_id="TEST_002",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=51000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill2)
        
        assert position.quantity == 1.0
        assert position.average_price == 50000.0  # Original entry price
        assert position.realized_pnl == 1000.0  # (51000 - 50000) * 1
    
    def test_position_full_close(self):
        """Test fully closing a position."""
        position = Position(symbol="BTCUSDT")
        
        # Open position
        fill1 = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill1)
        
        # Full close
        fill2 = Fill(
            order_id="TEST_002",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=51000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill2)
        
        assert position.quantity == 0.0
        assert position.is_flat
        assert position.realized_pnl == 1000.0
    
    def test_position_reverse(self):
        """Test reversing a position."""
        position = Position(symbol="BTCUSDT")
        
        # Open long position
        fill1 = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill1)
        
        # Reverse to short
        fill2 = Fill(
            order_id="TEST_002",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=2.0,
            price=51000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill2)
        
        assert position.quantity == -1.0  # Now short 1 unit
        assert position.average_price == 51000.0  # New entry price
        assert position.realized_pnl == 1000.0  # Profit from closing long
        assert position.is_short
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        position = Position(symbol="BTCUSDT")
        
        # Open long position
        fill = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=5.0,
            timestamp=datetime.utcnow()
        )
        position.add_fill(fill)
        
        # Update with current price
        position.update_unrealized_pnl(52000.0)
        
        assert position.unrealized_pnl == 2000.0  # (52000 - 50000) * 1


class TestOrderBook:
    """Test cases for OrderBook model."""
    
    def test_order_book_creation(self):
        """Test creating an OrderBook."""
        bids = [PriceLevel("50000.0", "1.0"), PriceLevel("49999.0", "2.0")]
        asks = [PriceLevel("50001.0", "1.5"), PriceLevel("50002.0", "1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
        
        assert order_book.symbol == "BTCUSDT"
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
    
    def test_best_bid_ask(self):
        """Test getting best bid and ask."""
        bids = [PriceLevel("50000.0", "1.0"), PriceLevel("49999.0", "2.0")]
        asks = [PriceLevel("50001.0", "1.5"), PriceLevel("50002.0", "1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
        
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        assert best_bid.price == 50000.0
        assert best_ask.price == 50001.0
    
    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        bids = [PriceLevel("50000.0", "1.0")]
        asks = [PriceLevel("50002.0", "1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
        
        mid_price = order_book.get_mid_price()
        assert mid_price == 50001.0  # (50000 + 50002) / 2
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        bids = [PriceLevel("50000.0", "1.0")]
        asks = [PriceLevel("50002.0", "1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
        
        spread = order_book.get_spread()
        assert spread == 2.0  # 50002 - 50000


class TestSlippageModel:
    """Test cases for SlippageModel."""
    
    def test_linear_slippage_model(self):
        """Test LinearSlippageModel."""
        model = LinearSlippageModel(
            base_slippage_bps=1.0,
            size_impact_factor=0.1,
            max_slippage_bps=10.0
        )
        
        # Create test order and order book
        order = Order(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        bids = [PriceLevel("50000.0", "2.0")]
        asks = [PriceLevel("50010.0", "2.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
        
        slippage = model.calculate_slippage(order, order_book)
        
        # Should be positive and reasonable
        assert 0 < slippage < 0.01  # Less than 1%
    
    def test_slippage_with_large_order(self):
        """Test slippage increases with order size."""
        model = LinearSlippageModel(
            base_slippage_bps=1.0,
            size_impact_factor=0.5,
            max_slippage_bps=50.0
        )
        
        # Small order
        small_order = Order(
            order_id="SMALL",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        # Large order
        large_order = Order(
            order_id="LARGE",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0
        )
        
        bids = [PriceLevel("50000.0", "1.0")]
        asks = [PriceLevel("50010.0", "1.0")]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
        
        small_slippage = model.calculate_slippage(small_order, order_book)
        large_slippage = model.calculate_slippage(large_order, order_book)
        
        assert large_slippage > small_slippage


class TestLatencyModel:
    """Test cases for LatencyModel."""
    
    def test_constant_latency_model(self):
        """Test ConstantLatencyModel."""
        model = ConstantLatencyModel(
            order_latency_ms=50.0,
            market_data_latency_ms=10.0
        )
        
        order = Order(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        order_latency = model.get_order_latency(order)
        market_data_latency = model.get_market_data_latency()
        
        assert order_latency == timedelta(milliseconds=50)
        assert market_data_latency == timedelta(milliseconds=10)
    
    def test_normal_distribution_latency_model(self):
        """Test NormalDistributionLatencyModel."""
        model = NormalDistributionLatencyModel(
            order_latency_mean_ms=50.0,
            order_latency_std_ms=10.0,
            min_latency_ms=10.0,
            max_latency_ms=200.0
        )
        
        order = Order(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Test multiple samples
        latencies = []
        for _ in range(100):
            latency = model.get_order_latency(order)
            latencies.append(latency.total_seconds() * 1000)
        
        # Should be within bounds
        assert all(10.0 <= lat <= 200.0 for lat in latencies)
        
        # Should have reasonable mean (within 2 standard deviations)
        mean_latency = np.mean(latencies)
        assert 30.0 <= mean_latency <= 70.0


class TestFeeModel:
    """Test cases for FeeModel."""
    
    def test_tiered_fee_model(self):
        """Test TieredFeeModel."""
        model = TieredFeeModel(
            maker_fee_bps=1.0,
            taker_fee_bps=1.5
        )
        
        fill = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow()
        )
        
        fee = model.calculate_fee(fill)
        
        # Should be 1.5 bps of notional value
        expected_fee = 1.0 * 50000.0 * 0.015 / 100
        assert abs(fee - expected_fee) < 0.01
    
    def test_tiered_fee_model_with_volume_tiers(self):
        """Test TieredFeeModel with volume tiers."""
        volume_tiers = {
            100000.0: (0.75, 1.0),  # Reduced fees after $100k volume
            500000.0: (0.5, 0.75)   # Further reduced after $500k
        }
        
        model = TieredFeeModel(
            maker_fee_bps=1.0,
            taker_fee_bps=1.5,
            volume_tiers=volume_tiers
        )
        
        # First fill - base tier
        fill1 = Fill(
            order_id="TEST_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow()
        )
        
        fee1 = model.calculate_fee(fill1)
        expected_fee1 = 1.0 * 50000.0 * 0.015 / 100
        assert abs(fee1 - expected_fee1) < 0.01
        
        # Large fill to trigger tier change
        fill2 = Fill(
            order_id="TEST_002",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=2.0,
            price=50000.0,
            fee=0.0,
            timestamp=datetime.utcnow()
        )
        
        fee2 = model.calculate_fee(fill2)
        # Should use reduced fee rate (1.0% instead of 1.5%)
        expected_fee2 = 2.0 * 50000.0 * 0.01 / 100
        assert abs(fee2 - expected_fee2) < 0.01


class TestBacktestConfig:
    """Test cases for BacktestConfig."""
    
    def test_config_creation(self):
        """Test creating BacktestConfig."""
        config = BacktestConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            initial_capital=100000.0,
            symbols=['BTCUSDT', 'ETHUSDT']
        )
        
        assert config.start_time == datetime(2024, 1, 1)
        assert config.end_time == datetime(2024, 1, 2)
        assert config.initial_capital == 100000.0
        assert config.symbols == ['BTCUSDT', 'ETHUSDT']
        assert config.market_impact_factor == 1.0
        assert config.enable_short_selling is True
        assert config.risk_free_rate == 0.02
    
    def test_config_with_custom_models(self):
        """Test BacktestConfig with custom models."""
        slippage_model = LinearSlippageModel(base_slippage_bps=2.0)
        latency_model = ConstantLatencyModel(order_latency_ms=100.0)
        fee_model = TieredFeeModel(taker_fee_bps=2.0)
        
        config = BacktestConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            initial_capital=50000.0,
            symbols=['BTCUSDT'],
            slippage_model=slippage_model,
            latency_model=latency_model,
            fee_model=fee_model,
            market_impact_factor=1.5
        )
        
        assert config.slippage_model == slippage_model
        assert config.latency_model == latency_model
        assert config.fee_model == fee_model
        assert config.market_impact_factor == 1.5
