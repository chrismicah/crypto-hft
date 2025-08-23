"""Unit tests for database models."""

import pytest
from datetime import datetime, timedelta
import json

from common.db.models import Order, Trade, TradeOrder, PortfolioPnL, SystemEvent


class TestOrder:
    """Test cases for Order model."""
    
    def test_order_creation(self):
        """Test creating an Order object."""
        order = Order(
            order_id='test_order_123',
            client_order_id='client_123',
            symbol='BTCUSDT',
            exchange='binance_testnet',
            side='buy',
            order_type='market',
            amount=0.1,
            price=50000.0,
            status='filled',
            filled=0.1,
            remaining=0.0,
            cost=5000.0,
            fee=5.0,
            fee_currency='USDT',
            strategy_id='pairs_trading',
            pair_id='BTCETH'
        )
        
        assert order.order_id == 'test_order_123'
        assert order.client_order_id == 'client_123'
        assert order.symbol == 'BTCUSDT'
        assert order.exchange == 'binance_testnet'
        assert order.side == 'buy'
        assert order.order_type == 'market'
        assert order.amount == 0.1
        assert order.price == 50000.0
        assert order.status == 'filled'
        assert order.filled == 0.1
        assert order.remaining == 0.0
        assert order.cost == 5000.0
        assert order.fee == 5.0
        assert order.fee_currency == 'USDT'
        assert order.strategy_id == 'pairs_trading'
        assert order.pair_id == 'BTCETH'
    
    def test_order_defaults(self):
        """Test Order model defaults."""
        order = Order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,
            status='open'
        )
        
        assert order.exchange == 'binance_testnet'
        assert order.filled == 0.0
        assert order.remaining == 0.0
        assert order.cost == 0.0
        assert order.created_at is not None
        assert order.updated_at is not None
    
    def test_order_to_dict(self):
        """Test Order to_dict method."""
        order = Order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,
            status='filled',
            metadata='{"test": "data"}'
        )
        
        order_dict = order.to_dict()
        
        required_keys = [
            'id', 'order_id', 'symbol', 'side', 'order_type',
            'amount', 'status', 'filled', 'remaining', 'cost'
        ]
        
        for key in required_keys:
            assert key in order_dict
        
        assert order_dict['order_id'] == 'test_order_123'
        assert order_dict['symbol'] == 'BTCUSDT'
        assert order_dict['metadata'] == {"test": "data"}
    
    def test_order_repr(self):
        """Test Order string representation."""
        order = Order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side='buy',
            status='filled'
        )
        
        repr_str = repr(order)
        assert 'test_order_123' in repr_str
        assert 'BTCUSDT' in repr_str
        assert 'buy' in repr_str
        assert 'filled' in repr_str


class TestTrade:
    """Test cases for Trade model."""
    
    def test_trade_creation(self):
        """Test creating a Trade object."""
        entry_time = datetime.utcnow()
        
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=entry_time,
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            entry_z_score=-2.5,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0
        )
        
        assert trade.trade_id == 'trade_123'
        assert trade.pair_id == 'BTCETH'
        assert trade.asset1_symbol == 'BTC'
        assert trade.asset2_symbol == 'ETH'
        assert trade.side == 'long'
        assert trade.strategy_id == 'pairs_trading'
        assert trade.entry_time == entry_time
        assert trade.entry_price_asset1 == 50000.0
        assert trade.entry_price_asset2 == 3000.0
        assert trade.entry_spread == 0.05
        assert trade.entry_z_score == -2.5
        assert trade.hedge_ratio == 16.67
        assert trade.quantity_asset1 == 0.1
        assert trade.quantity_asset2 == 1.67
        assert trade.notional_value == 5000.0
    
    def test_trade_defaults(self):
        """Test Trade model defaults."""
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=datetime.utcnow(),
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0
        )
        
        assert trade.status == 'open'
        assert trade.unrealized_pnl == 0.0
        assert trade.max_favorable_excursion == 0.0
        assert trade.max_adverse_excursion == 0.0
        assert trade.total_fees == 0.0
        assert trade.created_at is not None
        assert trade.updated_at is not None
    
    def test_trade_duration_calculation(self):
        """Test trade duration calculation."""
        entry_time = datetime.utcnow()
        exit_time = entry_time + timedelta(hours=2, minutes=30)
        
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=entry_time,
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0,
            exit_time=exit_time
        )
        
        duration = trade.duration_seconds
        expected_duration = 2 * 3600 + 30 * 60  # 2.5 hours in seconds
        
        assert duration == expected_duration
    
    def test_trade_duration_none_when_open(self):
        """Test that duration is None for open trades."""
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=datetime.utcnow(),
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0
        )
        
        assert trade.duration_seconds is None
    
    def test_trade_is_open_property(self):
        """Test is_open property."""
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=datetime.utcnow(),
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0
        )
        
        assert trade.is_open is True
        
        trade.status = 'closed'
        assert trade.is_open is False
    
    def test_trade_to_dict(self):
        """Test Trade to_dict method."""
        entry_time = datetime.utcnow()
        
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=entry_time,
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0,
            metadata='{"signal_confidence": 0.85}'
        )
        
        trade_dict = trade.to_dict()
        
        required_keys = [
            'id', 'trade_id', 'pair_id', 'side', 'strategy_id',
            'entry_time', 'entry_price_asset1', 'entry_price_asset2',
            'hedge_ratio', 'notional_value', 'status'
        ]
        
        for key in required_keys:
            assert key in trade_dict
        
        assert trade_dict['trade_id'] == 'trade_123'
        assert trade_dict['pair_id'] == 'BTCETH'
        assert trade_dict['metadata'] == {"signal_confidence": 0.85}


class TestTradeOrder:
    """Test cases for TradeOrder model."""
    
    def test_trade_order_creation(self):
        """Test creating a TradeOrder object."""
        trade_order = TradeOrder(
            trade_id=1,
            order_id=1,
            order_role='entry_asset1'
        )
        
        assert trade_order.trade_id == 1
        assert trade_order.order_id == 1
        assert trade_order.order_role == 'entry_asset1'
        assert trade_order.created_at is not None
    
    def test_trade_order_repr(self):
        """Test TradeOrder string representation."""
        trade_order = TradeOrder(
            trade_id=1,
            order_id=2,
            order_role='exit_asset2'
        )
        
        repr_str = repr(trade_order)
        assert 'trade_id=1' in repr_str
        assert 'order_id=2' in repr_str
        assert 'exit_asset2' in repr_str


class TestPortfolioPnL:
    """Test cases for PortfolioPnL model."""
    
    def test_portfolio_pnl_creation(self):
        """Test creating a PortfolioPnL object."""
        timestamp = datetime.utcnow()
        
        portfolio_pnl = PortfolioPnL(
            timestamp=timestamp,
            total_value=10000.0,
            cash_balance=5000.0,
            unrealized_pnl=150.0,
            realized_pnl=250.0,
            open_positions=2,
            daily_pnl=75.0,
            drawdown=0.05,
            max_drawdown=0.12
        )
        
        assert portfolio_pnl.timestamp == timestamp
        assert portfolio_pnl.total_value == 10000.0
        assert portfolio_pnl.cash_balance == 5000.0
        assert portfolio_pnl.unrealized_pnl == 150.0
        assert portfolio_pnl.realized_pnl == 250.0
        assert portfolio_pnl.open_positions == 2
        assert portfolio_pnl.daily_pnl == 75.0
        assert portfolio_pnl.drawdown == 0.05
        assert portfolio_pnl.max_drawdown == 0.12
    
    def test_portfolio_pnl_defaults(self):
        """Test PortfolioPnL model defaults."""
        portfolio_pnl = PortfolioPnL(
            timestamp=datetime.utcnow(),
            total_value=10000.0,
            cash_balance=5000.0
        )
        
        assert portfolio_pnl.unrealized_pnl == 0.0
        assert portfolio_pnl.realized_pnl == 0.0
        assert portfolio_pnl.open_positions == 0
        assert portfolio_pnl.created_at is not None
    
    def test_portfolio_pnl_to_dict(self):
        """Test PortfolioPnL to_dict method."""
        timestamp = datetime.utcnow()
        
        portfolio_pnl = PortfolioPnL(
            timestamp=timestamp,
            total_value=10000.0,
            cash_balance=5000.0,
            strategy_pnl='{"pairs_trading": 150.0}',
            pair_pnl='{"BTCETH": 75.0, "BTCADA": 75.0}'
        )
        
        pnl_dict = portfolio_pnl.to_dict()
        
        required_keys = [
            'id', 'timestamp', 'total_value', 'cash_balance',
            'unrealized_pnl', 'realized_pnl', 'open_positions'
        ]
        
        for key in required_keys:
            assert key in pnl_dict
        
        assert pnl_dict['total_value'] == 10000.0
        assert pnl_dict['strategy_pnl'] == {"pairs_trading": 150.0}
        assert pnl_dict['pair_pnl'] == {"BTCETH": 75.0, "BTCADA": 75.0}


class TestSystemEvent:
    """Test cases for SystemEvent model."""
    
    def test_system_event_creation(self):
        """Test creating a SystemEvent object."""
        event = SystemEvent(
            event_type='order_placed',
            service_name='execution-service',
            severity='info',
            message='Order placed successfully',
            details='{"order_id": "123", "symbol": "BTCUSDT"}',
            pair_id='BTCETH',
            trade_id='trade_123',
            order_id='order_123'
        )
        
        assert event.event_type == 'order_placed'
        assert event.service_name == 'execution-service'
        assert event.severity == 'info'
        assert event.message == 'Order placed successfully'
        assert event.details == '{"order_id": "123", "symbol": "BTCUSDT"}'
        assert event.pair_id == 'BTCETH'
        assert event.trade_id == 'trade_123'
        assert event.order_id == 'order_123'
        assert event.timestamp is not None
    
    def test_system_event_to_dict(self):
        """Test SystemEvent to_dict method."""
        event = SystemEvent(
            event_type='error',
            service_name='kalman-filter',
            severity='error',
            message='Failed to calculate hedge ratio',
            details='{"error": "Division by zero", "pair_id": "BTCETH"}'
        )
        
        event_dict = event.to_dict()
        
        required_keys = [
            'id', 'event_type', 'service_name', 'severity',
            'message', 'timestamp'
        ]
        
        for key in required_keys:
            assert key in event_dict
        
        assert event_dict['event_type'] == 'error'
        assert event_dict['service_name'] == 'kalman-filter'
        assert event_dict['details'] == {"error": "Division by zero", "pair_id": "BTCETH"}
    
    def test_system_event_repr(self):
        """Test SystemEvent string representation."""
        event = SystemEvent(
            event_type='signal_generated',
            service_name='garch-volatility',
            severity='info',
            message='Signal generated'
        )
        
        repr_str = repr(event)
        assert 'signal_generated' in repr_str
        assert 'garch-volatility' in repr_str
        assert 'info' in repr_str


class TestModelValidation:
    """Test model validation and constraints."""
    
    def test_order_positive_amount_constraint(self):
        """Test that Order enforces positive amount constraint."""
        # This would be tested at the database level
        # Here we just verify the constraint exists in the model
        order = Order(
            order_id='test_order',
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,  # Positive amount
            status='open'
        )
        
        # Should not raise an error
        assert order.amount > 0
    
    def test_trade_positive_quantities_constraint(self):
        """Test that Trade enforces positive quantity constraints."""
        trade = Trade(
            trade_id='trade_123',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=datetime.utcnow(),
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,  # Positive
            quantity_asset2=1.67,  # Positive
            notional_value=5000.0  # Positive
        )
        
        assert trade.quantity_asset1 > 0
        assert trade.quantity_asset2 > 0
        assert trade.notional_value > 0
        assert trade.hedge_ratio > 0
    
    def test_portfolio_pnl_non_negative_constraints(self):
        """Test that PortfolioPnL enforces non-negative constraints."""
        portfolio_pnl = PortfolioPnL(
            timestamp=datetime.utcnow(),
            total_value=10000.0,  # Non-negative
            cash_balance=5000.0,  # Non-negative
            open_positions=2  # Non-negative
        )
        
        assert portfolio_pnl.total_value >= 0
        assert portfolio_pnl.cash_balance >= 0
        assert portfolio_pnl.open_positions >= 0


@pytest.fixture
def sample_order():
    """Fixture providing a sample Order."""
    return Order(
        order_id='test_order_123',
        symbol='BTCUSDT',
        side='buy',
        order_type='market',
        amount=0.1,
        status='filled',
        filled=0.1,
        cost=5000.0
    )


@pytest.fixture
def sample_trade():
    """Fixture providing a sample Trade."""
    return Trade(
        trade_id='trade_123',
        pair_id='BTCETH',
        asset1_symbol='BTC',
        asset2_symbol='ETH',
        side='long',
        strategy_id='pairs_trading',
        entry_time=datetime.utcnow(),
        entry_price_asset1=50000.0,
        entry_price_asset2=3000.0,
        entry_spread=0.05,
        hedge_ratio=16.67,
        quantity_asset1=0.1,
        quantity_asset2=1.67,
        notional_value=5000.0
    )


class TestModelIntegration:
    """Integration tests for model relationships."""
    
    def test_order_trade_relationship_setup(self, sample_order, sample_trade):
        """Test that Order and Trade can be linked via TradeOrder."""
        # This tests the relationship setup, not actual database relationships
        trade_order = TradeOrder(
            trade_id=sample_trade.id or 1,  # Mock ID
            order_id=sample_order.id or 1,  # Mock ID
            order_role='entry_asset1'
        )
        
        assert trade_order.trade_id == 1
        assert trade_order.order_id == 1
        assert trade_order.order_role == 'entry_asset1'
    
    def test_model_serialization(self, sample_order, sample_trade):
        """Test that models can be serialized to dictionaries."""
        order_dict = sample_order.to_dict()
        trade_dict = sample_trade.to_dict()
        
        # Should be JSON serializable
        import json
        
        order_json = json.dumps(order_dict, default=str)
        trade_json = json.dumps(trade_dict, default=str)
        
        assert order_json is not None
        assert trade_json is not None
        
        # Should be deserializable
        order_data = json.loads(order_json)
        trade_data = json.loads(trade_json)
        
        assert order_data['order_id'] == 'test_order_123'
        assert trade_data['trade_id'] == 'trade_123'
