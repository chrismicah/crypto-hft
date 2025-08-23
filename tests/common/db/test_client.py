"""Unit tests for database client."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from common.db.client import DatabaseClient
from common.db.models import Order, Trade, TradeOrder, PortfolioPnL, SystemEvent


class TestDatabaseClient:
    """Test cases for DatabaseClient class."""
    
    @pytest.fixture
    def temp_db_client(self):
        """Create a temporary database client for testing."""
        # Create temporary database file
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        db_url = f"sqlite:///{temp_db.name}"
        client = DatabaseClient(database_url=db_url)
        client.create_tables()
        
        yield client
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
    
    def test_client_initialization(self):
        """Test database client initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
            db_url = f"sqlite:///{temp_db.name}"
            client = DatabaseClient(database_url=db_url)
            
            assert client.database_url == db_url
            assert client.engine is not None
            assert client.SessionLocal is not None
    
    def test_create_tables(self, temp_db_client):
        """Test table creation."""
        client = temp_db_client
        
        # Tables should be created without error
        # We can verify by trying to query them
        with client.get_session() as session:
            # Should not raise an error
            session.query(Order).count()
            session.query(Trade).count()
            session.query(TradeOrder).count()
            session.query(PortfolioPnL).count()
            session.query(SystemEvent).count()
    
    def test_session_context_manager(self, temp_db_client):
        """Test session context manager."""
        client = temp_db_client
        
        # Test successful transaction
        with client.get_session() as session:
            order = Order(
                order_id='test_order',
                symbol='BTCUSDT',
                side='buy',
                order_type='market',
                amount=0.1,
                status='open'
            )
            session.add(order)
            # Should commit automatically
        
        # Verify order was saved
        with client.get_session() as session:
            saved_order = session.query(Order).filter_by(order_id='test_order').first()
            assert saved_order is not None
            assert saved_order.order_id == 'test_order'
    
    def test_session_rollback_on_error(self, temp_db_client):
        """Test that session rolls back on error."""
        client = temp_db_client
        
        try:
            with client.get_session() as session:
                order = Order(
                    order_id='test_order',
                    symbol='BTCUSDT',
                    side='buy',
                    order_type='market',
                    amount=0.1,
                    status='open'
                )
                session.add(order)
                session.flush()  # Force the insert
                
                # Cause an error
                raise Exception("Test error")
        except Exception:
            pass  # Expected
        
        # Verify order was not saved due to rollback
        with client.get_session() as session:
            saved_order = session.query(Order).filter_by(order_id='test_order').first()
            assert saved_order is None
    
    def test_write_order_new(self, temp_db_client):
        """Test writing a new order."""
        client = temp_db_client
        
        order = client.write_order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,
            status='filled',
            price=50000.0,
            filled=0.1,
            cost=5000.0,
            fee=5.0,
            fee_currency='USDT',
            strategy_id='pairs_trading',
            pair_id='BTCETH'
        )
        
        assert order is not None
        assert order.order_id == 'test_order_123'
        assert order.symbol == 'BTCUSDT'
        assert order.side == 'buy'
        assert order.amount == 0.1
        assert order.status == 'filled'
        assert order.strategy_id == 'pairs_trading'
        
        # Verify it was saved to database
        with client.get_session() as session:
            saved_order = session.query(Order).filter_by(order_id='test_order_123').first()
            assert saved_order is not None
            assert saved_order.symbol == 'BTCUSDT'
    
    def test_write_order_update_existing(self, temp_db_client):
        """Test updating an existing order."""
        client = temp_db_client
        
        # Create initial order
        order1 = client.write_order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side='buy',
            order_type='limit',
            amount=0.1,
            status='open',
            price=49000.0
        )
        
        assert order1.status == 'open'
        assert order1.filled == 0.0
        
        # Update the order
        order2 = client.write_order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            side='buy',
            order_type='limit',
            amount=0.1,
            status='filled',
            price=49000.0,
            filled=0.1,
            cost=4900.0
        )
        
        assert order2.status == 'filled'
        assert order2.filled == 0.1
        assert order2.cost == 4900.0
        
        # Verify only one order exists in database
        with client.get_session() as session:
            order_count = session.query(Order).filter_by(order_id='test_order_123').count()
            assert order_count == 1
    
    def test_write_trade(self, temp_db_client):
        """Test writing a trade."""
        client = temp_db_client
        
        entry_time = datetime.utcnow()
        
        trade = client.write_trade(
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
            entry_z_score=-2.5
        )
        
        assert trade is not None
        assert trade.trade_id == 'trade_123'
        assert trade.pair_id == 'BTCETH'
        assert trade.side == 'long'
        assert trade.status == 'open'
        assert trade.entry_time == entry_time
        assert trade.notional_value == 5000.0
        
        # Verify it was saved to database
        with client.get_session() as session:
            saved_trade = session.query(Trade).filter_by(trade_id='trade_123').first()
            assert saved_trade is not None
            assert saved_trade.pair_id == 'BTCETH'
    
    def test_close_trade(self, temp_db_client):
        """Test closing a trade."""
        client = temp_db_client
        
        # Create a trade first
        entry_time = datetime.utcnow()
        trade = client.write_trade(
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
            notional_value=5000.0
        )
        
        assert trade.status == 'open'
        
        # Close the trade
        exit_time = datetime.utcnow()
        closed_trade = client.close_trade(
            trade_id='trade_123',
            exit_time=exit_time,
            exit_price_asset1=51000.0,
            exit_price_asset2=3050.0,
            exit_spread=0.03,
            realized_pnl=150.0,
            close_reason='profit_target',
            exit_z_score=-0.5,
            total_fees=10.0
        )
        
        assert closed_trade is not None
        assert closed_trade.status == 'closed'
        assert closed_trade.exit_time == exit_time
        assert closed_trade.realized_pnl == 150.0
        assert closed_trade.close_reason == 'profit_target'
        assert closed_trade.total_fees == 10.0
    
    def test_close_trade_not_found(self, temp_db_client):
        """Test closing a non-existent trade."""
        client = temp_db_client
        
        result = client.close_trade(
            trade_id='nonexistent_trade',
            exit_time=datetime.utcnow(),
            exit_price_asset1=50000.0,
            exit_price_asset2=3000.0,
            exit_spread=0.05,
            realized_pnl=0.0
        )
        
        assert result is None
    
    def test_update_trade_pnl(self, temp_db_client):
        """Test updating trade P&L."""
        client = temp_db_client
        
        # Create a trade first
        trade = client.write_trade(
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
        
        # Update P&L
        updated_trade = client.update_trade_pnl(
            trade_id='trade_123',
            unrealized_pnl=75.0,
            max_favorable_excursion=100.0,
            max_adverse_excursion=25.0
        )
        
        assert updated_trade is not None
        assert updated_trade.unrealized_pnl == 75.0
        assert updated_trade.max_favorable_excursion == 100.0
        assert updated_trade.max_adverse_excursion == 25.0
        
        # Update again with higher values
        updated_trade2 = client.update_trade_pnl(
            trade_id='trade_123',
            unrealized_pnl=50.0,  # Lower than before
            max_favorable_excursion=150.0,  # Higher than before
            max_adverse_excursion=50.0  # Higher than before
        )
        
        assert updated_trade2.unrealized_pnl == 50.0
        assert updated_trade2.max_favorable_excursion == 150.0  # Should be max
        assert updated_trade2.max_adverse_excursion == 50.0  # Should be max
    
    def test_link_trade_order(self, temp_db_client):
        """Test linking a trade to an order."""
        client = temp_db_client
        
        # Create trade and order
        trade = client.write_trade(
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
        
        order = client.write_order(
            order_id='order_123',
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,
            status='filled'
        )
        
        # Link them
        trade_order = client.link_trade_order(
            trade_id='trade_123',
            order_id='order_123',
            order_role='entry_asset1'
        )
        
        assert trade_order is not None
        assert trade_order.trade_id == trade.id
        assert trade_order.order_id == order.id
        assert trade_order.order_role == 'entry_asset1'
    
    def test_write_portfolio_pnl(self, temp_db_client):
        """Test writing portfolio P&L."""
        client = temp_db_client
        
        timestamp = datetime.utcnow()
        
        portfolio_pnl = client.write_portfolio_pnl(
            timestamp=timestamp,
            total_value=10000.0,
            cash_balance=5000.0,
            unrealized_pnl=150.0,
            realized_pnl=250.0,
            open_positions=2,
            daily_pnl=75.0,
            strategy_pnl={'pairs_trading': 150.0},
            pair_pnl={'BTCETH': 75.0, 'BTCADA': 75.0}
        )
        
        assert portfolio_pnl is not None
        assert portfolio_pnl.timestamp == timestamp
        assert portfolio_pnl.total_value == 10000.0
        assert portfolio_pnl.unrealized_pnl == 150.0
        assert portfolio_pnl.open_positions == 2
    
    def test_write_system_event(self, temp_db_client):
        """Test writing system event."""
        client = temp_db_client
        
        event = client.write_system_event(
            event_type='order_placed',
            service_name='execution-service',
            severity='info',
            message='Order placed successfully',
            details={'order_id': '123', 'symbol': 'BTCUSDT'},
            pair_id='BTCETH',
            trade_id='trade_123',
            order_id='order_123'
        )
        
        assert event is not None
        assert event.event_type == 'order_placed'
        assert event.service_name == 'execution-service'
        assert event.severity == 'info'
        assert event.message == 'Order placed successfully'
        assert event.pair_id == 'BTCETH'
    
    def test_get_open_trades(self, temp_db_client):
        """Test getting open trades."""
        client = temp_db_client
        
        # Create some trades
        client.write_trade(
            trade_id='trade_1',
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
        
        client.write_trade(
            trade_id='trade_2',
            pair_id='BTCADA',
            asset1_symbol='BTC',
            asset2_symbol='ADA',
            side='short',
            strategy_id='pairs_trading',
            entry_time=datetime.utcnow(),
            entry_price_asset1=50000.0,
            entry_price_asset2=0.5,
            entry_spread=-0.02,
            hedge_ratio=100000.0,
            quantity_asset1=0.05,
            quantity_asset2=5000.0,
            notional_value=2500.0
        )
        
        # Close one trade
        client.close_trade(
            trade_id='trade_2',
            exit_time=datetime.utcnow(),
            exit_price_asset1=49000.0,
            exit_price_asset2=0.52,
            exit_spread=0.01,
            realized_pnl=100.0
        )
        
        # Get all open trades
        open_trades = client.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].trade_id == 'trade_1'
        
        # Get open trades for specific pair
        btceth_trades = client.get_open_trades(pair_id='BTCETH')
        assert len(btceth_trades) == 1
        assert btceth_trades[0].pair_id == 'BTCETH'
        
        btcada_trades = client.get_open_trades(pair_id='BTCADA')
        assert len(btcada_trades) == 0  # Closed
    
    def test_get_recent_portfolio_pnl(self, temp_db_client):
        """Test getting recent portfolio P&L."""
        client = temp_db_client
        
        # Create some portfolio P&L records
        now = datetime.utcnow()
        
        # Recent record (within 24 hours)
        client.write_portfolio_pnl(
            timestamp=now - timedelta(hours=1),
            total_value=10000.0,
            cash_balance=5000.0
        )
        
        # Old record (beyond 24 hours)
        client.write_portfolio_pnl(
            timestamp=now - timedelta(hours=25),
            total_value=9500.0,
            cash_balance=4500.0
        )
        
        # Get recent records (default 24 hours)
        recent_pnl = client.get_recent_portfolio_pnl()
        assert len(recent_pnl) == 1
        assert recent_pnl[0].total_value == 10000.0
        
        # Get records for longer period
        all_pnl = client.get_recent_portfolio_pnl(hours=48)
        assert len(all_pnl) == 2
    
    def test_get_trade_performance_summary(self, temp_db_client):
        """Test getting trade performance summary."""
        client = temp_db_client
        
        # Create and close some trades
        now = datetime.utcnow()
        
        # Winning trade
        client.write_trade(
            trade_id='trade_win',
            pair_id='BTCETH',
            asset1_symbol='BTC',
            asset2_symbol='ETH',
            side='long',
            strategy_id='pairs_trading',
            entry_time=now - timedelta(days=1),
            entry_price_asset1=50000.0,
            entry_price_asset2=3000.0,
            entry_spread=0.05,
            hedge_ratio=16.67,
            quantity_asset1=0.1,
            quantity_asset2=1.67,
            notional_value=5000.0
        )
        
        client.close_trade(
            trade_id='trade_win',
            exit_time=now - timedelta(hours=12),
            exit_price_asset1=51000.0,
            exit_price_asset2=3050.0,
            exit_spread=0.03,
            realized_pnl=150.0
        )
        
        # Losing trade
        client.write_trade(
            trade_id='trade_loss',
            pair_id='BTCADA',
            asset1_symbol='BTC',
            asset2_symbol='ADA',
            side='short',
            strategy_id='pairs_trading',
            entry_time=now - timedelta(days=2),
            entry_price_asset1=50000.0,
            entry_price_asset2=0.5,
            entry_spread=-0.02,
            hedge_ratio=100000.0,
            quantity_asset1=0.05,
            quantity_asset2=5000.0,
            notional_value=2500.0
        )
        
        client.close_trade(
            trade_id='trade_loss',
            exit_time=now - timedelta(hours=6),
            exit_price_asset1=50500.0,
            exit_price_asset2=0.48,
            exit_spread=-0.05,
            realized_pnl=-75.0
        )
        
        # Get performance summary
        summary = client.get_trade_performance_summary(days=7)
        
        assert summary['total_trades'] == 2
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == 0.5
        assert summary['total_pnl'] == 75.0  # 150 - 75
        assert summary['average_pnl'] == 37.5  # 75 / 2
        assert summary['max_win'] == 150.0
        assert summary['max_loss'] == -75.0
    
    def test_cleanup_old_data(self, temp_db_client):
        """Test cleaning up old data."""
        client = temp_db_client
        
        now = datetime.utcnow()
        
        # Create old system events
        client.write_system_event(
            event_type='old_event',
            service_name='test-service',
            severity='info',
            message='Old event'
        )
        
        # Manually set timestamp to old date
        with client.get_session() as session:
            old_event = session.query(SystemEvent).filter_by(event_type='old_event').first()
            old_event.timestamp = now - timedelta(days=100)
            session.commit()
        
        # Create recent event
        client.write_system_event(
            event_type='recent_event',
            service_name='test-service',
            severity='info',
            message='Recent event'
        )
        
        # Verify both events exist
        with client.get_session() as session:
            event_count = session.query(SystemEvent).count()
            assert event_count == 2
        
        # Cleanup old data (90 days)
        client.cleanup_old_data(days=90)
        
        # Verify only recent event remains
        with client.get_session() as session:
            remaining_events = session.query(SystemEvent).all()
            assert len(remaining_events) == 1
            assert remaining_events[0].event_type == 'recent_event'
    
    @patch('common.db.client.logger')
    def test_error_handling(self, mock_logger, temp_db_client):
        """Test error handling in database operations."""
        client = temp_db_client
        
        # Test write_order with database error
        with patch.object(client, 'get_session') as mock_session:
            mock_session.side_effect = Exception("Database error")
            
            result = client.write_order(
                order_id='test_order',
                symbol='BTCUSDT',
                side='buy',
                order_type='market',
                amount=0.1,
                status='open'
            )
            
            assert result is None
            mock_logger.error.assert_called()
    
    def test_metadata_handling(self, temp_db_client):
        """Test handling of metadata fields."""
        client = temp_db_client
        
        # Test order with metadata
        metadata = {'signal_confidence': 0.85, 'volatility_regime': 'normal'}
        
        order = client.write_order(
            order_id='test_order',
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,
            status='filled',
            metadata=metadata
        )
        
        assert order is not None
        
        # Verify metadata was stored and can be retrieved
        with client.get_session() as session:
            saved_order = session.query(Order).filter_by(order_id='test_order').first()
            order_dict = saved_order.to_dict()
            assert order_dict['metadata'] == metadata
