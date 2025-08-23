"""Unit tests for portfolio tracking and PnL calculation."""

import pytest
import numpy as np
from datetime import datetime, timedelta, date
from threading import Thread
import time

from services.execution.portfolio import (
    Position, PositionSide, DailyPnL, PortfolioTracker
)


class TestPosition:
    """Test cases for Position class."""
    
    def test_position_creation(self):
        """Test position creation."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        assert position.symbol == "BTCUSDT"
        assert position.quantity == 0.0
        assert position.average_price == 0.0
        assert position.market_price == 50000.0
        assert position.side == PositionSide.FLAT
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0
    
    def test_position_long_entry(self):
        """Test opening a long position."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        realized_pnl = position.add_trade(1.0, 50000.0)
        
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
        assert position.side == PositionSide.LONG
        assert position.market_value == 50000.0
        assert position.notional_value == 50000.0
        assert realized_pnl == 0.0  # No realized P&L on opening
        assert position.unrealized_pnl == 0.0  # No unrealized P&L at entry price
    
    def test_position_short_entry(self):
        """Test opening a short position."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        realized_pnl = position.add_trade(-1.0, 50000.0)
        
        assert position.quantity == -1.0
        assert position.average_price == 50000.0
        assert position.side == PositionSide.SHORT
        assert position.market_value == -50000.0
        assert position.notional_value == 50000.0
        assert realized_pnl == 0.0
    
    def test_position_averaging_long(self):
        """Test averaging into a long position."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        # First trade
        position.add_trade(1.0, 50000.0)
        
        # Second trade at higher price
        position.add_trade(1.0, 52000.0)
        
        assert position.quantity == 2.0
        assert position.average_price == 51000.0  # (50000 + 52000) / 2
        assert position.side == PositionSide.LONG
    
    def test_position_partial_close(self):
        """Test partially closing a position."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        # Open position
        position.add_trade(2.0, 50000.0)
        
        # Partial close at higher price
        realized_pnl = position.add_trade(-1.0, 52000.0)
        
        assert position.quantity == 1.0
        assert position.average_price == 50000.0  # Average price unchanged
        assert realized_pnl == 2000.0  # (52000 - 50000) * 1
        assert position.realized_pnl == 2000.0
    
    def test_position_full_close(self):
        """Test fully closing a position."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        # Open position
        position.add_trade(1.0, 50000.0)
        
        # Full close at higher price
        realized_pnl = position.add_trade(-1.0, 52000.0)
        
        assert position.quantity == 0.0
        assert position.side == PositionSide.FLAT
        assert realized_pnl == 2000.0
        assert position.realized_pnl == 2000.0
    
    def test_position_reverse(self):
        """Test reversing a position."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        # Open long position
        position.add_trade(1.0, 50000.0)
        
        # Reverse to short
        realized_pnl = position.add_trade(-2.0, 52000.0)
        
        assert position.quantity == -1.0  # Now short
        assert position.average_price == 52000.0  # New entry price
        assert position.side == PositionSide.SHORT
        assert realized_pnl == 2000.0  # Profit from closing long
        assert position.realized_pnl == 2000.0
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        
        # Open long position
        position.add_trade(1.0, 50000.0)
        
        # Update market price
        position.update_market_price(52000.0)
        
        assert position.unrealized_pnl == 2000.0  # (52000 - 50000) * 1
        assert position.total_pnl == 2000.0
    
    def test_position_to_dict(self):
        """Test position serialization."""
        position = Position(symbol="BTCUSDT", market_price=50000.0)
        position.add_trade(1.0, 50000.0)
        position.update_market_price(52000.0)
        
        position_dict = position.to_dict()
        
        assert position_dict['symbol'] == 'BTCUSDT'
        assert position_dict['quantity'] == 1.0
        assert position_dict['average_price'] == 50000.0
        assert position_dict['market_price'] == 52000.0
        assert position_dict['side'] == 'LONG'
        assert position_dict['unrealized_pnl'] == 2000.0


class TestDailyPnL:
    """Test cases for DailyPnL class."""
    
    def test_daily_pnl_creation(self):
        """Test daily P&L creation."""
        today = date.today()
        daily_pnl = DailyPnL(date=today, starting_value=100000.0)
        
        assert daily_pnl.date == today
        assert daily_pnl.starting_value == 100000.0
        assert daily_pnl.total_pnl == 0.0
        assert daily_pnl.current_value == 100000.0
    
    def test_daily_pnl_calculation(self):
        """Test daily P&L calculation."""
        daily_pnl = DailyPnL(
            date=date.today(),
            starting_value=100000.0,
            realized_pnl=1000.0,
            unrealized_pnl=500.0,
            fees=50.0
        )
        
        assert daily_pnl.total_pnl == 1450.0  # 1000 + 500 - 50
        assert daily_pnl.current_value == 101450.0
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        daily_pnl = DailyPnL(date=date.today(), starting_value=100000.0)
        
        # Simulate price movements
        daily_pnl.update_drawdown(105000.0)  # New high
        assert daily_pnl.high_water_mark == 105000.0
        assert daily_pnl.max_drawdown == 0.0
        
        daily_pnl.update_drawdown(95000.0)   # Drawdown
        expected_drawdown = (105000.0 - 95000.0) / 105000.0
        assert abs(daily_pnl.max_drawdown - expected_drawdown) < 1e-6
        
        daily_pnl.update_drawdown(102000.0)  # Recovery
        assert daily_pnl.max_drawdown == expected_drawdown  # Max drawdown preserved


class TestPortfolioTracker:
    """Test cases for PortfolioTracker class."""
    
    @pytest.fixture
    def portfolio(self):
        """Create portfolio tracker instance."""
        return PortfolioTracker(initial_capital=100000.0)
    
    def test_portfolio_initialization(self, portfolio):
        """Test portfolio tracker initialization."""
        assert portfolio.initial_capital == 100000.0
        assert portfolio.get_portfolio_value() == 100000.0
        assert portfolio.total_fees == 0.0
        assert portfolio.trade_count == 0
        assert len(portfolio.positions) == 0
    
    def test_add_trade_new_position(self, portfolio):
        """Test adding trade for new position."""
        realized_pnl = portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        
        assert realized_pnl == 0.0  # No realized P&L on opening
        assert portfolio.trade_count == 1
        assert portfolio.total_fees == 25.0
        
        position = portfolio.get_position("BTCUSDT")
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
    
    def test_portfolio_value_calculation(self, portfolio):
        """Test portfolio value calculation."""
        # Add position
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        
        # Update market price
        portfolio.update_market_prices({"BTCUSDT": 52000.0})
        
        # Portfolio value = initial_capital + unrealized_pnl - fees
        expected_value = 100000.0 + 2000.0 - 25.0
        assert abs(portfolio.get_portfolio_value() - expected_value) < 1e-6
    
    def test_multiple_positions(self, portfolio):
        """Test portfolio with multiple positions."""
        # Add multiple positions
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        portfolio.add_trade("ETHUSDT", 10.0, 3000.0, 15.0)
        
        # Update market prices
        portfolio.update_market_prices({
            "BTCUSDT": 52000.0,  # +2000 unrealized
            "ETHUSDT": 3200.0    # +2000 unrealized
        })
        
        expected_value = 100000.0 + 2000.0 + 2000.0 - 40.0  # -40 total fees
        assert abs(portfolio.get_portfolio_value() - expected_value) < 1e-6
    
    def test_realized_pnl_tracking(self, portfolio):
        """Test realized P&L tracking."""
        # Open position
        portfolio.add_trade("BTCUSDT", 2.0, 50000.0, 25.0)
        
        # Partial close with profit
        realized_pnl = portfolio.add_trade("BTCUSDT", -1.0, 52000.0, 25.0)
        
        assert realized_pnl == 2000.0
        assert portfolio.total_fees == 50.0
        
        # Check position state
        position = portfolio.get_position("BTCUSDT")
        assert position.quantity == 1.0
        assert position.realized_pnl == 2000.0
    
    def test_exposure_calculations(self, portfolio):
        """Test exposure calculations."""
        # Add long and short positions
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)    # Long $50k
        portfolio.add_trade("ETHUSDT", -10.0, 3000.0, 15.0)   # Short $30k
        
        portfolio.update_market_prices({
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0
        })
        
        assert portfolio.get_gross_exposure() == 80000.0  # |50k| + |30k|
        assert portfolio.get_net_exposure() == 20000.0    # 50k - 30k
        assert portfolio.get_total_notional() == 80000.0
    
    def test_daily_pnl_tracking(self, portfolio):
        """Test daily P&L tracking."""
        today = datetime.utcnow().date()
        
        # Add some trades
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        portfolio.add_trade("BTCUSDT", -0.5, 52000.0, 25.0)  # Realize $1000 profit
        
        daily_pnl = portfolio.get_daily_pnl(today)
        
        assert daily_pnl is not None
        assert daily_pnl.realized_pnl == 1000.0
        assert daily_pnl.fees == 50.0
        assert daily_pnl.starting_value == 100000.0
    
    def test_drawdown_tracking(self, portfolio):
        """Test drawdown tracking."""
        # Add position
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        
        # Simulate price movements
        portfolio.update_market_prices({"BTCUSDT": 55000.0})  # Profit
        assert portfolio.all_time_high > 100000.0
        
        portfolio.update_market_prices({"BTCUSDT": 45000.0})  # Loss
        current_drawdown = portfolio.get_current_drawdown()
        assert current_drawdown > 0
        assert portfolio.max_drawdown > 0
    
    def test_position_concentration(self, portfolio):
        """Test position concentration calculation."""
        # Add position worth 20% of portfolio
        portfolio.add_trade("BTCUSDT", 0.4, 50000.0, 25.0)  # $20k position
        
        portfolio.update_market_prices({"BTCUSDT": 50000.0})
        
        concentration = portfolio.get_position_concentration("BTCUSDT")
        expected_concentration = 20000.0 / portfolio.get_portfolio_value()
        
        assert abs(concentration - expected_concentration) < 1e-6
    
    def test_largest_position_concentration(self, portfolio):
        """Test largest position concentration."""
        # Add multiple positions
        portfolio.add_trade("BTCUSDT", 0.4, 50000.0, 25.0)   # $20k
        portfolio.add_trade("ETHUSDT", 10.0, 3000.0, 15.0)   # $30k
        
        portfolio.update_market_prices({
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0
        })
        
        symbol, concentration = portfolio.get_largest_position_concentration()
        
        assert symbol == "ETHUSDT"  # Larger position
        assert concentration > 0.25  # Should be around 30%
    
    def test_performance_metrics(self, portfolio):
        """Test performance metrics calculation."""
        # Add some trades over time
        base_time = datetime.utcnow()
        
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0, base_time)
        portfolio.add_trade("BTCUSDT", -0.5, 52000.0, 25.0, base_time + timedelta(hours=1))
        
        portfolio.update_market_prices({"BTCUSDT": 51000.0})
        
        metrics = portfolio.get_performance_metrics()
        
        assert 'portfolio_value' in metrics
        assert 'total_return' in metrics
        assert 'realized_pnl' in metrics
        assert 'unrealized_pnl' in metrics
        assert 'max_drawdown' in metrics
        assert 'trade_count' in metrics
        
        assert metrics['trade_count'] == 2
        assert metrics['realized_pnl'] == 1000.0  # From partial close
    
    def test_thread_safety(self, portfolio):
        """Test thread safety of portfolio operations."""
        def add_trades():
            for i in range(100):
                portfolio.add_trade("BTCUSDT", 0.01, 50000.0 + i, 1.0)
        
        def update_prices():
            for i in range(100):
                portfolio.update_market_prices({"BTCUSDT": 50000.0 + i * 10})
                time.sleep(0.001)  # Small delay
        
        # Run operations in parallel
        thread1 = Thread(target=add_trades)
        thread2 = Thread(target=update_prices)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify final state is consistent
        assert portfolio.trade_count == 100
        position = portfolio.get_position("BTCUSDT")
        assert position.quantity == 1.0  # 100 * 0.01
    
    def test_portfolio_summary(self, portfolio):
        """Test portfolio summary generation."""
        # Add some positions
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        portfolio.add_trade("ETHUSDT", 10.0, 3000.0, 15.0)
        
        portfolio.update_market_prices({
            "BTCUSDT": 52000.0,
            "ETHUSDT": 3200.0
        })
        
        summary = portfolio.get_portfolio_summary()
        
        assert 'portfolio_value' in summary
        assert 'total_return' in summary
        assert 'active_positions' in summary
        assert 'trade_count' in summary
        
        # Should have formatted strings
        assert '$' in summary['portfolio_value']
        assert '%' in summary['total_return']
    
    def test_daily_reset(self, portfolio):
        """Test daily P&L reset functionality."""
        today = datetime.utcnow().date()
        
        # Add some activity
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        
        # Reset for new day
        portfolio.reset_daily_pnl(today)
        
        daily_pnl = portfolio.get_daily_pnl(today)
        assert daily_pnl is not None
        assert daily_pnl.starting_value > 0
        assert daily_pnl.realized_pnl == 0.0  # Reset
    
    def test_portfolio_serialization(self, portfolio):
        """Test portfolio serialization to dictionary."""
        # Add some data
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        portfolio.update_market_prices({"BTCUSDT": 52000.0})
        
        portfolio_dict = portfolio.to_dict()
        
        assert 'positions' in portfolio_dict
        assert 'performance_metrics' in portfolio_dict
        assert 'daily_pnl' in portfolio_dict
        
        # Check position data
        assert 'BTCUSDT' in portfolio_dict['positions']
        btc_position = portfolio_dict['positions']['BTCUSDT']
        assert btc_position['quantity'] == 1.0
        assert btc_position['unrealized_pnl'] == 2000.0
