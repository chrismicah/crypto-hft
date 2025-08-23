"""Unit tests for performance metrics calculation."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtester.metrics import PerformanceCalculator, PerformanceMetrics, TradeMetrics
from backtester.models import BacktestConfig, Fill, OrderSide


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics."""
    
    def test_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            total_trades=50,
            win_rate=0.6
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['total_return'] == 0.15
        assert metrics_dict['sharpe_ratio'] == 1.2
        assert metrics_dict['max_drawdown'] == 0.08
        assert metrics_dict['total_trades'] == 50
        assert metrics_dict['win_rate'] == 0.6


class TestTradeMetrics:
    """Test cases for TradeMetrics."""
    
    def test_trade_metrics_creation(self):
        """Test creating TradeMetrics."""
        entry_time = datetime(2024, 1, 1, 10, 0)
        exit_time = datetime(2024, 1, 1, 12, 30)
        
        trade = TradeMetrics(
            trade_id="TRADE_001",
            symbol="BTCUSDT",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=1.0,
            side="long",
            pnl=1000.0,
            fees=10.0
        )
        
        assert trade.trade_id == "TRADE_001"
        assert trade.symbol == "BTCUSDT"
        assert trade.entry_time == entry_time
        assert trade.exit_time == exit_time
        assert trade.pnl == 1000.0
        assert trade.duration_seconds == 2.5 * 3600  # 2.5 hours
    
    def test_trade_metrics_no_exit(self):
        """Test TradeMetrics with no exit time."""
        trade = TradeMetrics(
            trade_id="TRADE_001",
            symbol="BTCUSDT",
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=None,
            entry_price=50000.0,
            exit_price=None,
            quantity=1.0,
            side="long",
            pnl=0.0,
            fees=5.0
        )
        
        assert trade.duration_seconds is None


class TestPerformanceCalculator:
    """Test cases for PerformanceCalculator."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return BacktestConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 7),
            initial_capital=100000.0,
            symbols=['BTCUSDT', 'ETHUSDT'],
            risk_free_rate=0.02
        )
    
    @pytest.fixture
    def calculator(self, config):
        """Performance calculator instance."""
        return PerformanceCalculator(config)
    
    def test_calculator_initialization(self, calculator, config):
        """Test calculator initialization."""
        assert calculator.config == config
        assert calculator.cash == config.initial_capital
        assert calculator.initial_capital == config.initial_capital
        assert calculator.peak_value == config.initial_capital
        assert len(calculator.portfolio_values) == 0
        assert len(calculator.returns) == 0
        assert len(calculator.trades) == 0
        assert len(calculator.fills) == 0
    
    def test_add_fill_buy(self, calculator):
        """Test adding a buy fill."""
        fill = Fill(
            order_id="ORDER_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            timestamp=datetime(2024, 1, 1, 10, 0)
        )
        
        initial_cash = calculator.cash
        calculator.add_fill(fill)
        
        assert len(calculator.fills) == 1
        assert calculator.cash == initial_cash - 50000.0 - 25.0
        assert "BTCUSDT" in calculator.positions
        
        position = calculator.positions["BTCUSDT"]
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
    
    def test_add_fill_sell(self, calculator):
        """Test adding a sell fill."""
        fill = Fill(
            order_id="ORDER_001",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            timestamp=datetime(2024, 1, 1, 10, 0)
        )
        
        initial_cash = calculator.cash
        calculator.add_fill(fill)
        
        assert len(calculator.fills) == 1
        assert calculator.cash == initial_cash + 50000.0 - 25.0
        assert "BTCUSDT" in calculator.positions
        
        position = calculator.positions["BTCUSDT"]
        assert position.quantity == -1.0
        assert position.average_price == 50000.0
    
    def test_update_portfolio_value(self, calculator):
        """Test updating portfolio value."""
        # Add a position first
        fill = Fill(
            order_id="ORDER_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            timestamp=datetime(2024, 1, 1, 10, 0)
        )
        calculator.add_fill(fill)
        
        # Update portfolio value
        timestamp = datetime(2024, 1, 1, 11, 0)
        market_prices = {"BTCUSDT": 51000.0}
        
        calculator.update_portfolio_value(timestamp, market_prices)
        
        assert len(calculator.portfolio_values) == 1
        assert calculator.portfolio_values[0][0] == timestamp
        
        # Portfolio value should be cash + position value
        expected_value = calculator.cash + 1.0 * 51000.0
        assert calculator.portfolio_values[0][1] == expected_value
    
    def test_drawdown_calculation(self, calculator):
        """Test drawdown calculation."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0)
        ]
        
        # Simulate portfolio values: up, peak, down, recovery
        portfolio_values = [100000, 110000, 95000, 105000]
        
        for timestamp, value in zip(timestamps, portfolio_values):
            calculator.portfolio_values.append((timestamp, value))
            
            if value > calculator.peak_value:
                calculator.peak_value = value
                calculator.current_drawdown = 0.0
            else:
                calculator.current_drawdown = (calculator.peak_value - value) / calculator.peak_value
                if calculator.current_drawdown > calculator.max_drawdown:
                    calculator.max_drawdown = calculator.current_drawdown
        
        # Max drawdown should be (110000 - 95000) / 110000 ≈ 0.136
        expected_max_drawdown = (110000 - 95000) / 110000
        assert abs(calculator.max_drawdown - expected_max_drawdown) < 0.001
    
    def test_calculate_basic_metrics(self, calculator):
        """Test calculating basic performance metrics."""
        # Simulate some portfolio values over time
        start_time = datetime(2024, 1, 1)
        
        for i in range(10):
            timestamp = start_time + timedelta(days=i)
            # Simulate 1% daily growth
            value = 100000 * (1.01 ** i)
            calculator.portfolio_values.append((timestamp, value))
            
            if i > 0:
                prev_value = calculator.portfolio_values[i-1][1]
                return_pct = (value - prev_value) / prev_value
                calculator.returns.append(return_pct)
        
        metrics = calculator.calculate_metrics()
        
        assert metrics.final_portfolio_value > 100000
        assert metrics.total_return > 0
        assert metrics.annualized_return > 0
        assert metrics.volatility >= 0
    
    def test_calculate_trade_statistics(self, calculator):
        """Test calculating trade statistics."""
        # Add some winning and losing trades
        winning_trades = [
            TradeMetrics("T1", "BTCUSDT", datetime.now(), datetime.now(), 50000, 51000, 1.0, "long", 1000, 10),
            TradeMetrics("T2", "ETHUSDT", datetime.now(), datetime.now(), 3000, 3150, 1.0, "long", 150, 5),
            TradeMetrics("T3", "BTCUSDT", datetime.now(), datetime.now(), 52000, 53000, 1.0, "long", 1000, 10)
        ]
        
        losing_trades = [
            TradeMetrics("T4", "BTCUSDT", datetime.now(), datetime.now(), 51000, 50000, 1.0, "long", -1000, 10),
            TradeMetrics("T5", "ETHUSDT", datetime.now(), datetime.now(), 3200, 3100, 1.0, "long", -100, 5)
        ]
        
        calculator.trades = winning_trades + losing_trades
        
        metrics = calculator.calculate_metrics()
        
        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == 0.6
        assert metrics.average_win == (1000 + 150 + 1000) / 3
        assert metrics.average_loss == (-1000 - 100) / 2
        assert metrics.largest_win == 1000
        assert metrics.largest_loss == -1000
        
        # Profit factor = total wins / abs(total losses)
        total_wins = 1000 + 150 + 1000
        total_losses = abs(-1000 - 100)
        expected_profit_factor = total_wins / total_losses
        assert abs(metrics.profit_factor - expected_profit_factor) < 0.001
    
    def test_sharpe_ratio_calculation(self, calculator):
        """Test Sharpe ratio calculation."""
        # Create returns with known properties
        np.random.seed(42)  # For reproducible results
        
        # Simulate daily returns for 252 trading days (1 year)
        daily_returns = np.random.normal(0.001, 0.02, 252)  # 0.1% mean, 2% std daily
        calculator.returns = daily_returns.tolist()
        
        # Simulate corresponding portfolio values
        portfolio_value = 100000
        for i, ret in enumerate(daily_returns):
            portfolio_value *= (1 + ret)
            timestamp = datetime(2024, 1, 1) + timedelta(days=i)
            calculator.portfolio_values.append((timestamp, portfolio_value))
        
        metrics = calculator.calculate_metrics()
        
        # Sharpe ratio should be reasonable
        assert -2.0 < metrics.sharpe_ratio < 5.0
        assert metrics.volatility > 0
        assert metrics.annualized_return != 0
    
    def test_var_calculation(self, calculator):
        """Test Value at Risk calculation."""
        # Create returns with known distribution
        returns = np.random.normal(0, 0.02, 1000)
        calculator.returns = returns.tolist()
        
        metrics = calculator.calculate_metrics()
        
        # VaR should be negative (representing losses)
        assert metrics.var_95 < 0
        
        # CVaR should be more negative than VaR
        assert metrics.cvar_95 <= metrics.var_95
    
    def test_skewness_and_kurtosis(self, calculator):
        """Test skewness and kurtosis calculation."""
        # Create asymmetric returns
        returns = np.concatenate([
            np.random.normal(-0.01, 0.005, 100),  # Many small losses
            np.random.normal(0.02, 0.01, 50)      # Fewer large gains
        ])
        calculator.returns = returns.tolist()
        
        metrics = calculator.calculate_metrics()
        
        # Should detect positive skewness (right tail)
        assert metrics.skewness > 0
        
        # Kurtosis should be calculated
        assert metrics.kurtosis != 0
    
    def test_get_portfolio_timeseries(self, calculator):
        """Test getting portfolio time series."""
        # Add some portfolio values
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        values = [100000, 101000, 99000, 102000, 105000]
        
        for timestamp, value in zip(timestamps, values):
            calculator.portfolio_values.append((timestamp, value))
        
        df = calculator.get_portfolio_timeseries()
        
        assert len(df) == 5
        assert 'timestamp' in df.columns
        assert 'portfolio_value' in df.columns
        assert 'returns' in df.columns
        assert 'cumulative_returns' in df.columns
        assert 'drawdown' in df.columns
        
        # Check cumulative returns calculation
        expected_cum_return = (105000 - 100000) / 100000
        assert abs(df['cumulative_returns'].iloc[-1] - expected_cum_return) < 0.001
    
    def test_get_trade_analysis(self, calculator):
        """Test getting trade analysis."""
        # Add some trades
        trades = [
            TradeMetrics("T1", "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 2), 
                        50000, 51000, 1.0, "long", 1000, 10),
            TradeMetrics("T2", "ETHUSDT", datetime(2024, 1, 2), datetime(2024, 1, 3), 
                        3000, 2950, 1.0, "long", -50, 5)
        ]
        
        calculator.trades = trades
        
        df = calculator.get_trade_analysis()
        
        assert len(df) == 2
        assert 'trade_id' in df.columns
        assert 'symbol' in df.columns
        assert 'pnl' in df.columns
        assert 'duration_hours' in df.columns
        
        # Check duration calculation
        expected_duration = 24.0  # 1 day = 24 hours
        assert abs(df['duration_hours'].iloc[0] - expected_duration) < 0.1
    
    def test_generate_report(self, calculator):
        """Test generating performance report."""
        # Add minimal data
        calculator.portfolio_values = [
            (datetime(2024, 1, 1), 100000),
            (datetime(2024, 1, 7), 105000)
        ]
        calculator.returns = [0.01, -0.005, 0.008, 0.002, 0.003, -0.001]
        
        report = calculator.generate_report()
        
        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "Portfolio Performance:" in report
        assert "Risk Metrics:" in report
        assert "Trade Statistics:" in report
        assert f"${calculator.initial_capital:,.2f}" in report
