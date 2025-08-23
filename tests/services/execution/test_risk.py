"""Unit tests for risk management layer."""

import pytest
import numpy as np
from datetime import datetime, timedelta, date
from unittest.mock import Mock

from services.execution.risk import (
    RiskManager, RiskParameters, KellyCriterionCalculator,
    RiskCheckResult, RiskViolationType, RiskCheckResponse
)
from services.execution.portfolio import PortfolioTracker


class TestKellyCriterionCalculator:
    """Test cases for Kelly Criterion calculator."""
    
    @pytest.fixture
    def kelly_calc(self):
        """Create Kelly calculator instance."""
        return KellyCriterionCalculator(lookback_period=50)
    
    def test_kelly_calculator_initialization(self, kelly_calc):
        """Test Kelly calculator initialization."""
        assert kelly_calc.lookback_period == 50
        assert len(kelly_calc.trade_history) == 0
    
    def test_add_trade_result(self, kelly_calc):
        """Test adding trade results."""
        kelly_calc.add_trade_result(
            symbol="BTCUSDT",
            pnl=1000.0,
            quantity=1.0,
            entry_price=50000.0,
            exit_price=52000.0
        )
        
        assert len(kelly_calc.trade_history) == 1
        trade = kelly_calc.trade_history[0]
        
        assert trade['symbol'] == "BTCUSDT"
        assert trade['pnl'] == 1000.0
        assert trade['return_pct'] == 0.04  # 2000/50000
    
    def test_kelly_calculation_insufficient_trades(self, kelly_calc):
        """Test Kelly calculation with insufficient trades."""
        # Add only a few trades
        for i in range(5):
            kelly_calc.add_trade_result("BTCUSDT", 100.0, 1.0, 50000.0, 50100.0)
        
        kelly_fraction, stats = kelly_calc.calculate_kelly_fraction("BTCUSDT", min_trades=20)
        
        assert kelly_fraction == 0.0
        assert stats['reason'] == 'insufficient_trades'
        assert stats['trade_count'] == 5
    
    def test_kelly_calculation_with_wins_and_losses(self, kelly_calc):
        """Test Kelly calculation with mixed results."""
        # Add winning trades
        for i in range(15):
            kelly_calc.add_trade_result("BTCUSDT", 1000.0, 1.0, 50000.0, 52000.0)
        
        # Add losing trades
        for i in range(10):
            kelly_calc.add_trade_result("BTCUSDT", -800.0, 1.0, 50000.0, 48400.0)
        
        kelly_fraction, stats = kelly_calc.calculate_kelly_fraction("BTCUSDT", min_trades=20)
        
        assert kelly_fraction > 0  # Should be positive with profitable system
        assert stats['trade_count'] == 25
        assert stats['win_rate'] == 0.6  # 15/25
        assert stats['avg_win'] > 0
        assert stats['avg_loss'] > 0
        assert 'odds_ratio' in stats
    
    def test_kelly_calculation_no_wins(self, kelly_calc):
        """Test Kelly calculation with no winning trades."""
        # Add only losing trades
        for i in range(25):
            kelly_calc.add_trade_result("BTCUSDT", -500.0, 1.0, 50000.0, 49000.0)
        
        kelly_fraction, stats = kelly_calc.calculate_kelly_fraction("BTCUSDT", min_trades=20)
        
        assert kelly_fraction == 0.0
        assert stats['reason'] == 'no_wins_or_losses'
    
    def test_kelly_calculation_no_losses(self, kelly_calc):
        """Test Kelly calculation with no losing trades."""
        # Add only winning trades
        for i in range(25):
            kelly_calc.add_trade_result("BTCUSDT", 1000.0, 1.0, 50000.0, 52000.0)
        
        kelly_fraction, stats = kelly_calc.calculate_kelly_fraction("BTCUSDT", min_trades=20)
        
        assert kelly_fraction == 0.0
        assert stats['reason'] == 'no_wins_or_losses'
    
    def test_get_position_size(self, kelly_calc):
        """Test position size recommendation."""
        # Add profitable trade history
        for i in range(30):
            if i % 3 == 0:  # 1/3 losing trades
                kelly_calc.add_trade_result("BTCUSDT", -500.0, 1.0, 50000.0, 49000.0)
            else:  # 2/3 winning trades
                kelly_calc.add_trade_result("BTCUSDT", 1000.0, 1.0, 50000.0, 52000.0)
        
        portfolio_value = 100000.0
        position_size, stats = kelly_calc.get_position_size(
            symbol="BTCUSDT",
            portfolio_value=portfolio_value,
            kelly_fraction_multiplier=0.25,
            max_position_pct=0.1
        )
        
        assert position_size > 0
        assert position_size <= portfolio_value * 0.1  # Shouldn't exceed max
        assert 'recommended_size' in stats
        assert 'size_limited_by' in stats
    
    def test_trade_history_limit(self, kelly_calc):
        """Test trade history size limit."""
        # Add more trades than lookback period
        for i in range(150):
            kelly_calc.add_trade_result("BTCUSDT", 100.0, 1.0, 50000.0, 50100.0)
        
        # Should be limited to 2x lookback period
        assert len(kelly_calc.trade_history) <= kelly_calc.lookback_period * 2


class TestRiskParameters:
    """Test cases for RiskParameters."""
    
    def test_valid_parameters(self):
        """Test valid risk parameters."""
        params = RiskParameters(
            max_position_size_pct=0.1,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
            kelly_fraction=0.25
        )
        
        assert params.max_position_size_pct == 0.1
        assert params.max_daily_loss_pct == 0.05
        assert params.max_drawdown_pct == 0.15
        assert params.kelly_fraction == 0.25
    
    def test_invalid_position_size(self):
        """Test invalid position size parameter."""
        with pytest.raises(ValueError, match="max_position_size_pct must be between 0 and 1"):
            RiskParameters(max_position_size_pct=1.5)
    
    def test_invalid_kelly_fraction(self):
        """Test invalid Kelly fraction parameter."""
        with pytest.raises(ValueError, match="kelly_fraction must be between 0 and 1"):
            RiskParameters(kelly_fraction=1.5)
    
    def test_invalid_win_rate(self):
        """Test invalid win rate parameter."""
        with pytest.raises(ValueError, match="min_win_rate must be between 0 and 1"):
            RiskParameters(min_win_rate=1.5)


class TestRiskCheckResponse:
    """Test cases for RiskCheckResponse."""
    
    def test_approved_response(self):
        """Test approved risk check response."""
        response = RiskCheckResponse(
            result=RiskCheckResult.APPROVED,
            original_quantity=1.0,
            approved_quantity=1.0,
            reason="Order approved"
        )
        
        assert response.is_approved
        assert not response.is_rejected
        assert not response.is_modified
        assert not response.is_halted
    
    def test_rejected_response(self):
        """Test rejected risk check response."""
        response = RiskCheckResponse(
            result=RiskCheckResult.REJECTED,
            original_quantity=1.0,
            approved_quantity=0.0,
            violation_type=RiskViolationType.POSITION_SIZE_EXCEEDED,
            reason="Position size exceeded"
        )
        
        assert not response.is_approved
        assert response.is_rejected
        assert not response.is_modified
        assert not response.is_halted
        assert response.violation_type == RiskViolationType.POSITION_SIZE_EXCEEDED
    
    def test_modified_response(self):
        """Test modified risk check response."""
        response = RiskCheckResponse(
            result=RiskCheckResult.MODIFIED,
            original_quantity=2.0,
            approved_quantity=1.0,
            reason="Order size reduced"
        )
        
        assert not response.is_approved
        assert not response.is_rejected
        assert response.is_modified
        assert not response.is_halted
    
    def test_halted_response(self):
        """Test halted risk check response."""
        response = RiskCheckResponse(
            result=RiskCheckResult.HALTED,
            original_quantity=1.0,
            approved_quantity=0.0,
            violation_type=RiskViolationType.DAILY_LOSS_LIMIT,
            reason="Daily loss limit exceeded"
        )
        
        assert not response.is_approved
        assert not response.is_rejected
        assert not response.is_modified
        assert response.is_halted


class TestRiskManager:
    """Test cases for RiskManager."""
    
    @pytest.fixture
    def portfolio(self):
        """Create portfolio tracker."""
        return PortfolioTracker(initial_capital=100000.0)
    
    @pytest.fixture
    def risk_params(self):
        """Create risk parameters."""
        return RiskParameters(
            max_position_size_pct=0.1,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
            kelly_fraction=0.25,
            use_kelly_sizing=False  # Disable for simpler testing
        )
    
    @pytest.fixture
    def risk_manager(self, portfolio, risk_params):
        """Create risk manager."""
        return RiskManager(portfolio, risk_params)
    
    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert not risk_manager.trading_halted
        assert risk_manager.halt_reason == ""
        assert risk_manager.halt_timestamp is None
    
    def test_valid_order_approval(self, risk_manager):
        """Test approval of valid order."""
        response = risk_manager.check_order(
            symbol="BTCUSDT",
            quantity=0.1,  # $5k order (5% of portfolio)
            price=50000.0
        )
        
        assert response.is_approved
        assert response.approved_quantity == 0.1
        assert response.violation_type is None
    
    def test_position_size_limit_rejection(self, risk_manager):
        """Test rejection due to position size limit."""
        response = risk_manager.check_order(
            symbol="BTCUSDT",
            quantity=0.5,  # $25k order (25% of portfolio, exceeds 10% limit)
            price=50000.0
        )
        
        assert response.is_rejected
        assert response.approved_quantity == 0.0
        assert response.violation_type == RiskViolationType.POSITION_SIZE_EXCEEDED
    
    def test_position_size_modification(self, risk_manager):
        """Test order size modification to fit limits."""
        # First add a small position
        risk_manager.portfolio.add_trade("BTCUSDT", 0.05, 50000.0, 25.0)
        
        # Try to add more that would exceed limit
        response = risk_manager.check_order(
            symbol="BTCUSDT",
            quantity=0.15,  # Would make total position 0.2 (20% of portfolio)
            price=50000.0
        )
        
        # Should be modified to fit within 10% limit
        assert response.is_modified
        assert response.approved_quantity < 0.15
        assert response.approved_quantity > 0
    
    def test_daily_loss_limit_halt(self, risk_manager):
        """Test trading halt due to daily loss limit."""
        # Simulate large loss
        risk_manager.portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        risk_manager.portfolio.update_market_prices({"BTCUSDT": 45000.0})  # -$5k loss
        
        # This should trigger daily loss limit (5% = $5k)
        response = risk_manager.check_order("ETHUSDT", 1.0, 3000.0)
        
        assert response.is_halted
        assert response.violation_type == RiskViolationType.DAILY_LOSS_LIMIT
        assert risk_manager.trading_halted
    
    def test_max_drawdown_halt(self, risk_manager):
        """Test trading halt due to max drawdown."""
        # Simulate large drawdown
        risk_manager.portfolio.add_trade("BTCUSDT", 2.0, 50000.0, 25.0)
        risk_manager.portfolio.update_market_prices({"BTCUSDT": 58000.0})  # Profit first
        risk_manager.portfolio.update_market_prices({"BTCUSDT": 42000.0})  # Then large loss
        
        # This should trigger max drawdown limit (15%)
        response = risk_manager.check_order("ETHUSDT", 1.0, 3000.0)
        
        assert response.is_halted
        assert response.violation_type == RiskViolationType.MAX_DRAWDOWN_EXCEEDED
        assert risk_manager.trading_halted
    
    def test_concentration_risk_rejection(self, risk_manager):
        """Test rejection due to concentration risk."""
        # Set concentration limit to 15%
        risk_manager.params.max_concentration_pct = 0.15
        
        response = risk_manager.check_order(
            symbol="BTCUSDT",
            quantity=0.4,  # $20k order (20% of portfolio, exceeds 15% concentration limit)
            price=50000.0
        )
        
        assert response.is_rejected
        assert response.violation_type == RiskViolationType.CONCENTRATION_RISK
    
    def test_insufficient_capital_rejection(self, risk_manager):
        """Test rejection due to insufficient capital."""
        # Set high margin requirement
        risk_manager.params.margin_requirement = 0.9
        risk_manager.params.min_capital_buffer = 50000.0
        
        response = risk_manager.check_order(
            symbol="BTCUSDT",
            quantity=0.2,  # $10k order but high margin requirement
            price=50000.0
        )
        
        assert response.is_rejected
        assert response.violation_type == RiskViolationType.INSUFFICIENT_CAPITAL
    
    def test_invalid_order_rejection(self, risk_manager):
        """Test rejection of invalid orders."""
        # Zero quantity
        response = risk_manager.check_order("BTCUSDT", 0.0, 50000.0)
        assert response.is_rejected
        assert response.violation_type == RiskViolationType.INVALID_ORDER
        
        # Negative price
        response = risk_manager.check_order("BTCUSDT", 1.0, -50000.0)
        assert response.is_rejected
        assert response.violation_type == RiskViolationType.INVALID_ORDER
    
    def test_trading_halt_override(self, risk_manager):
        """Test that halted trading blocks all orders."""
        # Manually halt trading
        risk_manager._halt_trading("Manual halt for testing")
        
        response = risk_manager.check_order("BTCUSDT", 0.01, 50000.0)
        
        assert response.is_halted
        assert response.violation_type == RiskViolationType.TRADING_HALTED
        assert "Manual halt" in response.reason
    
    def test_trading_resume(self, risk_manager):
        """Test resuming trading after halt."""
        # Halt trading
        risk_manager._halt_trading("Test halt")
        assert risk_manager.trading_halted
        
        # Resume trading
        success = risk_manager.resume_trading("Test resume")
        assert success
        assert not risk_manager.trading_halted
        assert risk_manager.halt_reason == ""
    
    def test_trading_resume_with_violations(self, risk_manager):
        """Test that trading cannot resume with active violations."""
        # Create daily loss violation
        risk_manager.portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        risk_manager.portfolio.update_market_prices({"BTCUSDT": 45000.0})
        
        # This should halt trading
        risk_manager.check_order("ETHUSDT", 1.0, 3000.0)
        assert risk_manager.trading_halted
        
        # Try to resume - should fail due to ongoing violation
        success = risk_manager.resume_trading("Attempted resume")
        assert not success
        assert risk_manager.trading_halted
    
    def test_daily_reset_functionality(self, risk_manager):
        """Test daily reset functionality."""
        # Simulate loss on previous day
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        risk_manager.last_daily_reset = yesterday
        
        risk_manager.portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        risk_manager.portfolio.update_market_prices({"BTCUSDT": 45000.0})
        
        # Halt trading due to loss
        risk_manager._halt_trading("Daily loss limit")
        
        # Check order should trigger daily reset and resume trading
        response = risk_manager.check_order("ETHUSDT", 0.01, 3000.0)
        
        # Should be approved after daily reset
        assert response.is_approved or not risk_manager.trading_halted
    
    def test_kelly_sizing_integration(self, risk_manager):
        """Test Kelly Criterion integration."""
        # Enable Kelly sizing
        risk_manager.params.use_kelly_sizing = True
        
        # Add trade history to Kelly calculator
        for i in range(30):
            if i % 3 == 0:  # Losing trades
                risk_manager.kelly_calc.add_trade_result("BTCUSDT", -500.0, 1.0, 50000.0, 49000.0)
            else:  # Winning trades
                risk_manager.kelly_calc.add_trade_result("BTCUSDT", 1000.0, 1.0, 50000.0, 52000.0)
        
        # Large order should be reduced by Kelly sizing
        response = risk_manager.check_order("BTCUSDT", 0.3, 50000.0)  # 30% of portfolio
        
        # Should be modified to smaller size based on Kelly
        assert response.is_modified or response.approved_quantity < 0.3
    
    def test_risk_status_reporting(self, risk_manager):
        """Test risk status reporting."""
        status = risk_manager.get_risk_status()
        
        assert 'trading_halted' in status
        assert 'portfolio_value' in status
        assert 'current_drawdown' in status
        assert 'max_drawdown_limit' in status
        assert 'daily_loss_limit' in status
        
        assert status['trading_halted'] is False
        assert status['portfolio_value'] == 100000.0
    
    def test_position_sizing_recommendation(self, risk_manager):
        """Test position sizing recommendations."""
        # Test percentage-based sizing
        recommendation = risk_manager.get_position_sizing_recommendation("BTCUSDT", 0.05)
        
        assert 'recommended_size' in recommendation
        assert 'method' in recommendation
        assert recommendation['method'] == 'percentage_based'
        assert recommendation['recommended_size'] == 5000.0  # 5% of $100k
    
    def test_add_trade_result_integration(self, risk_manager):
        """Test adding trade results for Kelly calculation."""
        risk_manager.add_trade_result(
            symbol="BTCUSDT",
            pnl=1000.0,
            quantity=1.0,
            entry_price=50000.0,
            exit_price=52000.0
        )
        
        assert len(risk_manager.kelly_calc.trade_history) == 1
        trade = risk_manager.kelly_calc.trade_history[0]
        assert trade['symbol'] == "BTCUSDT"
        assert trade['pnl'] == 1000.0


class TestRiskManagerIntegration:
    """Integration tests for risk manager with realistic scenarios."""
    
    def test_gradual_loss_scenario(self):
        """Test gradual loss leading to halt."""
        portfolio = PortfolioTracker(initial_capital=100000.0)
        params = RiskParameters(max_daily_loss_pct=0.05)
        risk_manager = RiskManager(portfolio, params)
        
        # Add position
        portfolio.add_trade("BTCUSDT", 2.0, 50000.0, 50.0)
        
        # Simulate gradual price decline
        prices = [49000, 48000, 47000, 46000, 45000]  # -10% total
        
        for price in prices:
            portfolio.update_market_prices({"BTCUSDT": price})
            response = risk_manager.check_order("ETHUSDT", 0.1, 3000.0)
            
            if response.is_halted:
                break
        
        # Should be halted before reaching -10%
        assert risk_manager.trading_halted
        assert "Daily loss" in risk_manager.halt_reason
    
    def test_position_building_with_limits(self):
        """Test building position within risk limits."""
        portfolio = PortfolioTracker(initial_capital=100000.0)
        params = RiskParameters(max_position_size_pct=0.1)
        risk_manager = RiskManager(portfolio, params)
        
        # Build position gradually
        orders = [0.02, 0.03, 0.04, 0.05]  # Total would be 14% without limits
        approved_total = 0.0
        
        for order_size in orders:
            response = risk_manager.check_order("BTCUSDT", order_size, 50000.0)
            
            if response.is_approved or response.is_modified:
                portfolio.add_trade("BTCUSDT", response.approved_quantity, 50000.0, 25.0)
                approved_total += response.approved_quantity
            else:
                break
        
        # Total position should not exceed 10% limit
        final_position = portfolio.get_position("BTCUSDT")
        position_value = final_position.quantity * 50000.0
        position_pct = position_value / portfolio.get_portfolio_value()
        
        assert position_pct <= 0.1 + 1e-6  # Allow small floating point error
    
    def test_recovery_after_halt(self):
        """Test recovery and resumption after halt."""
        portfolio = PortfolioTracker(initial_capital=100000.0)
        params = RiskParameters(max_daily_loss_pct=0.05)
        risk_manager = RiskManager(portfolio, params)
        
        # Create loss scenario
        portfolio.add_trade("BTCUSDT", 1.0, 50000.0, 25.0)
        portfolio.update_market_prices({"BTCUSDT": 45000.0})  # -5% loss
        
        # Should halt trading
        response = risk_manager.check_order("ETHUSDT", 0.1, 3000.0)
        assert response.is_halted
        
        # Simulate price recovery
        portfolio.update_market_prices({"BTCUSDT": 52000.0})  # +4% gain
        
        # Should be able to resume trading
        success = risk_manager.resume_trading("Price recovered")
        assert success
        
        # New orders should be approved
        response = risk_manager.check_order("ETHUSDT", 0.1, 3000.0)
        assert response.is_approved
