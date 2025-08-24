"""
Unit tests for EVOP Framework data models.

Tests the data structures used for strategy management,
performance tracking, and evolutionary optimization.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from services.orchestrator.models import (
    StrategyParameters, PerformanceMetrics, StrategyInstance, PromotionEvent,
    EVOPConfiguration, StrategyStatus, PromotionReason
)


class TestStrategyParameters:
    """Test the StrategyParameters data class."""
    
    def test_strategy_parameters_creation(self):
        """Test creation with default values."""
        params = StrategyParameters()
        
        assert params.entry_z_score == 2.0
        assert params.exit_z_score == 0.5
        assert params.stop_loss_z_score == 4.0
        assert params.max_position_size == 10000.0
        assert params.max_drawdown_percent == 10.0
        assert params.max_daily_loss == 5000.0
        assert params.kelly_fraction == 0.25
        assert params.kalman_process_noise == 1e-5
        assert params.kalman_observation_noise == 1e-3
        assert params.garch_window_size == 500
        assert params.bocd_hazard_rate == 0.004
        assert params.order_timeout_seconds == 30
        assert params.max_slippage_bps == 10
        assert params.min_order_size == 100.0
        assert params.custom_params == {}
    
    def test_strategy_parameters_custom_values(self):
        """Test creation with custom values."""
        custom_params = {'test_param': 'test_value'}
        
        params = StrategyParameters(
            entry_z_score=3.0,
            exit_z_score=1.0,
            max_position_size=20000.0,
            custom_params=custom_params
        )
        
        assert params.entry_z_score == 3.0
        assert params.exit_z_score == 1.0
        assert params.max_position_size == 20000.0
        assert params.custom_params == custom_params
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        params = StrategyParameters(
            entry_z_score=2.5,
            max_position_size=15000.0,
            custom_params={'key': 'value'}
        )
        
        params_dict = params.to_dict()
        
        assert isinstance(params_dict, dict)
        assert params_dict['entry_z_score'] == 2.5
        assert params_dict['max_position_size'] == 15000.0
        assert params_dict['custom_params'] == {'key': 'value'}
        assert 'exit_z_score' in params_dict
        assert 'kelly_fraction' in params_dict
    
    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        params_dict = {
            'entry_z_score': 2.5,
            'exit_z_score': 0.8,
            'max_position_size': 15000.0,
            'custom_params': {'test': 'value'}
        }
        
        params = StrategyParameters.from_dict(params_dict)
        
        assert params.entry_z_score == 2.5
        assert params.exit_z_score == 0.8
        assert params.max_position_size == 15000.0
        assert params.custom_params == {'test': 'value'}
        
        # Other fields should have default values
        assert params.stop_loss_z_score == 4.0
        assert params.kelly_fraction == 0.25


class TestPerformanceMetrics:
    """Test the PerformanceMetrics data class."""
    
    def test_performance_metrics_defaults(self):
        """Test default values."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_pnl == Decimal('0.0')
        assert metrics.realized_pnl == Decimal('0.0')
        assert metrics.unrealized_pnl == Decimal('0.0')
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio is None
        assert metrics.calmar_ratio is None
        assert metrics.max_drawdown == 0.0
        assert metrics.current_drawdown == 0.0
        assert metrics.volatility == 0.0
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
        assert metrics.profit_factor is None
        assert metrics.avg_fill_time == 0.0
        assert metrics.avg_slippage_bps == 0.0
        assert metrics.total_fees == Decimal('0.0')
        assert metrics.daily_returns == []
        assert metrics.monthly_returns == []
        assert isinstance(metrics.start_time, datetime)
        assert isinstance(metrics.last_updated, datetime)
    
    def test_update_trade_metrics(self):
        """Test trade metrics calculation."""
        metrics = PerformanceMetrics()
        
        # Set up trade data
        metrics.total_trades = 10
        metrics.winning_trades = 6
        metrics.losing_trades = 4
        metrics.avg_win = 100.0
        metrics.avg_loss = -50.0
        
        # Update metrics
        metrics.update_trade_metrics()
        
        assert metrics.win_rate == 0.6  # 6/10
        assert metrics.profit_factor == 3.0  # (6*100) / (4*50)
    
    def test_update_trade_metrics_no_losses(self):
        """Test trade metrics when there are no losing trades."""
        metrics = PerformanceMetrics()
        
        metrics.total_trades = 5
        metrics.winning_trades = 5
        metrics.losing_trades = 0
        metrics.avg_win = 100.0
        metrics.avg_loss = 0.0
        
        metrics.update_trade_metrics()
        
        assert metrics.win_rate == 1.0
        assert metrics.profit_factor is None  # Can't calculate without losses
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics()
        metrics.total_pnl = Decimal('1000.0')
        metrics.sharpe_ratio = 1.5
        metrics.total_trades = 10
        metrics.daily_returns = [0.01, 0.02, -0.005]
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['total_pnl'] == 1000.0  # Converted from Decimal
        assert metrics_dict['sharpe_ratio'] == 1.5
        assert metrics_dict['total_trades'] == 10
        assert metrics_dict['daily_returns'] == [0.01, 0.02, -0.005]
        assert 'start_time' in metrics_dict
        assert 'last_updated' in metrics_dict


class TestStrategyInstance:
    """Test the StrategyInstance data class."""
    
    def test_strategy_instance_creation(self):
        """Test creation with default values."""
        instance = StrategyInstance()
        
        assert isinstance(instance.instance_id, str)
        assert len(instance.instance_id) > 0  # Should generate UUID
        assert instance.name == "Strategy"
        assert instance.is_champion is False
        assert isinstance(instance.parameters, StrategyParameters)
        assert instance.status == StrategyStatus.PENDING
        assert isinstance(instance.created_at, datetime)
        assert instance.started_at is None
        assert instance.stopped_at is None
        assert isinstance(instance.performance, PerformanceMetrics)
        assert instance.allocated_capital == Decimal('100000.0')
        assert instance.current_positions == {}
        assert instance.description == ""
        assert instance.tags == []
    
    def test_strategy_instance_custom_values(self):
        """Test creation with custom values."""
        custom_params = StrategyParameters(entry_z_score=3.0)
        
        instance = StrategyInstance(
            name="Test Strategy",
            is_champion=True,
            parameters=custom_params,
            allocated_capital=Decimal('200000.0'),
            description="Test description",
            tags=["test", "champion"]
        )
        
        assert instance.name == "Test Strategy"
        assert instance.is_champion is True
        assert instance.parameters.entry_z_score == 3.0
        assert instance.allocated_capital == Decimal('200000.0')
        assert instance.description == "Test description"
        assert instance.tags == ["test", "champion"]
    
    def test_get_runtime(self):
        """Test runtime calculation."""
        instance = StrategyInstance()
        
        # No start time
        assert instance.get_runtime() is None
        
        # With start time, no stop time
        instance.started_at = datetime.now() - timedelta(hours=2)
        runtime = instance.get_runtime()
        assert isinstance(runtime, timedelta)
        assert runtime.total_seconds() > 7000  # ~2 hours
        
        # With start and stop time
        instance.stopped_at = instance.started_at + timedelta(hours=1)
        runtime = instance.get_runtime()
        assert runtime == timedelta(hours=1)
    
    def test_is_active(self):
        """Test active status check."""
        instance = StrategyInstance()
        
        # Pending state
        assert not instance.is_active()
        
        # Running state
        instance.status = StrategyStatus.RUNNING
        assert instance.is_active()
        
        # Stopped state
        instance.status = StrategyStatus.STOPPED
        assert not instance.is_active()
        
        # Failed state
        instance.status = StrategyStatus.FAILED
        assert not instance.is_active()
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        instance = StrategyInstance(
            name="Test Strategy",
            is_champion=True,
            allocated_capital=Decimal('150000.0')
        )
        instance.started_at = datetime(2024, 1, 1, 12, 0, 0)
        instance.current_positions = {"BTCUSDT": 1000.0}
        
        instance_dict = instance.to_dict()
        
        assert isinstance(instance_dict, dict)
        assert instance_dict['name'] == "Test Strategy"
        assert instance_dict['is_champion'] is True
        assert instance_dict['allocated_capital'] == 150000.0
        assert instance_dict['started_at'] == "2024-01-01T12:00:00"
        assert instance_dict['current_positions'] == {"BTCUSDT": 1000.0}
        assert 'instance_id' in instance_dict
        assert 'parameters' in instance_dict
        assert 'performance' in instance_dict


class TestPromotionEvent:
    """Test the PromotionEvent data class."""
    
    def test_promotion_event_creation(self):
        """Test creation with default values."""
        event = PromotionEvent()
        
        assert isinstance(event.event_id, str)
        assert len(event.event_id) > 0
        assert isinstance(event.timestamp, datetime)
        assert event.old_champion_id == ""
        assert event.new_champion_id == ""
        assert event.old_champion_name == ""
        assert event.new_champion_name == ""
        assert event.reason == PromotionReason.MANUAL_OVERRIDE
        assert event.confidence_score == 0.0
        assert event.old_champion_performance == {}
        assert event.new_champion_performance == {}
        assert event.evaluation_period_days == 0
        assert event.notes == ""
    
    def test_promotion_event_custom_values(self):
        """Test creation with custom values."""
        old_perf = {"sharpe_ratio": 1.0}
        new_perf = {"sharpe_ratio": 1.5}
        
        event = PromotionEvent(
            old_champion_id="old_id",
            new_champion_id="new_id",
            old_champion_name="Old Champion",
            new_champion_name="New Champion",
            reason=PromotionReason.SUPERIOR_SHARPE,
            confidence_score=0.95,
            old_champion_performance=old_perf,
            new_champion_performance=new_perf,
            evaluation_period_days=14,
            notes="Automatic promotion"
        )
        
        assert event.old_champion_id == "old_id"
        assert event.new_champion_id == "new_id"
        assert event.old_champion_name == "Old Champion"
        assert event.new_champion_name == "New Champion"
        assert event.reason == PromotionReason.SUPERIOR_SHARPE
        assert event.confidence_score == 0.95
        assert event.old_champion_performance == old_perf
        assert event.new_champion_performance == new_perf
        assert event.evaluation_period_days == 14
        assert event.notes == "Automatic promotion"
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        event = PromotionEvent(
            old_champion_id="old_id",
            new_champion_id="new_id",
            reason=PromotionReason.SUPERIOR_SHARPE,
            confidence_score=0.95
        )
        
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert event_dict['old_champion_id'] == "old_id"
        assert event_dict['new_champion_id'] == "new_id"
        assert event_dict['reason'] == "superior_sharpe"
        assert event_dict['confidence_score'] == 0.95
        assert 'event_id' in event_dict
        assert 'timestamp' in event_dict


class TestEVOPConfiguration:
    """Test the EVOPConfiguration data class."""
    
    def test_evop_configuration_defaults(self):
        """Test default configuration values."""
        config = EVOPConfiguration()
        
        assert config.max_challengers == 3
        assert config.challenger_capital_fraction == 0.2
        assert config.min_evaluation_period_days == 7
        assert config.max_evaluation_period_days == 30
        assert config.min_trades_for_evaluation == 10
        assert config.required_confidence_level == 0.95
        assert config.min_sharpe_improvement == 0.1
        assert config.min_calmar_improvement == 0.1
        assert config.max_drawdown_tolerance == 0.15
        assert config.max_total_allocation == 1.0
        assert config.emergency_stop_drawdown == 0.25
        assert config.parameter_mutation_rate == 0.1
        assert config.parameter_mutation_std == 0.05
        assert config.evaluation_frequency_hours == 6
        assert config.challenger_restart_on_failure is True
    
    def test_evop_configuration_custom_values(self):
        """Test configuration with custom values."""
        config = EVOPConfiguration(
            max_challengers=5,
            challenger_capital_fraction=0.3,
            required_confidence_level=0.99,
            evaluation_frequency_hours=12
        )
        
        assert config.max_challengers == 5
        assert config.challenger_capital_fraction == 0.3
        assert config.required_confidence_level == 0.99
        assert config.evaluation_frequency_hours == 12
        
        # Other values should remain default
        assert config.min_evaluation_period_days == 7
        assert config.parameter_mutation_rate == 0.1
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        config = EVOPConfiguration(
            max_challengers=4,
            required_confidence_level=0.98
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['max_challengers'] == 4
        assert config_dict['required_confidence_level'] == 0.98
        assert 'challenger_capital_fraction' in config_dict
        assert 'min_evaluation_period_days' in config_dict
        assert 'parameter_mutation_rate' in config_dict


class TestEnums:
    """Test the enumeration classes."""
    
    def test_strategy_status_values(self):
        """Test StrategyStatus enum values."""
        assert StrategyStatus.PENDING.value == "pending"
        assert StrategyStatus.RUNNING.value == "running"
        assert StrategyStatus.PAUSED.value == "paused"
        assert StrategyStatus.STOPPED.value == "stopped"
        assert StrategyStatus.FAILED.value == "failed"
        assert StrategyStatus.PROMOTING.value == "promoting"
        assert StrategyStatus.DEMOTING.value == "demoting"
    
    def test_promotion_reason_values(self):
        """Test PromotionReason enum values."""
        assert PromotionReason.SUPERIOR_SHARPE.value == "superior_sharpe"
        assert PromotionReason.SUPERIOR_CALMAR.value == "superior_calmar"
        assert PromotionReason.LOWER_DRAWDOWN.value == "lower_drawdown"
        assert PromotionReason.HIGHER_RETURNS.value == "higher_returns"
        assert PromotionReason.MANUAL_OVERRIDE.value == "manual_override"
        assert PromotionReason.CHAMPION_FAILURE.value == "champion_failure"
