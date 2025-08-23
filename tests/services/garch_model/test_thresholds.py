"""Unit tests for dynamic threshold calculator."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.garch_model.thresholds import (
    DynamicThresholdCalculator,
    MultiPairThresholdManager,
    ThresholdSignal,
    AdaptiveThresholds
)
from services.garch_model.model import GARCHForecast


class TestDynamicThresholdCalculator:
    """Test cases for DynamicThresholdCalculator class."""
    
    def test_initialization(self):
        """Test threshold calculator initialization with default parameters."""
        calculator = DynamicThresholdCalculator()
        
        assert calculator.base_entry_threshold == 2.0
        assert calculator.base_exit_threshold == 0.5
        assert calculator.volatility_adjustment_factor == 0.5
        assert calculator.min_threshold == 1.0
        assert calculator.max_threshold == 4.0
        assert calculator.lookback_window == 50
        assert len(calculator.spread_history) == 0
        assert len(calculator.volatility_history) == 0
        assert calculator.current_mean == 0.0
        assert calculator.current_std == 1.0
    
    def test_initialization_with_custom_parameters(self):
        """Test threshold calculator initialization with custom parameters."""
        calculator = DynamicThresholdCalculator(
            base_entry_threshold=2.5,
            base_exit_threshold=0.7,
            volatility_adjustment_factor=0.3,
            min_threshold=1.5,
            max_threshold=3.5,
            lookback_window=30
        )
        
        assert calculator.base_entry_threshold == 2.5
        assert calculator.base_exit_threshold == 0.7
        assert calculator.volatility_adjustment_factor == 0.3
        assert calculator.min_threshold == 1.5
        assert calculator.max_threshold == 3.5
        assert calculator.lookback_window == 30
    
    def test_update_statistics_single_observation(self):
        """Test updating statistics with single observation."""
        calculator = DynamicThresholdCalculator()
        timestamp = datetime.utcnow()
        
        calculator.update_statistics(1.5, 0.1, timestamp)
        
        assert len(calculator.spread_history) == 1
        assert len(calculator.volatility_history) == 1
        assert len(calculator.timestamp_history) == 1
        assert calculator.spread_history[0] == 1.5
        assert calculator.volatility_history[0] == 0.1
        assert calculator.timestamp_history[0] == timestamp
    
    def test_update_statistics_multiple_observations(self):
        """Test updating statistics with multiple observations."""
        calculator = DynamicThresholdCalculator()
        
        # Add multiple observations
        spread_values = [1.0, 1.2, 0.8, 1.1, 0.9]
        volatility_values = [0.1, 0.12, 0.08, 0.11, 0.09]
        
        for spread, vol in zip(spread_values, volatility_values):
            calculator.update_statistics(spread, vol)
        
        # Check statistics were updated
        assert len(calculator.spread_history) == 5
        assert len(calculator.volatility_history) == 5
        assert calculator.current_mean == np.mean(spread_values)
        assert calculator.current_std == np.std(spread_values, ddof=1)
    
    def test_update_statistics_rolling_window(self):
        """Test that rolling window maintains correct size."""
        window_size = 3
        calculator = DynamicThresholdCalculator(lookback_window=window_size)
        
        # Add more observations than window size
        for i in range(5):
            calculator.update_statistics(float(i), float(i) * 0.1)
        
        # Should only keep last window_size observations
        assert len(calculator.spread_history) == window_size
        assert len(calculator.volatility_history) == window_size
        assert list(calculator.spread_history) == [2.0, 3.0, 4.0]
        assert list(calculator.volatility_history) == [0.2, 0.3, 0.4]
    
    def test_calculate_z_score(self):
        """Test z-score calculation."""
        calculator = DynamicThresholdCalculator()
        
        # Add data to establish mean and std
        spread_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in spread_values:
            calculator.update_statistics(value)
        
        # Calculate z-score for known value
        mean = np.mean(spread_values)
        std = np.std(spread_values, ddof=1)
        test_value = 6.0
        expected_z_score = (test_value - mean) / std
        
        z_score = calculator.calculate_z_score(test_value)
        assert abs(z_score - expected_z_score) < 1e-10
    
    def test_calculate_z_score_zero_std(self):
        """Test z-score calculation with zero standard deviation."""
        calculator = DynamicThresholdCalculator()
        
        # Add constant data (zero std)
        for _ in range(5):
            calculator.update_statistics(1.0)
        
        z_score = calculator.calculate_z_score(1.0)
        assert z_score == 0.0
    
    def test_classify_volatility_regime_insufficient_data(self):
        """Test volatility regime classification with insufficient data."""
        calculator = DynamicThresholdCalculator()
        
        # Add insufficient volatility data
        for i in range(5):
            calculator.update_statistics(float(i), float(i) * 0.1)
        
        regime = calculator.classify_volatility_regime(0.2)
        assert regime == 'normal'
    
    def test_classify_volatility_regime_sufficient_data(self):
        """Test volatility regime classification with sufficient data."""
        calculator = DynamicThresholdCalculator()
        
        # Add sufficient volatility data
        volatility_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for i, vol in enumerate(volatility_values):
            calculator.update_statistics(float(i), vol)
        
        # Test different regimes
        low_vol = 0.1  # Should be low (below 25th percentile)
        high_vol = 0.45  # Should be high (above 75th percentile)
        normal_vol = 0.25  # Should be normal (between percentiles)
        
        assert calculator.classify_volatility_regime(low_vol) == 'low'
        assert calculator.classify_volatility_regime(high_vol) == 'high'
        assert calculator.classify_volatility_regime(normal_vol) == 'normal'
    
    def test_calculate_adaptive_thresholds(self):
        """Test adaptive threshold calculation."""
        calculator = DynamicThresholdCalculator(
            base_entry_threshold=2.0,
            base_exit_threshold=0.5,
            volatility_adjustment_factor=0.5
        )
        
        # Add volatility history
        volatility_values = [0.1, 0.15, 0.2, 0.25, 0.3]
        for i, vol in enumerate(volatility_values):
            calculator.update_statistics(float(i), vol)
        
        # Test with higher volatility (should increase thresholds)
        high_vol_forecast = 0.4
        thresholds = calculator.calculate_adaptive_thresholds(high_vol_forecast)
        
        assert isinstance(thresholds, AdaptiveThresholds)
        assert thresholds.entry_long < 0  # Should be negative
        assert thresholds.entry_short > 0  # Should be positive
        assert abs(thresholds.entry_long) == thresholds.entry_short  # Should be symmetric
        assert thresholds.exit < abs(thresholds.entry_long)  # Exit should be smaller
        assert thresholds.volatility_regime in ['low', 'normal', 'high']
        assert 0 <= thresholds.confidence <= 1
    
    def test_calculate_adaptive_thresholds_bounds(self):
        """Test that adaptive thresholds respect bounds."""
        calculator = DynamicThresholdCalculator(
            base_entry_threshold=2.0,
            min_threshold=1.5,
            max_threshold=3.0,
            volatility_adjustment_factor=2.0  # High adjustment factor
        )
        
        # Add volatility history
        for i in range(10):
            calculator.update_statistics(float(i), 0.1)
        
        # Test with very high volatility
        very_high_vol = 1.0  # Much higher than history
        thresholds = calculator.calculate_adaptive_thresholds(very_high_vol)
        
        # Should be capped at max_threshold
        assert abs(thresholds.entry_long) <= calculator.max_threshold
        assert thresholds.entry_short <= calculator.max_threshold
    
    def test_generate_signal_no_position_hold(self):
        """Test signal generation with no position and no signal."""
        calculator = DynamicThresholdCalculator()
        
        # Add neutral data
        for i in range(10):
            calculator.update_statistics(1.0, 0.1)  # Constant spread
        
        # Create GARCH forecast
        garch_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        # Generate signal for neutral spread (should be hold)
        signal = calculator.generate_signal("BTCETH", 1.0, garch_forecast)
        
        assert isinstance(signal, ThresholdSignal)
        assert signal.pair_id == "BTCETH"
        assert signal.spread_value == 1.0
        assert signal.z_score == 0.0  # Neutral
        assert signal.signal_type == 'hold'
        assert signal.signal_strength == 0.0
    
    def test_generate_signal_entry_long(self):
        """Test signal generation for long entry."""
        calculator = DynamicThresholdCalculator(base_entry_threshold=1.5)
        
        # Add data with higher mean
        for i in range(10):
            calculator.update_statistics(2.0, 0.1)
        
        garch_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        # Generate signal for low spread (should trigger long entry)
        low_spread = 0.0  # Much lower than mean of 2.0
        signal = calculator.generate_signal("BTCETH", low_spread, garch_forecast)
        
        assert signal.signal_type == 'entry_long'
        assert signal.signal_strength > 0
        assert signal.z_score < 0  # Below mean
    
    def test_generate_signal_entry_short(self):
        """Test signal generation for short entry."""
        calculator = DynamicThresholdCalculator(base_entry_threshold=1.5)
        
        # Add data with lower mean
        for i in range(10):
            calculator.update_statistics(1.0, 0.1)
        
        garch_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        # Generate signal for high spread (should trigger short entry)
        high_spread = 4.0  # Much higher than mean of 1.0
        signal = calculator.generate_signal("BTCETH", high_spread, garch_forecast)
        
        assert signal.signal_type == 'entry_short'
        assert signal.signal_strength > 0
        assert signal.z_score > 0  # Above mean
    
    def test_generate_signal_exit_from_long(self):
        """Test signal generation for exit from long position."""
        calculator = DynamicThresholdCalculator(base_exit_threshold=0.5)
        
        # Add data
        for i in range(10):
            calculator.update_statistics(1.0, 0.1)
        
        garch_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        # Generate signal with current long position and spread near mean
        signal = calculator.generate_signal(
            "BTCETH", 1.0, garch_forecast, current_position='long'
        )
        
        assert signal.signal_type == 'exit'
        assert signal.signal_strength > 0
    
    def test_get_statistics(self):
        """Test getting current statistics."""
        calculator = DynamicThresholdCalculator()
        
        # Add some data
        for i in range(5):
            calculator.update_statistics(float(i), float(i) * 0.1)
        
        stats = calculator.get_statistics()
        
        required_keys = [
            'current_mean', 'current_std', 'volatility_p25', 'volatility_p75',
            'spread_history_size', 'volatility_history_size',
            'base_entry_threshold', 'base_exit_threshold',
            'volatility_adjustment_factor'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['spread_history_size'] == 5
        assert stats['volatility_history_size'] == 5
        assert stats['base_entry_threshold'] == 2.0
    
    def test_reset(self):
        """Test resetting the calculator."""
        calculator = DynamicThresholdCalculator()
        
        # Add data
        for i in range(5):
            calculator.update_statistics(float(i), float(i) * 0.1)
        
        # Verify data exists
        assert len(calculator.spread_history) == 5
        assert calculator.current_mean != 0.0
        
        # Reset
        calculator.reset()
        
        # Verify reset
        assert len(calculator.spread_history) == 0
        assert len(calculator.volatility_history) == 0
        assert len(calculator.timestamp_history) == 0
        assert calculator.current_mean == 0.0
        assert calculator.current_std == 1.0


class TestMultiPairThresholdManager:
    """Test cases for MultiPairThresholdManager class."""
    
    def test_initialization(self):
        """Test multi-pair threshold manager initialization."""
        manager = MultiPairThresholdManager()
        
        assert manager.default_base_entry_threshold == 2.0
        assert manager.default_base_exit_threshold == 0.5
        assert manager.default_volatility_adjustment_factor == 0.5
        assert len(manager.calculators) == 0
        assert len(manager.pair_configs) == 0
        assert len(manager.current_positions) == 0
    
    def test_add_pair_default_params(self):
        """Test adding pair with default parameters."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        assert pair_id in manager.calculators
        assert pair_id in manager.pair_configs
        assert pair_id in manager.current_positions
        assert isinstance(manager.calculators[pair_id], DynamicThresholdCalculator)
        assert manager.current_positions[pair_id] is None
        
        config = manager.pair_configs[pair_id]
        assert config['base_entry_threshold'] == 2.0
        assert config['base_exit_threshold'] == 0.5
    
    def test_add_pair_custom_params(self):
        """Test adding pair with custom parameters."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        custom_params = {
            'base_entry_threshold': 2.5,
            'base_exit_threshold': 0.7,
            'volatility_adjustment_factor': 0.3,
            'lookback_window': 30
        }
        
        manager.add_pair(pair_id, **custom_params)
        
        config = manager.pair_configs[pair_id]
        assert config['base_entry_threshold'] == 2.5
        assert config['base_exit_threshold'] == 0.7
        assert config['volatility_adjustment_factor'] == 0.3
        assert config['lookback_window'] == 30
    
    def test_generate_signal_valid_pair(self):
        """Test generating signal for valid pair."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Mock the calculator's generate_signal method
        mock_signal = ThresholdSignal(
            pair_id=pair_id,
            spread_value=1.0,
            z_score=0.0,
            volatility_forecast=0.1,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5,
            signal_type='hold',
            signal_strength=0.0,
            confidence_level=0.8,
            volatility_regime='normal',
            timestamp=datetime.utcnow()
        )
        
        manager.calculators[pair_id].generate_signal = Mock(return_value=mock_signal)
        
        garch_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        result = manager.generate_signal(pair_id, 1.0, garch_forecast)
        
        assert result == mock_signal
        manager.calculators[pair_id].generate_signal.assert_called_once()
    
    def test_generate_signal_invalid_pair(self):
        """Test generating signal for unknown pair."""
        manager = MultiPairThresholdManager()
        
        garch_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        result = manager.generate_signal("UNKNOWN", 1.0, garch_forecast)
        assert result is None
    
    def test_position_tracking_entry_signals(self):
        """Test position tracking with entry signals."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Mock entry long signal
        mock_signal_long = ThresholdSignal(
            pair_id=pair_id,
            spread_value=1.0,
            z_score=-2.5,
            volatility_forecast=0.1,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5,
            signal_type='entry_long',
            signal_strength=0.8,
            confidence_level=0.8,
            volatility_regime='normal',
            timestamp=datetime.utcnow()
        )
        
        manager.calculators[pair_id].generate_signal = Mock(return_value=mock_signal_long)
        
        garch_forecast = Mock()
        manager.generate_signal(pair_id, 1.0, garch_forecast)
        
        # Position should be updated to 'long'
        assert manager.current_positions[pair_id] == 'long'
    
    def test_position_tracking_exit_signals(self):
        """Test position tracking with exit signals."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Set initial position
        manager.current_positions[pair_id] = 'long'
        
        # Mock exit signal
        mock_signal_exit = ThresholdSignal(
            pair_id=pair_id,
            spread_value=1.0,
            z_score=0.0,
            volatility_forecast=0.1,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5,
            signal_type='exit',
            signal_strength=0.6,
            confidence_level=0.8,
            volatility_regime='normal',
            timestamp=datetime.utcnow()
        )
        
        manager.calculators[pair_id].generate_signal = Mock(return_value=mock_signal_exit)
        
        garch_forecast = Mock()
        manager.generate_signal(pair_id, 1.0, garch_forecast)
        
        # Position should be cleared
        assert manager.current_positions[pair_id] is None
    
    def test_update_position_manually(self):
        """Test manually updating position."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Update position manually
        manager.update_position(pair_id, 'short')
        assert manager.current_positions[pair_id] == 'short'
        
        # Clear position
        manager.update_position(pair_id, None)
        assert manager.current_positions[pair_id] is None
    
    def test_get_current_thresholds(self):
        """Test getting current thresholds for a pair."""
        manager = MultiPairThresholdManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Mock the calculator's get_current_thresholds method
        mock_thresholds = AdaptiveThresholds(
            entry_long=-2.0,
            entry_short=2.0,
            exit=0.5,
            stop_loss_long=-3.0,
            stop_loss_short=3.0,
            volatility_regime='normal',
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        manager.calculators[pair_id].get_current_thresholds = Mock(return_value=mock_thresholds)
        
        result = manager.get_current_thresholds(pair_id, 0.1)
        assert result == mock_thresholds
        
        # Test unknown pair
        result = manager.get_current_thresholds("UNKNOWN", 0.1)
        assert result is None
    
    def test_get_all_statistics(self):
        """Test getting statistics for all pairs."""
        manager = MultiPairThresholdManager()
        
        pairs = ["BTCETH", "BTCADA"]
        for pair_id in pairs:
            manager.add_pair(pair_id)
            manager.current_positions[pair_id] = 'long' if pair_id == "BTCETH" else None
        
        # Mock statistics
        for pair_id in pairs:
            mock_stats = {
                'current_mean': 1.0,
                'current_std': 0.1,
                'spread_history_size': 10
            }
            manager.calculators[pair_id].get_statistics = Mock(return_value=mock_stats)
        
        results = manager.get_all_statistics()
        
        assert len(results) == 2
        for pair_id in pairs:
            assert pair_id in results
            assert 'current_position' in results[pair_id]
            assert results[pair_id]['current_mean'] == 1.0
        
        assert results["BTCETH"]['current_position'] == 'long'
        assert results["BTCADA"]['current_position'] is None


class TestThresholdSignal:
    """Test cases for ThresholdSignal dataclass."""
    
    def test_creation(self):
        """Test creating a ThresholdSignal."""
        timestamp = datetime.utcnow()
        
        signal = ThresholdSignal(
            pair_id="BTCETH",
            spread_value=1.0,
            z_score=-2.5,
            volatility_forecast=0.1,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5,
            signal_type='entry_long',
            signal_strength=0.8,
            confidence_level=0.9,
            volatility_regime='normal',
            timestamp=timestamp
        )
        
        assert signal.pair_id == "BTCETH"
        assert signal.spread_value == 1.0
        assert signal.z_score == -2.5
        assert signal.volatility_forecast == 0.1
        assert signal.entry_threshold_long == -2.0
        assert signal.entry_threshold_short == 2.0
        assert signal.exit_threshold == 0.5
        assert signal.signal_type == 'entry_long'
        assert signal.signal_strength == 0.8
        assert signal.confidence_level == 0.9
        assert signal.volatility_regime == 'normal'
        assert signal.timestamp == timestamp


class TestAdaptiveThresholds:
    """Test cases for AdaptiveThresholds dataclass."""
    
    def test_creation(self):
        """Test creating AdaptiveThresholds."""
        timestamp = datetime.utcnow()
        
        thresholds = AdaptiveThresholds(
            entry_long=-2.0,
            entry_short=2.0,
            exit=0.5,
            stop_loss_long=-3.0,
            stop_loss_short=3.0,
            volatility_regime='high',
            confidence=0.85,
            timestamp=timestamp
        )
        
        assert thresholds.entry_long == -2.0
        assert thresholds.entry_short == 2.0
        assert thresholds.exit == 0.5
        assert thresholds.stop_loss_long == -3.0
        assert thresholds.stop_loss_short == 3.0
        assert thresholds.volatility_regime == 'high'
        assert thresholds.confidence == 0.85
        assert thresholds.timestamp == timestamp


@pytest.fixture
def sample_threshold_calculator():
    """Fixture providing a sample threshold calculator with data."""
    calculator = DynamicThresholdCalculator()
    
    # Add sample data
    np.random.seed(42)
    for i in range(20):
        spread = 1.0 + np.random.normal(0, 0.2)
        volatility = 0.1 + np.random.normal(0, 0.02)
        calculator.update_statistics(spread, max(0.01, volatility))
    
    return calculator


@pytest.fixture
def sample_threshold_manager():
    """Fixture providing a sample threshold manager with pairs."""
    manager = MultiPairThresholdManager()
    
    pairs = ["BTCETH", "BTCADA"]
    for pair_id in pairs:
        manager.add_pair(pair_id)
    
    return manager


class TestThresholdIntegration:
    """Integration tests for threshold components."""
    
    def test_full_threshold_workflow(self, sample_threshold_calculator):
        """Test complete threshold calculation workflow."""
        calculator = sample_threshold_calculator
        
        # Should have data
        assert len(calculator.spread_history) == 20
        assert len(calculator.volatility_history) == 20
        
        # Create GARCH forecast
        garch_forecast = GARCHForecast(
            volatility_forecast=0.12,
            variance_forecast=0.0144,
            confidence_interval_lower=0.10,
            confidence_interval_upper=0.14,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        # Generate signal
        signal = calculator.generate_signal("BTCETH", 1.5, garch_forecast)
        
        assert isinstance(signal, ThresholdSignal)
        assert signal.pair_id == "BTCETH"
        assert signal.spread_value == 1.5
        assert signal.volatility_forecast == 0.12
        assert signal.signal_type in ['entry_long', 'entry_short', 'exit', 'hold']
        assert 0 <= signal.signal_strength <= 1
        assert 0 <= signal.confidence_level <= 1
        assert signal.volatility_regime in ['low', 'normal', 'high']
    
    def test_multi_pair_threshold_workflow(self, sample_threshold_manager):
        """Test workflow with multiple pairs."""
        manager = sample_threshold_manager
        
        # Create GARCH forecasts for both pairs
        garch_forecast_1 = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        garch_forecast_2 = GARCHForecast(
            volatility_forecast=0.15,
            variance_forecast=0.0225,
            confidence_interval_lower=0.12,
            confidence_interval_upper=0.18,
            forecast_horizon=1,
            model_aic=210.0,
            model_bic=215.0,
            timestamp=datetime.utcnow()
        )
        
        # Add some data to calculators first
        for pair_id in ["BTCETH", "BTCADA"]:
            for i in range(10):
                manager.calculators[pair_id].update_statistics(1.0 + i * 0.1, 0.1)
        
        # Generate signals
        signal_1 = manager.generate_signal("BTCETH", 1.0, garch_forecast_1)
        signal_2 = manager.generate_signal("BTCADA", 2.0, garch_forecast_2)
        
        assert signal_1 is not None
        assert signal_2 is not None
        assert signal_1.pair_id == "BTCETH"
        assert signal_2.pair_id == "BTCADA"
        
        # Get statistics
        stats = manager.get_all_statistics()
        assert len(stats) == 2
        assert "BTCETH" in stats
        assert "BTCADA" in stats
