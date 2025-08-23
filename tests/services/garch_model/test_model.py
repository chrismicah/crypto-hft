"""Unit tests for GARCH model implementation."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from services.garch_model.model import (
    RollingGARCHModel,
    MultiPairGARCHManager,
    GARCHForecast,
    GARCHModelState
)


class TestRollingGARCHModel:
    """Test cases for RollingGARCHModel class."""
    
    def test_initialization(self):
        """Test GARCH model initialization with default parameters."""
        model = RollingGARCHModel()
        
        assert model.window_size == 252
        assert model.min_observations == 50
        assert model.refit_frequency == 10
        assert model.confidence_level == 0.95
        assert len(model.spread_data) == 0
        assert len(model.timestamps) == 0
        assert model.model is None
        assert model.fitted_model is None
        assert model.total_observations == 0
    
    def test_initialization_with_custom_parameters(self):
        """Test GARCH model initialization with custom parameters."""
        window_size = 100
        min_obs = 30
        refit_freq = 5
        confidence = 0.99
        
        model = RollingGARCHModel(
            window_size=window_size,
            min_observations=min_obs,
            refit_frequency=refit_freq,
            confidence_level=confidence
        )
        
        assert model.window_size == window_size
        assert model.min_observations == min_obs
        assert model.refit_frequency == refit_freq
        assert model.confidence_level == confidence
    
    def test_add_observation_valid(self):
        """Test adding valid observations."""
        model = RollingGARCHModel()
        timestamp = datetime.utcnow()
        
        # Add single observation
        success = model.add_observation(1.5, timestamp)
        
        assert success is True
        assert len(model.spread_data) == 1
        assert len(model.timestamps) == 1
        assert model.spread_data[0] == 1.5
        assert model.timestamps[0] == timestamp
        assert model.total_observations == 1
        assert model.observations_since_fit == 1
    
    def test_add_observation_invalid(self):
        """Test adding invalid observations."""
        model = RollingGARCHModel()
        
        # Test with NaN
        success = model.add_observation(np.nan)
        assert success is False
        assert len(model.spread_data) == 0
        
        # Test with infinity
        success = model.add_observation(np.inf)
        assert success is False
        assert len(model.spread_data) == 0
    
    def test_add_observation_rolling_window(self):
        """Test that rolling window maintains correct size."""
        window_size = 5
        model = RollingGARCHModel(window_size=window_size)
        
        # Add more observations than window size
        for i in range(10):
            model.add_observation(float(i))
        
        # Should only keep last window_size observations
        assert len(model.spread_data) == window_size
        assert list(model.spread_data) == [5.0, 6.0, 7.0, 8.0, 9.0]
        assert model.total_observations == 10
    
    def test_should_refit_logic(self):
        """Test the refit decision logic."""
        model = RollingGARCHModel(min_observations=5, refit_frequency=3)
        
        # Not enough observations
        for i in range(4):
            model.add_observation(float(i))
        assert model.should_refit() is False
        
        # Enough observations, no fitted model
        model.add_observation(4.0)
        assert model.should_refit() is True
        
        # Simulate fitted model
        model.fitted_model = Mock()
        model.observations_since_fit = 2
        assert model.should_refit() is False
        
        # Enough new observations since fit
        model.observations_since_fit = 3
        assert model.should_refit() is True
    
    def test_fit_model_insufficient_data(self):
        """Test fitting with insufficient data."""
        model = RollingGARCHModel(min_observations=10)
        
        # Add insufficient data
        for i in range(5):
            model.add_observation(float(i))
        
        success = model.fit_model(force=True)
        assert success is False
        assert model.fitted_model is None
    
    def test_fit_model_constant_data(self):
        """Test fitting with constant data (zero variance)."""
        model = RollingGARCHModel(min_observations=10)
        
        # Add constant data
        for i in range(15):
            model.add_observation(1.0)  # All same value
        
        success = model.fit_model(force=True)
        assert success is False  # Should fail due to zero variance
    
    @patch('services.garch_model.model.arch_model')
    def test_fit_model_success(self, mock_arch_model):
        """Test successful model fitting."""
        model = RollingGARCHModel(min_observations=10)
        
        # Add varying data
        np.random.seed(42)
        for i in range(15):
            model.add_observation(1.0 + np.random.normal(0, 0.1))
        
        # Mock ARCH model
        mock_fitted = Mock()
        mock_fitted.params = {
            'omega': 0.001,
            'alpha[1]': 0.1,
            'beta[1]': 0.8
        }
        mock_fitted.loglikelihood = -100.0
        mock_fitted.aic = 206.0
        mock_fitted.bic = 210.0
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_arch_model.return_value = mock_model
        
        success = model.fit_model(force=True)
        
        assert success is True
        assert model.fitted_model == mock_fitted
        assert model.model_state is not None
        assert model.model_state.omega == 0.001
        assert model.model_state.alpha == 0.1
        assert model.model_state.beta == 0.8
        assert model.successful_fits == 1
        assert model.observations_since_fit == 0
    
    @patch('services.garch_model.model.arch_model')
    def test_fit_model_failure(self, mock_arch_model):
        """Test model fitting failure."""
        model = RollingGARCHModel(min_observations=10)
        
        # Add data
        for i in range(15):
            model.add_observation(1.0 + i * 0.1)
        
        # Mock ARCH model to raise exception
        mock_arch_model.side_effect = Exception("Fitting failed")
        
        success = model.fit_model(force=True)
        
        assert success is False
        assert model.fitted_model is None
        assert model.failed_fits == 1
    
    def test_forecast_volatility_no_model(self):
        """Test forecasting without fitted model."""
        model = RollingGARCHModel()
        
        forecast = model.forecast_volatility()
        assert forecast is None
    
    @patch('services.garch_model.model.arch_model')
    def test_forecast_volatility_success(self, mock_arch_model):
        """Test successful volatility forecasting."""
        model = RollingGARCHModel(min_observations=10)
        
        # Add data and fit model
        np.random.seed(42)
        for i in range(15):
            model.add_observation(1.0 + np.random.normal(0, 0.1))
        
        # Mock fitted model
        mock_fitted = Mock()
        mock_fitted.params = {
            'omega': 0.001,
            'alpha[1]': 0.1,
            'beta[1]': 0.8
        }
        mock_fitted.loglikelihood = -100.0
        mock_fitted.aic = 206.0
        mock_fitted.bic = 210.0
        mock_fitted.resid = np.random.normal(0, 0.1, 15)
        
        # Mock forecast
        mock_forecast = Mock()
        mock_forecast.variance = pd.DataFrame([[0.01]], columns=[0])
        mock_fitted.forecast.return_value = mock_forecast
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_arch_model.return_value = mock_model
        
        # Fit and forecast
        model.fit_model(force=True)
        forecast = model.forecast_volatility(horizon=1)
        
        assert forecast is not None
        assert isinstance(forecast, GARCHForecast)
        assert forecast.volatility_forecast == np.sqrt(0.01)
        assert forecast.variance_forecast == 0.01
        assert forecast.forecast_horizon == 1
        assert forecast.confidence_interval_lower >= 0  # Volatility can't be negative
    
    def test_get_model_diagnostics(self):
        """Test getting model diagnostics."""
        model = RollingGARCHModel()
        
        # Add some data
        for i in range(5):
            model.add_observation(float(i))
        
        diagnostics = model.get_model_diagnostics()
        
        required_keys = [
            'total_observations', 'window_size', 'successful_fits',
            'failed_fits', 'observations_since_fit', 'last_fit_time',
            'model_fitted'
        ]
        
        for key in required_keys:
            assert key in diagnostics
        
        assert diagnostics['total_observations'] == 5
        assert diagnostics['window_size'] == 5
        assert diagnostics['model_fitted'] is False
    
    def test_get_current_data(self):
        """Test getting current data as DataFrame."""
        model = RollingGARCHModel()
        
        # Empty model
        df = model.get_current_data()
        assert df.empty
        
        # Add data
        timestamps = [datetime.utcnow() + timedelta(minutes=i) for i in range(3)]
        values = [1.0, 1.1, 1.2]
        
        for value, timestamp in zip(values, timestamps):
            model.add_observation(value, timestamp)
        
        df = model.get_current_data()
        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'spread' in df.columns
        assert df['spread'].tolist() == values
    
    def test_reset(self):
        """Test resetting the model."""
        model = RollingGARCHModel()
        
        # Add data and simulate fitted model
        for i in range(5):
            model.add_observation(float(i))
        
        model.fitted_model = Mock()
        model.total_observations = 10
        model.successful_fits = 2
        
        # Reset
        model.reset()
        
        assert len(model.spread_data) == 0
        assert len(model.timestamps) == 0
        assert model.fitted_model is None
        assert model.total_observations == 0
        assert model.successful_fits == 0


class TestMultiPairGARCHManager:
    """Test cases for MultiPairGARCHManager class."""
    
    def test_initialization(self):
        """Test multi-pair manager initialization."""
        manager = MultiPairGARCHManager()
        
        assert manager.default_window_size == 252
        assert manager.default_min_observations == 50
        assert manager.default_refit_frequency == 10
        assert len(manager.models) == 0
        assert len(manager.pair_configs) == 0
    
    def test_add_pair_default_params(self):
        """Test adding pair with default parameters."""
        manager = MultiPairGARCHManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        assert pair_id in manager.models
        assert pair_id in manager.pair_configs
        assert isinstance(manager.models[pair_id], RollingGARCHModel)
        
        config = manager.pair_configs[pair_id]
        assert config['window_size'] == 252
        assert config['min_observations'] == 50
        assert config['refit_frequency'] == 10
    
    def test_add_pair_custom_params(self):
        """Test adding pair with custom parameters."""
        manager = MultiPairGARCHManager()
        
        pair_id = "BTCETH"
        custom_params = {
            'window_size': 100,
            'min_observations': 30,
            'refit_frequency': 5
        }
        
        manager.add_pair(pair_id, **custom_params)
        
        config = manager.pair_configs[pair_id]
        assert config['window_size'] == 100
        assert config['min_observations'] == 30
        assert config['refit_frequency'] == 5
    
    def test_add_observation_valid_pair(self):
        """Test adding observation to valid pair."""
        manager = MultiPairGARCHManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        timestamp = datetime.utcnow()
        success = manager.add_observation(pair_id, 1.5, timestamp)
        
        assert success is True
        assert len(manager.models[pair_id].spread_data) == 1
    
    def test_add_observation_invalid_pair(self):
        """Test adding observation to unknown pair."""
        manager = MultiPairGARCHManager()
        
        success = manager.add_observation("UNKNOWN", 1.5)
        assert success is False
    
    def test_forecast_volatility_valid_pair(self):
        """Test forecasting for valid pair."""
        manager = MultiPairGARCHManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Mock the model's forecast method
        mock_forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=datetime.utcnow()
        )
        
        manager.models[pair_id].forecast_volatility = Mock(return_value=mock_forecast)
        
        result = manager.forecast_volatility(pair_id, horizon=1)
        assert result == mock_forecast
    
    def test_forecast_volatility_invalid_pair(self):
        """Test forecasting for unknown pair."""
        manager = MultiPairGARCHManager()
        
        result = manager.forecast_volatility("UNKNOWN")
        assert result is None
    
    def test_get_all_forecasts(self):
        """Test getting forecasts for all pairs."""
        manager = MultiPairGARCHManager()
        
        pairs = ["BTCETH", "BTCADA"]
        for pair_id in pairs:
            manager.add_pair(pair_id)
        
        # Mock forecasts
        mock_forecasts = {}
        for pair_id in pairs:
            mock_forecast = GARCHForecast(
                volatility_forecast=0.1,
                variance_forecast=0.01,
                confidence_interval_lower=0.08,
                confidence_interval_upper=0.12,
                forecast_horizon=1,
                model_aic=200.0,
                model_bic=205.0,
                timestamp=datetime.utcnow()
            )
            mock_forecasts[pair_id] = mock_forecast
            manager.models[pair_id].forecast_volatility = Mock(return_value=mock_forecast)
        
        results = manager.get_all_forecasts()
        
        assert len(results) == 2
        for pair_id in pairs:
            assert pair_id in results
            assert results[pair_id] == mock_forecasts[pair_id]
    
    def test_get_pair_diagnostics(self):
        """Test getting diagnostics for specific pair."""
        manager = MultiPairGARCHManager()
        
        pair_id = "BTCETH"
        manager.add_pair(pair_id)
        
        # Mock diagnostics
        mock_diagnostics = {'total_observations': 10, 'model_fitted': True}
        manager.models[pair_id].get_model_diagnostics = Mock(return_value=mock_diagnostics)
        
        result = manager.get_pair_diagnostics(pair_id)
        assert result == mock_diagnostics
        
        # Test unknown pair
        result = manager.get_pair_diagnostics("UNKNOWN")
        assert result is None
    
    def test_get_all_diagnostics(self):
        """Test getting diagnostics for all pairs."""
        manager = MultiPairGARCHManager()
        
        pairs = ["BTCETH", "BTCADA"]
        for pair_id in pairs:
            manager.add_pair(pair_id)
        
        # Mock diagnostics
        for pair_id in pairs:
            mock_diagnostics = {'total_observations': 10, 'pair_id': pair_id}
            manager.models[pair_id].get_model_diagnostics = Mock(return_value=mock_diagnostics)
        
        results = manager.get_all_diagnostics()
        
        assert len(results) == 2
        for pair_id in pairs:
            assert pair_id in results
            assert results[pair_id]['pair_id'] == pair_id


class TestGARCHForecast:
    """Test cases for GARCHForecast dataclass."""
    
    def test_creation(self):
        """Test creating a GARCHForecast."""
        timestamp = datetime.utcnow()
        
        forecast = GARCHForecast(
            volatility_forecast=0.1,
            variance_forecast=0.01,
            confidence_interval_lower=0.08,
            confidence_interval_upper=0.12,
            forecast_horizon=1,
            model_aic=200.0,
            model_bic=205.0,
            timestamp=timestamp
        )
        
        assert forecast.volatility_forecast == 0.1
        assert forecast.variance_forecast == 0.01
        assert forecast.confidence_interval_lower == 0.08
        assert forecast.confidence_interval_upper == 0.12
        assert forecast.forecast_horizon == 1
        assert forecast.model_aic == 200.0
        assert forecast.model_bic == 205.0
        assert forecast.timestamp == timestamp


class TestGARCHModelState:
    """Test cases for GARCHModelState dataclass."""
    
    def test_creation(self):
        """Test creating a GARCHModelState."""
        timestamp = datetime.utcnow()
        
        state = GARCHModelState(
            omega=0.001,
            alpha=0.1,
            beta=0.8,
            log_likelihood=-100.0,
            aic=206.0,
            bic=210.0,
            fitted_at=timestamp,
            n_observations=100
        )
        
        assert state.omega == 0.001
        assert state.alpha == 0.1
        assert state.beta == 0.8
        assert state.log_likelihood == -100.0
        assert state.aic == 206.0
        assert state.bic == 210.0
        assert state.fitted_at == timestamp
        assert state.n_observations == 100


@pytest.fixture
def sample_garch_model():
    """Fixture providing a sample GARCH model with data."""
    model = RollingGARCHModel(min_observations=10)
    
    # Add sample data
    np.random.seed(42)
    for i in range(15):
        model.add_observation(1.0 + np.random.normal(0, 0.1))
    
    return model


@pytest.fixture
def sample_garch_manager():
    """Fixture providing a sample GARCH manager with pairs."""
    manager = MultiPairGARCHManager()
    
    pairs = ["BTCETH", "BTCADA"]
    for pair_id in pairs:
        manager.add_pair(pair_id)
    
    return manager


class TestGARCHIntegration:
    """Integration tests for GARCH components."""
    
    def test_full_workflow_single_model(self, sample_garch_model):
        """Test complete workflow for single GARCH model."""
        model = sample_garch_model
        
        # Should have data
        assert len(model.spread_data) == 15
        assert model.total_observations == 15
        
        # Should be able to fit (with mocking)
        with patch('services.garch_model.model.arch_model') as mock_arch:
            mock_fitted = Mock()
            mock_fitted.params = {
                'omega': 0.001,
                'alpha[1]': 0.1,
                'beta[1]': 0.8
            }
            mock_fitted.loglikelihood = -100.0
            mock_fitted.aic = 206.0
            mock_fitted.bic = 210.0
            mock_fitted.resid = np.random.normal(0, 0.1, 15)
            
            mock_forecast_result = Mock()
            mock_forecast_result.variance = pd.DataFrame([[0.01]], columns=[0])
            mock_fitted.forecast.return_value = mock_forecast_result
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arch.return_value = mock_model
            
            # Fit model
            success = model.fit_model(force=True)
            assert success is True
            
            # Generate forecast
            forecast = model.forecast_volatility()
            assert forecast is not None
            assert forecast.volatility_forecast > 0
            assert forecast.variance_forecast > 0
    
    def test_multi_pair_workflow(self, sample_garch_manager):
        """Test workflow with multiple pairs."""
        manager = sample_garch_manager
        
        # Add observations to both pairs
        pairs = ["BTCETH", "BTCADA"]
        
        for pair_id in pairs:
            for i in range(10):
                success = manager.add_observation(pair_id, 1.0 + i * 0.1)
                assert success is True
        
        # Check that data was added
        for pair_id in pairs:
            assert len(manager.models[pair_id].spread_data) == 10
        
        # Get diagnostics
        diagnostics = manager.get_all_diagnostics()
        assert len(diagnostics) == 2
        
        for pair_id in pairs:
            assert pair_id in diagnostics
            assert diagnostics[pair_id]['total_observations'] == 10
