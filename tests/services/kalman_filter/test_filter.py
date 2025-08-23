"""Unit tests for Kalman filter implementation."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.kalman_filter.filter import (
    DynamicHedgeRatioKalman,
    PairTradingKalmanFilter,
    KalmanState
)


class TestDynamicHedgeRatioKalman:
    """Test cases for DynamicHedgeRatioKalman class."""
    
    def test_initialization(self):
        """Test filter initialization with default parameters."""
        filter_obj = DynamicHedgeRatioKalman()
        
        assert filter_obj.state_dim == 2
        assert filter_obj.obs_dim == 1
        assert filter_obj.n_observations == 0
        assert filter_obj.last_update is None
        
        # Check initial state
        expected_state = np.array([1.0, 0.0])
        np.testing.assert_array_equal(filter_obj.state_mean, expected_state)
        
        # Check matrix dimensions
        assert filter_obj.transition_matrix.shape == (2, 2)
        assert filter_obj.observation_matrix.shape == (1, 2)
        assert filter_obj.process_covariance.shape == (2, 2)
        assert filter_obj.observation_covariance.shape == (1, 1)
    
    def test_initialization_with_custom_parameters(self):
        """Test filter initialization with custom parameters."""
        initial_ratio = 2.5
        process_var = 1e-4
        obs_var = 1e-2
        
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=initial_ratio,
            process_variance=process_var,
            observation_variance=obs_var
        )
        
        assert filter_obj.state_mean[0] == initial_ratio
        assert filter_obj.process_variance == process_var
        assert filter_obj.observation_variance == obs_var
        assert filter_obj.observation_covariance[0, 0] == obs_var
    
    def test_predict_step(self):
        """Test the prediction step of the Kalman filter."""
        filter_obj = DynamicHedgeRatioKalman(initial_hedge_ratio=1.0)
        
        # Set a known state
        filter_obj.state_mean = np.array([1.5, 0.1])
        filter_obj.state_covariance = np.eye(2) * 0.1
        
        pred_mean, pred_cov = filter_obj.predict()
        
        # Check prediction
        expected_mean = filter_obj.transition_matrix @ np.array([1.5, 0.1])
        np.testing.assert_array_almost_equal(pred_mean, expected_mean)
        
        # Check covariance prediction
        assert pred_cov.shape == (2, 2)
        assert np.all(np.linalg.eigvals(pred_cov) > 0)  # Positive definite
    
    def test_update_single_observation(self):
        """Test updating the filter with a single observation."""
        filter_obj = DynamicHedgeRatioKalman(initial_hedge_ratio=1.0)
        
        # Update with observation
        observation = 1.2
        timestamp = datetime.utcnow()
        result = filter_obj.update(observation, timestamp)
        
        # Check result structure
        required_keys = [
            'hedge_ratio', 'hedge_ratio_velocity', 'hedge_ratio_variance',
            'confidence_interval_95', 'innovation', 'innovation_variance',
            'log_likelihood', 'n_observations', 'timestamp'
        ]
        for key in required_keys:
            assert key in result
        
        # Check state updates
        assert filter_obj.n_observations == 1
        assert filter_obj.last_update == timestamp
        assert result['n_observations'] == 1
        
        # Hedge ratio should move towards observation
        assert abs(result['hedge_ratio'] - observation) < abs(1.0 - observation)
    
    def test_update_multiple_observations(self):
        """Test updating the filter with multiple observations."""
        filter_obj = DynamicHedgeRatioKalman(initial_hedge_ratio=1.0)
        
        observations = [1.1, 1.2, 1.15, 1.25, 1.18]
        results = []
        
        for i, obs in enumerate(observations):
            result = filter_obj.update(obs)
            results.append(result)
            
            # Check observation count
            assert result['n_observations'] == i + 1
            assert filter_obj.n_observations == i + 1
        
        # Hedge ratio should converge towards the mean of observations
        final_hedge_ratio = results[-1]['hedge_ratio']
        obs_mean = np.mean(observations)
        
        # Should be closer to observation mean than initial value
        assert abs(final_hedge_ratio - obs_mean) < abs(1.0 - obs_mean)
    
    def test_matrix_dimensions_consistency(self):
        """Test that all matrix dimensions are consistent."""
        filter_obj = DynamicHedgeRatioKalman()
        
        # Check transition matrix
        assert filter_obj.transition_matrix.shape == (filter_obj.state_dim, filter_obj.state_dim)
        
        # Check observation matrix
        assert filter_obj.observation_matrix.shape == (filter_obj.obs_dim, filter_obj.state_dim)
        
        # Check covariance matrices
        assert filter_obj.process_covariance.shape == (filter_obj.state_dim, filter_obj.state_dim)
        assert filter_obj.observation_covariance.shape == (filter_obj.obs_dim, filter_obj.obs_dim)
        
        # Check state vectors
        assert filter_obj.state_mean.shape == (filter_obj.state_dim,)
        assert filter_obj.state_covariance.shape == (filter_obj.state_dim, filter_obj.state_dim)
    
    def test_state_prediction_logic_known_example(self):
        """Test state prediction logic against a known example."""
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.0,
            process_variance=0.01,
            observation_variance=0.1
        )
        
        # Set known state
        filter_obj.state_mean = np.array([2.0, 0.5])
        filter_obj.state_covariance = np.array([[0.1, 0.0], [0.0, 0.1]])
        
        # Predict next state
        pred_mean, pred_cov = filter_obj.predict()
        
        # Manual calculation
        F = filter_obj.transition_matrix
        Q = filter_obj.process_covariance
        
        expected_mean = F @ np.array([2.0, 0.5])
        expected_cov = F @ np.array([[0.1, 0.0], [0.0, 0.1]]) @ F.T + Q
        
        np.testing.assert_array_almost_equal(pred_mean, expected_mean)
        np.testing.assert_array_almost_equal(pred_cov, expected_cov)
    
    def test_covariance_positive_definite(self):
        """Test that covariance matrices remain positive definite."""
        filter_obj = DynamicHedgeRatioKalman()
        
        # Update with several observations
        observations = [1.1, 0.9, 1.2, 0.8, 1.3]
        
        for obs in observations:
            filter_obj.update(obs)
            
            # Check that covariance is positive definite
            eigenvals = np.linalg.eigvals(filter_obj.state_covariance)
            assert np.all(eigenvals > 0), f"Covariance not positive definite: {eigenvals}"
    
    def test_get_current_state(self):
        """Test getting current filter state."""
        filter_obj = DynamicHedgeRatioKalman(initial_hedge_ratio=1.5)
        
        # Update with observation
        timestamp = datetime.utcnow()
        filter_obj.update(1.2, timestamp)
        
        # Get state
        state = filter_obj.get_current_state()
        
        assert isinstance(state, KalmanState)
        assert state.n_observations == 1
        assert state.timestamp == timestamp
        np.testing.assert_array_equal(state.state_mean, filter_obj.state_mean)
        np.testing.assert_array_equal(state.state_covariance, filter_obj.state_covariance)
    
    def test_set_state(self):
        """Test setting filter state."""
        filter_obj = DynamicHedgeRatioKalman()
        
        # Create state
        state_mean = np.array([2.5, 0.3])
        state_cov = np.array([[0.2, 0.1], [0.1, 0.15]])
        timestamp = datetime.utcnow()
        n_obs = 10
        
        state = KalmanState(
            state_mean=state_mean,
            state_covariance=state_cov,
            timestamp=timestamp,
            n_observations=n_obs
        )
        
        # Set state
        filter_obj.set_state(state)
        
        # Verify state was set
        np.testing.assert_array_equal(filter_obj.state_mean, state_mean)
        np.testing.assert_array_equal(filter_obj.state_covariance, state_cov)
        assert filter_obj.last_update == timestamp
        assert filter_obj.n_observations == n_obs
    
    def test_reset(self):
        """Test resetting the filter."""
        filter_obj = DynamicHedgeRatioKalman()
        
        # Update with observations
        filter_obj.update(1.5)
        filter_obj.update(1.3)
        
        assert filter_obj.n_observations == 2
        assert filter_obj.last_update is not None
        
        # Reset
        new_ratio = 2.0
        filter_obj.reset(new_ratio)
        
        # Check reset state
        assert filter_obj.n_observations == 0
        assert filter_obj.last_update is None
        assert filter_obj.state_mean[0] == new_ratio
        assert filter_obj.state_mean[1] == 0.0
    
    def test_get_diagnostics(self):
        """Test getting diagnostic information."""
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.5,
            process_variance=0.01,
            observation_variance=0.1
        )
        
        # Update with observation
        timestamp = datetime.utcnow()
        filter_obj.update(1.2, timestamp)
        
        # Get diagnostics
        diagnostics = filter_obj.get_diagnostics()
        
        required_keys = [
            'state_mean', 'state_covariance', 'state_covariance_determinant',
            'state_covariance_condition_number', 'n_observations', 'last_update',
            'process_variance', 'observation_variance'
        ]
        
        for key in required_keys:
            assert key in diagnostics
        
        assert diagnostics['n_observations'] == 1
        assert diagnostics['process_variance'] == 0.01
        assert diagnostics['observation_variance'] == 0.1
        assert diagnostics['last_update'] == timestamp.isoformat()


class TestPairTradingKalmanFilter:
    """Test cases for PairTradingKalmanFilter class."""
    
    def test_initialization(self):
        """Test pair trading filter initialization."""
        pair_filter = PairTradingKalmanFilter()
        
        assert len(pair_filter.filters) == 0
        assert len(pair_filter.pair_configs) == 0
    
    def test_add_pair(self):
        """Test adding a trading pair."""
        pair_filter = PairTradingKalmanFilter()
        
        pair_id = "BTCETH"
        asset1 = "BTCUSDT"
        asset2 = "ETHUSDT"
        initial_ratio = 15.0
        
        pair_filter.add_pair(
            pair_id=pair_id,
            asset1=asset1,
            asset2=asset2,
            initial_hedge_ratio=initial_ratio
        )
        
        # Check pair was added
        assert pair_id in pair_filter.filters
        assert pair_id in pair_filter.pair_configs
        
        # Check configuration
        config = pair_filter.pair_configs[pair_id]
        assert config['asset1'] == asset1
        assert config['asset2'] == asset2
        assert config['initial_hedge_ratio'] == initial_ratio
        
        # Check filter
        filter_obj = pair_filter.filters[pair_id]
        assert isinstance(filter_obj, DynamicHedgeRatioKalman)
        assert filter_obj.state_mean[0] == initial_ratio
    
    def test_update_pair_valid(self):
        """Test updating a pair with valid prices."""
        pair_filter = PairTradingKalmanFilter()
        
        pair_id = "BTCETH"
        pair_filter.add_pair(
            pair_id=pair_id,
            asset1="BTCUSDT",
            asset2="ETHUSDT",
            initial_hedge_ratio=15.0
        )
        
        # Update with prices
        asset1_price = 45000.0
        asset2_price = 3000.0
        timestamp = datetime.utcnow()
        
        result = pair_filter.update_pair(pair_id, asset1_price, asset2_price, timestamp)
        
        # Check result
        assert result is not None
        assert result['pair_id'] == pair_id
        assert result['asset1'] == "BTCUSDT"
        assert result['asset2'] == "ETHUSDT"
        assert result['asset1_price'] == asset1_price
        assert result['asset2_price'] == asset2_price
        assert result['price_ratio'] == asset1_price / asset2_price
        assert 'hedge_ratio' in result
        assert 'n_observations' in result
    
    def test_update_pair_unknown(self):
        """Test updating an unknown pair."""
        pair_filter = PairTradingKalmanFilter()
        
        result = pair_filter.update_pair("UNKNOWN", 100.0, 50.0)
        
        assert result is None
    
    def test_update_pair_zero_price(self):
        """Test updating a pair with zero price for asset2."""
        pair_filter = PairTradingKalmanFilter()
        
        pair_id = "BTCETH"
        pair_filter.add_pair(
            pair_id=pair_id,
            asset1="BTCUSDT",
            asset2="ETHUSDT"
        )
        
        result = pair_filter.update_pair(pair_id, 45000.0, 0.0)
        
        assert result is None
    
    def test_get_pair_state(self):
        """Test getting pair state."""
        pair_filter = PairTradingKalmanFilter()
        
        pair_id = "BTCETH"
        pair_filter.add_pair(pair_id=pair_id, asset1="BTCUSDT", asset2="ETHUSDT")
        
        # Update pair
        pair_filter.update_pair(pair_id, 45000.0, 3000.0)
        
        # Get state
        state = pair_filter.get_pair_state(pair_id)
        
        assert state is not None
        assert isinstance(state, KalmanState)
        assert state.n_observations == 1
    
    def test_get_pair_state_unknown(self):
        """Test getting state for unknown pair."""
        pair_filter = PairTradingKalmanFilter()
        
        state = pair_filter.get_pair_state("UNKNOWN")
        
        assert state is None
    
    def test_get_all_pairs(self):
        """Test getting information about all pairs."""
        pair_filter = PairTradingKalmanFilter()
        
        # Add multiple pairs
        pairs = [
            ("BTCETH", "BTCUSDT", "ETHUSDT", 15.0),
            ("BTCADA", "BTCUSDT", "ADAUSDT", 100000.0)
        ]
        
        for pair_id, asset1, asset2, ratio in pairs:
            pair_filter.add_pair(
                pair_id=pair_id,
                asset1=asset1,
                asset2=asset2,
                initial_hedge_ratio=ratio
            )
        
        # Update one pair
        pair_filter.update_pair("BTCETH", 45000.0, 3000.0)
        
        # Get all pairs info
        all_pairs = pair_filter.get_all_pairs()
        
        assert len(all_pairs) == 2
        
        for pair_id, asset1, asset2, ratio in pairs:
            assert pair_id in all_pairs
            pair_info = all_pairs[pair_id]
            
            assert pair_info['asset1'] == asset1
            assert pair_info['asset2'] == asset2
            assert pair_info['initial_hedge_ratio'] == ratio
            assert 'current_hedge_ratio' in pair_info
            assert 'n_observations' in pair_info
            assert 'last_update' in pair_info
        
        # Check that updated pair has observations
        assert all_pairs["BTCETH"]['n_observations'] == 1
        assert all_pairs["BTCADA"]['n_observations'] == 0


class TestKalmanState:
    """Test cases for KalmanState dataclass."""
    
    def test_creation(self):
        """Test creating a KalmanState."""
        state_mean = np.array([1.5, 0.1])
        state_cov = np.eye(2) * 0.1
        timestamp = datetime.utcnow()
        n_obs = 5
        
        state = KalmanState(
            state_mean=state_mean,
            state_covariance=state_cov,
            timestamp=timestamp,
            n_observations=n_obs
        )
        
        np.testing.assert_array_equal(state.state_mean, state_mean)
        np.testing.assert_array_equal(state.state_covariance, state_cov)
        assert state.timestamp == timestamp
        assert state.n_observations == n_obs


@pytest.fixture
def sample_filter():
    """Fixture providing a sample Kalman filter."""
    return DynamicHedgeRatioKalman(
        initial_hedge_ratio=1.0,
        process_variance=1e-4,
        observation_variance=1e-2
    )


@pytest.fixture
def sample_pair_filter():
    """Fixture providing a sample pair trading filter."""
    pair_filter = PairTradingKalmanFilter()
    pair_filter.add_pair(
        pair_id="BTCETH",
        asset1="BTCUSDT",
        asset2="ETHUSDT",
        initial_hedge_ratio=15.0
    )
    return pair_filter


class TestKalmanFilterIntegration:
    """Integration tests for Kalman filter components."""
    
    def test_filter_convergence_synthetic_data(self, sample_filter):
        """Test filter convergence with synthetic cointegrated data."""
        # Generate synthetic cointegrated data
        np.random.seed(42)
        true_hedge_ratio = 2.0
        n_observations = 100
        
        # Generate price ratios around true hedge ratio with noise
        observations = []
        for i in range(n_observations):
            # Add some drift and noise
            drift = 0.001 * i  # Small drift over time
            noise = np.random.normal(0, 0.1)
            obs = true_hedge_ratio + drift + noise
            observations.append(obs)
        
        # Update filter with observations
        results = []
        for obs in observations:
            result = sample_filter.update(obs)
            results.append(result)
        
        # Check convergence
        final_hedge_ratio = results[-1]['hedge_ratio']
        final_variance = results[-1]['hedge_ratio_variance']
        
        # Should converge close to true ratio (within 2 standard deviations)
        assert abs(final_hedge_ratio - true_hedge_ratio) < 2 * np.sqrt(final_variance)
        
        # Variance should decrease over time (learning)
        initial_variance = results[10]['hedge_ratio_variance']  # Skip first few
        assert final_variance < initial_variance
    
    def test_structural_break_adaptation(self, sample_filter):
        """Test filter adaptation to structural breaks."""
        np.random.seed(42)
        
        # First regime
        regime1_ratio = 1.5
        regime1_obs = [regime1_ratio + np.random.normal(0, 0.05) for _ in range(50)]
        
        # Structural break
        regime2_ratio = 2.5
        regime2_obs = [regime2_ratio + np.random.normal(0, 0.05) for _ in range(50)]
        
        all_observations = regime1_obs + regime2_obs
        results = []
        
        for obs in all_observations:
            result = sample_filter.update(obs)
            results.append(result)
        
        # Check adaptation
        regime1_final = results[49]['hedge_ratio']  # End of regime 1
        regime2_final = results[-1]['hedge_ratio']   # End of regime 2
        
        # Should be closer to respective regime means
        assert abs(regime1_final - regime1_ratio) < abs(regime1_final - regime2_ratio)
        assert abs(regime2_final - regime2_ratio) < abs(regime2_final - regime1_ratio)
    
    def test_real_time_performance_simulation(self, sample_pair_filter):
        """Test real-time performance with simulated tick data."""
        import time
        
        # Simulate high-frequency tick data
        base_time = datetime.utcnow()
        tick_interval = timedelta(milliseconds=100)  # 10 Hz
        
        processing_times = []
        
        for i in range(100):
            # Generate synthetic prices
            btc_price = 45000 + np.random.normal(0, 100)
            eth_price = 3000 + np.random.normal(0, 50)
            timestamp = base_time + i * tick_interval
            
            # Measure processing time
            start_time = time.perf_counter()
            
            result = sample_pair_filter.update_pair(
                "BTCETH", btc_price, eth_price, timestamp
            )
            
            end_time = time.perf_counter()
            processing_times.append(end_time - start_time)
            
            assert result is not None
        
        # Check performance metrics
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Should process each update quickly (< 1ms on average)
        assert avg_processing_time < 0.001, f"Average processing time too high: {avg_processing_time:.6f}s"
        assert max_processing_time < 0.01, f"Max processing time too high: {max_processing_time:.6f}s"
        
        # All updates should have been processed
        assert sample_pair_filter.filters["BTCETH"].n_observations == 100
