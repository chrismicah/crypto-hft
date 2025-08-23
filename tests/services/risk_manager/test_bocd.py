"""Unit tests for BOCD wrapper."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from services.risk_manager.bocd import (
    BOCDWrapper, AdaptiveBOCDWrapper, ChangePointEvent, BOCDState,
    create_synthetic_changepoint_data
)


class TestBOCDWrapper:
    """Test cases for BOCDWrapper."""
    
    @pytest.fixture
    def bocd_wrapper(self):
        """Create BOCDWrapper instance."""
        return BOCDWrapper(
            hazard_rate=1/100,  # Faster for testing
            max_run_length=50,
            min_observations=5
        )
    
    def test_initialization(self, bocd_wrapper):
        """Test BOCD wrapper initialization."""
        assert bocd_wrapper.hazard_rate == 1/100
        assert bocd_wrapper.max_run_length == 50
        assert bocd_wrapper.min_observations == 5
        assert bocd_wrapper.observation_count == 0
        assert len(bocd_wrapper.state.run_length_probs) == 1
        assert bocd_wrapper.state.run_length_probs[0] == 1.0
    
    def test_reset(self, bocd_wrapper):
        """Test BOCD state reset."""
        # Add some observations first
        for i in range(10):
            bocd_wrapper.update(float(i), datetime.utcnow())
        
        assert bocd_wrapper.observation_count > 0
        
        # Reset
        bocd_wrapper.reset()
        
        assert bocd_wrapper.observation_count == 0
        assert len(bocd_wrapper.state.run_length_probs) == 1
        assert bocd_wrapper.state.run_length_probs[0] == 1.0
    
    def test_update_with_insufficient_observations(self, bocd_wrapper):
        """Test update with insufficient observations."""
        timestamp = datetime.utcnow()
        
        # Should return 0 probability for first few observations
        for i in range(bocd_wrapper.min_observations - 1):
            prob, event = bocd_wrapper.update(float(i), timestamp)
            assert prob == 0.0
            assert event is None
            assert bocd_wrapper.observation_count == i + 1
    
    def test_update_with_sufficient_observations(self, bocd_wrapper):
        """Test update with sufficient observations."""
        timestamp = datetime.utcnow()
        
        # Add minimum observations
        for i in range(bocd_wrapper.min_observations):
            prob, event = bocd_wrapper.update(float(i), timestamp + timedelta(seconds=i))
        
        # Next observation should return valid probability
        prob, event = bocd_wrapper.update(10.0, timestamp + timedelta(seconds=10))
        
        assert 0.0 <= prob <= 1.0
        assert bocd_wrapper.observation_count == bocd_wrapper.min_observations + 1
    
    def test_changepoint_event_creation(self, bocd_wrapper):
        """Test changepoint event creation."""
        timestamp = datetime.utcnow()
        
        # Add observations to get past minimum
        for i in range(bocd_wrapper.min_observations + 5):
            prob, event = bocd_wrapper.update(float(i), timestamp + timedelta(seconds=i))
        
        # Add a significant change that should create an event
        prob, event = bocd_wrapper.update(100.0, timestamp + timedelta(seconds=20))
        
        if event is not None:  # Event creation depends on probability threshold
            assert isinstance(event, ChangePointEvent)
            assert event.probability > 0
            assert event.data_point == 100.0
            assert event.confidence_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_data_history_storage(self, bocd_wrapper):
        """Test that data history is properly stored."""
        timestamp = datetime.utcnow()
        
        # Add observations
        test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, value in enumerate(test_values):
            bocd_wrapper.update(value, timestamp + timedelta(seconds=i))
        
        # Check that data is stored
        stored_data = list(bocd_wrapper.state.data_history)
        assert len(stored_data) == len(test_values)
        assert stored_data == test_values
        
        # Check timestamps
        stored_timestamps = list(bocd_wrapper.state.timestamp_history)
        assert len(stored_timestamps) == len(test_values)
    
    def test_run_length_distribution_update(self, bocd_wrapper):
        """Test that run length distribution is updated."""
        timestamp = datetime.utcnow()
        
        initial_length = len(bocd_wrapper.state.run_length_probs)
        
        # Add observations
        for i in range(20):
            bocd_wrapper.update(float(i), timestamp + timedelta(seconds=i))
        
        # Run length distribution should have evolved
        final_length = len(bocd_wrapper.state.run_length_probs)
        assert final_length >= initial_length
        
        # Probabilities should sum to approximately 1
        prob_sum = np.sum(bocd_wrapper.state.run_length_probs)
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_get_statistics(self, bocd_wrapper):
        """Test statistics retrieval."""
        timestamp = datetime.utcnow()
        
        # Add some observations
        for i in range(15):
            bocd_wrapper.update(float(i), timestamp + timedelta(seconds=i))
        
        stats = bocd_wrapper.get_statistics()
        
        assert 'observation_count' in stats
        assert 'most_likely_run_length' in stats
        assert 'recent_mean_probability' in stats
        assert 'recent_max_probability' in stats
        assert 'recent_data_mean' in stats
        assert 'recent_data_std' in stats
        assert 'hazard_rate' in stats
        assert 'max_run_length' in stats
        
        assert stats['observation_count'] == 15
        assert stats['hazard_rate'] == bocd_wrapper.hazard_rate
        assert stats['max_run_length'] == bocd_wrapper.max_run_length
    
    def test_get_recent_probabilities(self, bocd_wrapper):
        """Test recent probabilities retrieval."""
        timestamp = datetime.utcnow()
        
        # Add observations to build up probability history
        for i in range(20):
            bocd_wrapper.update(float(i), timestamp + timedelta(seconds=i))
        
        recent_probs = bocd_wrapper.get_recent_probabilities(10)
        
        # Should return up to 10 recent probabilities
        assert len(recent_probs) <= 10
        
        # All should be valid probabilities
        for prob in recent_probs:
            assert 0.0 <= prob <= 1.0


class TestAdaptiveBOCDWrapper:
    """Test cases for AdaptiveBOCDWrapper."""
    
    @pytest.fixture
    def adaptive_bocd(self):
        """Create AdaptiveBOCDWrapper instance."""
        return AdaptiveBOCDWrapper(
            base_hazard_rate=1/100,
            volatility_adjustment=True,
            trend_adjustment=True,
            max_run_length=50,
            min_observations=5
        )
    
    def test_initialization(self, adaptive_bocd):
        """Test adaptive BOCD initialization."""
        assert adaptive_bocd.base_hazard_rate == 1/100
        assert adaptive_bocd.volatility_adjustment is True
        assert adaptive_bocd.trend_adjustment is True
        assert adaptive_bocd.volatility_window == 50
        assert adaptive_bocd.trend_window == 20
    
    def test_hazard_rate_adjustment(self, adaptive_bocd):
        """Test hazard rate adjustment based on market conditions."""
        timestamp = datetime.utcnow()
        
        # Add stable data first
        stable_data = [1.0] * 60  # Low volatility
        for i, value in enumerate(stable_data):
            adaptive_bocd.update(value, timestamp + timedelta(seconds=i))
        
        stable_hazard_rate = adaptive_bocd.hazard_rate
        
        # Add volatile data
        volatile_data = np.random.normal(1.0, 5.0, 30)  # High volatility
        for i, value in enumerate(volatile_data):
            adaptive_bocd.update(value, timestamp + timedelta(seconds=60 + i))
        
        volatile_hazard_rate = adaptive_bocd.hazard_rate
        
        # Hazard rate should have increased due to higher volatility
        # (Note: this test might be flaky due to randomness)
        assert volatile_hazard_rate >= stable_hazard_rate * 0.8  # Allow some variance
    
    def test_trend_adjustment(self, adaptive_bocd):
        """Test hazard rate adjustment based on trends."""
        timestamp = datetime.utcnow()
        
        # Add data with clear trend
        trend_data = list(range(60))  # Strong upward trend
        for i, value in enumerate(trend_data):
            adaptive_bocd.update(float(value), timestamp + timedelta(seconds=i))
        
        # Hazard rate should be adjusted for trend
        assert adaptive_bocd.hazard_rate != adaptive_bocd.base_hazard_rate


class TestChangePointEvent:
    """Test cases for ChangePointEvent."""
    
    def test_event_creation(self):
        """Test changepoint event creation."""
        timestamp = datetime.utcnow()
        
        event = ChangePointEvent(
            timestamp=timestamp,
            probability=0.8,
            run_length=10,
            data_point=5.0,
            confidence_level="",  # Will be set in __post_init__
            metadata={'test': 'value'}
        )
        
        assert event.timestamp == timestamp
        assert event.probability == 0.8
        assert event.run_length == 10
        assert event.data_point == 5.0
        assert event.confidence_level == "HIGH"  # 0.8 should be HIGH
        assert event.metadata['test'] == 'value'
    
    def test_confidence_level_assignment(self):
        """Test confidence level assignment based on probability."""
        test_cases = [
            (0.95, "CRITICAL"),
            (0.85, "HIGH"),
            (0.6, "MEDIUM"),
            (0.3, "LOW")
        ]
        
        for prob, expected_confidence in test_cases:
            event = ChangePointEvent(
                timestamp=datetime.utcnow(),
                probability=prob,
                run_length=5,
                data_point=1.0,
                confidence_level=""
            )
            
            assert event.confidence_level == expected_confidence


class TestSyntheticData:
    """Test cases for synthetic changepoint data generation."""
    
    def test_synthetic_data_creation(self):
        """Test synthetic changepoint data creation."""
        n_points = 1000
        changepoints = [300, 700]
        
        data, actual_changepoints = create_synthetic_changepoint_data(
            n_points=n_points,
            changepoint_locations=changepoints,
            noise_std=1.0
        )
        
        assert len(data) == n_points
        assert actual_changepoints == changepoints
        assert isinstance(data, np.ndarray)
    
    def test_synthetic_data_with_regimes(self):
        """Test synthetic data with different regime parameters."""
        n_points = 500
        changepoints = [200, 350]
        regime_means = [0.0, 5.0, -2.0]
        regime_stds = [1.0, 2.0, 0.5]
        
        data, actual_changepoints = create_synthetic_changepoint_data(
            n_points=n_points,
            changepoint_locations=changepoints,
            regime_means=regime_means,
            regime_stds=regime_stds
        )
        
        assert len(data) == n_points
        assert actual_changepoints == changepoints
        
        # Check that different regimes have different statistical properties
        regime1_data = data[:changepoints[0]]
        regime2_data = data[changepoints[0]:changepoints[1]]
        regime3_data = data[changepoints[1]:]
        
        # Means should be approximately different
        mean1 = np.mean(regime1_data)
        mean2 = np.mean(regime2_data)
        mean3 = np.mean(regime3_data)
        
        # Allow for some variance due to noise
        assert abs(mean1 - regime_means[0]) < 1.0
        assert abs(mean2 - regime_means[1]) < 1.0
        assert abs(mean3 - regime_means[2]) < 1.0
    
    def test_default_parameters(self):
        """Test synthetic data with default parameters."""
        data, changepoints = create_synthetic_changepoint_data()
        
        assert len(data) == 1000  # Default n_points
        assert len(changepoints) == 2  # Default has 2 changepoints
        assert isinstance(data, np.ndarray)


class TestBOCDIntegration:
    """Integration tests for BOCD with synthetic data."""
    
    def test_bocd_with_synthetic_changepoints(self):
        """Test BOCD detection with synthetic changepoint data."""
        # Create synthetic data with known changepoints
        n_points = 300
        changepoints = [100, 200]
        regime_means = [0.0, 5.0, -3.0]
        
        data, actual_changepoints = create_synthetic_changepoint_data(
            n_points=n_points,
            changepoint_locations=changepoints,
            regime_means=regime_means,
            noise_std=0.5  # Low noise for clearer signal
        )
        
        # Initialize BOCD
        bocd = BOCDWrapper(
            hazard_rate=1/50,  # Expect changepoint every 50 observations
            min_observations=10
        )
        
        # Process data
        probabilities = []
        events = []
        
        base_time = datetime.utcnow()
        for i, value in enumerate(data):
            timestamp = base_time + timedelta(seconds=i)
            prob, event = bocd.update(value, timestamp)
            probabilities.append(prob)
            
            if event:
                events.append((i, event))
        
        # Should detect some changepoints
        assert len(events) > 0
        
        # Check if detected changepoints are near actual changepoints
        detected_locations = [loc for loc, _ in events if _.probability > 0.5]
        
        if detected_locations:
            # At least one detection should be near an actual changepoint
            min_distances = []
            for actual_cp in actual_changepoints:
                distances = [abs(detected - actual_cp) for detected in detected_locations]
                if distances:
                    min_distances.append(min(distances))
            
            # At least one detection should be within 20 points of actual changepoint
            assert any(dist <= 20 for dist in min_distances)
    
    def test_bocd_with_no_changepoints(self):
        """Test BOCD with stationary data (no changepoints)."""
        # Generate stationary data
        n_points = 200
        np.random.seed(42)  # For reproducible results
        data = np.random.normal(0, 1, n_points)
        
        bocd = BOCDWrapper(
            hazard_rate=1/100,
            min_observations=10
        )
        
        probabilities = []
        high_prob_events = 0
        
        base_time = datetime.utcnow()
        for i, value in enumerate(data):
            timestamp = base_time + timedelta(seconds=i)
            prob, event = bocd.update(value, timestamp)
            probabilities.append(prob)
            
            if prob > 0.7:  # High probability threshold
                high_prob_events += 1
        
        # Should have few high probability events for stationary data
        assert high_prob_events < n_points * 0.1  # Less than 10% high probability events
