"""Unit tests for Kalman filter state management."""

import pytest
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import joblib

from services.kalman_filter.state import StateManager
from services.kalman_filter.filter import (
    DynamicHedgeRatioKalman,
    PairTradingKalmanFilter,
    KalmanState
)


class TestStateManager:
    """Test cases for StateManager class."""
    
    @pytest.fixture
    def temp_state_dir(self):
        """Create a temporary directory for state files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def state_manager(self, temp_state_dir):
        """Create a StateManager with temporary directory."""
        return StateManager(
            state_dir=temp_state_dir,
            auto_save_interval=0,  # Disable auto-save for tests
            backup_retention_days=1
        )
    
    @pytest.fixture
    def sample_filter(self):
        """Create a sample Kalman filter with some state."""
        filter_obj = DynamicHedgeRatioKalman(initial_hedge_ratio=1.5)
        
        # Add some observations to create meaningful state
        observations = [1.2, 1.3, 1.1, 1.4, 1.25]
        for obs in observations:
            filter_obj.update(obs)
        
        return filter_obj
    
    def test_initialization(self, temp_state_dir):
        """Test StateManager initialization."""
        state_manager = StateManager(
            state_dir=temp_state_dir,
            auto_save_interval=300,
            backup_retention_days=7
        )
        
        assert state_manager.state_dir == Path(temp_state_dir)
        assert state_manager.auto_save_interval == 300
        assert state_manager.backup_retention_days == 7
        assert Path(temp_state_dir).exists()
    
    def test_save_state_success(self, state_manager, sample_filter):
        """Test successful state saving."""
        identifier = "test_filter"
        
        # Save state
        success = state_manager.save_state(sample_filter, identifier)
        
        assert success is True
        
        # Check file was created
        state_file = state_manager.state_dir / f"{identifier}.joblib"
        assert state_file.exists()
        
        # Check file contents
        state_data = joblib.load(state_file)
        
        required_keys = ['state', 'filter_params', 'metadata']
        for key in required_keys:
            assert key in state_data
        
        # Check state data
        assert 'state_mean' in state_data['state']
        assert 'state_covariance' in state_data['state']
        assert 'timestamp' in state_data['state']
        assert 'n_observations' in state_data['state']
        
        # Check metadata
        assert state_data['metadata']['identifier'] == identifier
        assert 'saved_at' in state_data['metadata']
    
    def test_save_state_with_backup(self, state_manager, sample_filter):
        """Test state saving with backup creation."""
        identifier = "test_filter"
        state_file = state_manager.state_dir / f"{identifier}.joblib"
        
        # Create initial file
        initial_data = {'test': 'data'}
        joblib.dump(initial_data, state_file)
        
        # Save new state (should create backup)
        success = state_manager.save_state(sample_filter, identifier, create_backup=True)
        
        assert success is True
        
        # Check backup was created
        backup_files = list(state_manager.state_dir.glob(f"{identifier}_backup_*.joblib"))
        assert len(backup_files) == 1
        
        # Check backup contains original data
        backup_data = joblib.load(backup_files[0])
        assert backup_data == initial_data
    
    def test_save_state_failure(self, state_manager):
        """Test state saving failure handling."""
        # Create a filter with invalid state
        filter_obj = DynamicHedgeRatioKalman()
        
        # Mock joblib.dump to raise exception
        with patch('services.kalman_filter.state.joblib.dump', side_effect=Exception("Save failed")):
            success = state_manager.save_state(filter_obj, "test_filter")
            
            assert success is False
    
    def test_load_state_success(self, state_manager, sample_filter):
        """Test successful state loading."""
        identifier = "test_filter"
        
        # Save state first
        state_manager.save_state(sample_filter, identifier)
        
        # Load state
        loaded_filter = state_manager.load_state(identifier)
        
        assert loaded_filter is not None
        assert isinstance(loaded_filter, DynamicHedgeRatioKalman)
        
        # Check state was loaded correctly
        original_state = sample_filter.get_current_state()
        loaded_state = loaded_filter.get_current_state()
        
        np.testing.assert_array_almost_equal(
            original_state.state_mean, 
            loaded_state.state_mean
        )
        np.testing.assert_array_almost_equal(
            original_state.state_covariance, 
            loaded_state.state_covariance
        )
        assert loaded_state.n_observations == original_state.n_observations
    
    def test_load_state_into_existing_filter(self, state_manager, sample_filter):
        """Test loading state into an existing filter."""
        identifier = "test_filter"
        
        # Save state
        state_manager.save_state(sample_filter, identifier)
        
        # Create new filter
        new_filter = DynamicHedgeRatioKalman(initial_hedge_ratio=2.0)
        
        # Load state into existing filter
        loaded_filter = state_manager.load_state(identifier, new_filter)
        
        assert loaded_filter is new_filter  # Same object
        
        # Check state was loaded
        original_state = sample_filter.get_current_state()
        loaded_state = loaded_filter.get_current_state()
        
        np.testing.assert_array_almost_equal(
            original_state.state_mean, 
            loaded_state.state_mean
        )
    
    def test_load_state_file_not_found(self, state_manager):
        """Test loading state when file doesn't exist."""
        loaded_filter = state_manager.load_state("nonexistent_filter")
        
        assert loaded_filter is None
    
    def test_load_state_invalid_data(self, state_manager):
        """Test loading state with invalid data format."""
        identifier = "invalid_filter"
        state_file = state_manager.state_dir / f"{identifier}.joblib"
        
        # Save invalid data
        invalid_data = {'invalid': 'format'}
        joblib.dump(invalid_data, state_file)
        
        # Try to load
        loaded_filter = state_manager.load_state(identifier)
        
        assert loaded_filter is None
    
    def test_save_pair_filters(self, state_manager):
        """Test saving pair trading filters."""
        # Create pair filter with multiple pairs
        pair_filter = PairTradingKalmanFilter()
        
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
            
            # Add some observations
            pair_filter.update_pair(pair_id, 45000.0, 3000.0 if "ETH" in asset2 else 0.5)
        
        # Save pair filters
        identifier = "test_pairs"
        success = state_manager.save_pair_filters(pair_filter, identifier)
        
        assert success is True
        
        # Check individual filter files were created
        for pair_id, _, _, _ in pairs:
            filter_file = state_manager.state_dir / f"{identifier}_{pair_id}.joblib"
            assert filter_file.exists()
        
        # Check config file was created
        config_file = state_manager.state_dir / f"{identifier}_configs.joblib"
        assert config_file.exists()
    
    def test_load_pair_filters(self, state_manager):
        """Test loading pair trading filters."""
        # Create and save pair filter
        original_pair_filter = PairTradingKalmanFilter()
        
        pairs = [
            ("BTCETH", "BTCUSDT", "ETHUSDT", 15.0),
            ("BTCADA", "BTCUSDT", "ADAUSDT", 100000.0)
        ]
        
        for pair_id, asset1, asset2, ratio in pairs:
            original_pair_filter.add_pair(
                pair_id=pair_id,
                asset1=asset1,
                asset2=asset2,
                initial_hedge_ratio=ratio
            )
            
            # Add observations
            for _ in range(5):
                original_pair_filter.update_pair(pair_id, 45000.0, 3000.0 if "ETH" in asset2 else 0.5)
        
        identifier = "test_pairs"
        state_manager.save_pair_filters(original_pair_filter, identifier)
        
        # Load pair filters
        loaded_pair_filter = state_manager.load_pair_filters(identifier)
        
        assert loaded_pair_filter is not None
        assert isinstance(loaded_pair_filter, PairTradingKalmanFilter)
        
        # Check all pairs were loaded
        assert len(loaded_pair_filter.filters) == len(pairs)
        
        for pair_id, asset1, asset2, ratio in pairs:
            assert pair_id in loaded_pair_filter.filters
            assert pair_id in loaded_pair_filter.pair_configs
            
            # Check configuration
            config = loaded_pair_filter.pair_configs[pair_id]
            assert config['asset1'] == asset1
            assert config['asset2'] == asset2
            assert config['initial_hedge_ratio'] == ratio
            
            # Check state was loaded
            filter_obj = loaded_pair_filter.filters[pair_id]
            assert filter_obj.n_observations == 5
    
    def test_load_pair_filters_config_not_found(self, state_manager):
        """Test loading pair filters when config file doesn't exist."""
        loaded_pair_filter = state_manager.load_pair_filters("nonexistent")
        
        assert loaded_pair_filter is None
    
    def test_auto_save_registration(self, state_manager, sample_filter):
        """Test registering filters for auto-save."""
        identifier = "auto_save_filter"
        
        # Register filter
        state_manager.register_for_auto_save(identifier, sample_filter)
        
        assert identifier in state_manager._filters_to_save
        assert state_manager._filters_to_save[identifier] is sample_filter
    
    def test_auto_save_pair_filter_registration(self, state_manager):
        """Test registering pair filters for auto-save."""
        pair_filter = PairTradingKalmanFilter()
        identifier = "auto_save_pairs"
        
        # Register pair filter
        state_manager.register_pair_filter_for_auto_save(identifier, pair_filter)
        
        assert identifier in state_manager._pair_filters_to_save
        assert state_manager._pair_filters_to_save[identifier] is pair_filter
    
    def test_validate_state_data_valid(self, state_manager):
        """Test state data validation with valid data."""
        valid_data = {
            'state': {
                'state_mean': np.array([1.0, 0.0]),
                'state_covariance': np.eye(2),
                'timestamp': datetime.utcnow(),
                'n_observations': 5
            },
            'filter_params': {
                'process_variance': 1e-5,
                'observation_variance': 1e-3
            },
            'metadata': {
                'saved_at': datetime.utcnow(),
                'identifier': 'test',
                'version': '1.0'
            }
        }
        
        assert state_manager._validate_state_data(valid_data) is True
    
    def test_validate_state_data_invalid(self, state_manager):
        """Test state data validation with invalid data."""
        # Missing required keys
        invalid_data1 = {
            'state': {},
            'filter_params': {}
            # Missing 'metadata'
        }
        
        assert state_manager._validate_state_data(invalid_data1) is False
        
        # Missing state keys
        invalid_data2 = {
            'state': {
                'state_mean': np.array([1.0, 0.0])
                # Missing other state keys
            },
            'filter_params': {
                'process_variance': 1e-5,
                'observation_variance': 1e-3
            },
            'metadata': {}
        }
        
        assert state_manager._validate_state_data(invalid_data2) is False
    
    def test_get_available_states(self, state_manager, sample_filter):
        """Test getting available saved states."""
        # Save multiple states
        identifiers = ["filter1", "filter2", "filter3"]
        
        for identifier in identifiers:
            state_manager.save_state(sample_filter, identifier)
        
        # Get available states
        available_states = state_manager.get_available_states()
        
        assert len(available_states) == len(identifiers)
        
        for identifier in identifiers:
            assert identifier in available_states
            
            state_info = available_states[identifier]
            required_keys = ['file', 'size_bytes', 'modified', 'n_observations', 'saved_at', 'version']
            
            for key in required_keys:
                assert key in state_info
            
            assert state_info['n_observations'] == sample_filter.n_observations
    
    def test_cleanup_old_backups(self, state_manager):
        """Test cleanup of old backup files."""
        # Create old backup files
        old_time = datetime.utcnow() - timedelta(days=10)
        recent_time = datetime.utcnow() - timedelta(hours=1)
        
        old_backup = state_manager.state_dir / "filter_backup_20230101_120000.joblib"
        recent_backup = state_manager.state_dir / "filter_backup_20231201_120000.joblib"
        
        # Create files
        old_backup.touch()
        recent_backup.touch()
        
        # Set modification times
        import os
        os.utime(old_backup, (old_time.timestamp(), old_time.timestamp()))
        os.utime(recent_backup, (recent_time.timestamp(), recent_time.timestamp()))
        
        # Run cleanup
        state_manager._cleanup_old_backups()
        
        # Check results
        assert not old_backup.exists()  # Should be deleted
        assert recent_backup.exists()   # Should remain
    
    def test_context_manager(self, temp_state_dir):
        """Test StateManager as context manager."""
        with StateManager(state_dir=temp_state_dir, auto_save_interval=1) as state_manager:
            # Auto-save should be started
            assert state_manager._auto_save_thread is not None
            
            # Register a filter
            filter_obj = DynamicHedgeRatioKalman()
            state_manager.register_for_auto_save("test", filter_obj)
        
        # Auto-save should be stopped after context exit
        # Note: Thread might take a moment to stop, so we don't assert thread state
    
    @patch('services.kalman_filter.state.joblib.dump')
    def test_auto_save_loop_error_handling(self, mock_dump, state_manager, sample_filter):
        """Test error handling in auto-save loop."""
        # Register filter
        state_manager.register_for_auto_save("test", sample_filter)
        
        # Make dump raise exception
        mock_dump.side_effect = Exception("Save failed")
        
        # Run one iteration of auto-save loop
        state_manager._auto_save_loop()
        
        # Should not raise exception (error should be caught and logged)
        # The method should complete without crashing


class TestStateManagerIntegration:
    """Integration tests for StateManager."""
    
    @pytest.fixture
    def temp_state_dir(self):
        """Create a temporary directory for state files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_save_load_cycle(self, temp_state_dir):
        """Test complete save and load cycle."""
        state_manager = StateManager(state_dir=temp_state_dir, auto_save_interval=0)
        
        # Create filter with meaningful state
        original_filter = DynamicHedgeRatioKalman(
            initial_hedge_ratio=2.5,
            process_variance=1e-4,
            observation_variance=1e-2
        )
        
        # Add observations
        observations = [2.3, 2.7, 2.4, 2.6, 2.5, 2.8, 2.2]
        for obs in observations:
            original_filter.update(obs)
        
        identifier = "integration_test"
        
        # Save state
        save_success = state_manager.save_state(original_filter, identifier)
        assert save_success is True
        
        # Load state
        loaded_filter = state_manager.load_state(identifier)
        assert loaded_filter is not None
        
        # Compare states
        original_state = original_filter.get_current_state()
        loaded_state = loaded_filter.get_current_state()
        
        # Check all state components
        np.testing.assert_array_almost_equal(
            original_state.state_mean, 
            loaded_state.state_mean,
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            original_state.state_covariance, 
            loaded_state.state_covariance,
            decimal=10
        )
        assert loaded_state.n_observations == original_state.n_observations
        
        # Check filter parameters
        assert loaded_filter.process_variance == original_filter.process_variance
        assert loaded_filter.observation_variance == original_filter.observation_variance
        
        # Test that loaded filter continues to work
        new_obs = 2.4
        original_result = original_filter.update(new_obs)
        loaded_result = loaded_filter.update(new_obs)
        
        # Results should be very similar (might have tiny numerical differences)
        assert abs(original_result['hedge_ratio'] - loaded_result['hedge_ratio']) < 1e-10
    
    def test_pair_filter_persistence_cycle(self, temp_state_dir):
        """Test complete pair filter save and load cycle."""
        state_manager = StateManager(state_dir=temp_state_dir, auto_save_interval=0)
        
        # Create pair filter with multiple pairs
        original_pair_filter = PairTradingKalmanFilter()
        
        pairs_config = [
            ("BTCETH", "BTCUSDT", "ETHUSDT", 15.0, 1e-4, 1e-2),
            ("BTCADA", "BTCUSDT", "ADAUSDT", 100000.0, 1e-3, 1e-1),
            ("ETHADA", "ETHUSDT", "ADAUSDT", 6666.0, 5e-4, 5e-2)
        ]
        
        for pair_id, asset1, asset2, ratio, proc_var, obs_var in pairs_config:
            original_pair_filter.add_pair(
                pair_id=pair_id,
                asset1=asset1,
                asset2=asset2,
                initial_hedge_ratio=ratio,
                process_variance=proc_var,
                observation_variance=obs_var
            )
            
            # Add different number of observations for each pair
            n_obs = hash(pair_id) % 10 + 5  # 5-14 observations
            for i in range(n_obs):
                price1 = 45000.0 + i * 100
                price2 = 3000.0 + i * 10 if "ETH" in asset2 else 0.5 + i * 0.01
                original_pair_filter.update_pair(pair_id, price1, price2)
        
        identifier = "integration_pairs"
        
        # Save pair filters
        save_success = state_manager.save_pair_filters(original_pair_filter, identifier)
        assert save_success is True
        
        # Load pair filters
        loaded_pair_filter = state_manager.load_pair_filters(identifier)
        assert loaded_pair_filter is not None
        
        # Compare all pairs
        original_pairs = original_pair_filter.get_all_pairs()
        loaded_pairs = loaded_pair_filter.get_all_pairs()
        
        assert len(loaded_pairs) == len(original_pairs)
        
        for pair_id in original_pairs.keys():
            assert pair_id in loaded_pairs
            
            # Compare configurations
            orig_config = original_pairs[pair_id]
            loaded_config = loaded_pairs[pair_id]
            
            for key in ['asset1', 'asset2', 'initial_hedge_ratio', 'process_variance', 'observation_variance']:
                assert orig_config[key] == loaded_config[key]
            
            assert loaded_config['n_observations'] == orig_config['n_observations']
            
            # Compare filter states
            orig_state = original_pair_filter.get_pair_state(pair_id)
            loaded_state = loaded_pair_filter.get_pair_state(pair_id)
            
            np.testing.assert_array_almost_equal(
                orig_state.state_mean,
                loaded_state.state_mean,
                decimal=10
            )
            np.testing.assert_array_almost_equal(
                orig_state.state_covariance,
                loaded_state.state_covariance,
                decimal=10
            )
        
        # Test that loaded filters continue to work
        test_pair = list(original_pairs.keys())[0]
        
        orig_result = original_pair_filter.update_pair(test_pair, 50000.0, 3500.0)
        loaded_result = loaded_pair_filter.update_pair(test_pair, 50000.0, 3500.0)
        
        # Results should be very similar
        assert abs(orig_result['hedge_ratio'] - loaded_result['hedge_ratio']) < 1e-10
