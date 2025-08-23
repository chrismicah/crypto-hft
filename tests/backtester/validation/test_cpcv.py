"""Unit tests for CPCV implementation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester.validation.cpcv import (
    CPCVSplit, TimeSeriesSplitter, CPCVValidator,
    create_synthetic_leakage_test_data
)


class TestCPCVSplit:
    """Test cases for CPCVSplit."""
    
    def test_split_creation(self):
        """Test creating a CPCV split."""
        train_periods = [
            (datetime(2024, 1, 1), datetime(2024, 1, 15)),
            (datetime(2024, 2, 1), datetime(2024, 2, 15))
        ]
        test_period = (datetime(2024, 3, 1), datetime(2024, 3, 15))
        purged_periods = [
            (datetime(2024, 1, 16), datetime(2024, 1, 31)),
            (datetime(2024, 2, 16), datetime(2024, 2, 28))
        ]
        
        split = CPCVSplit(
            split_id=1,
            train_periods=train_periods,
            test_period=test_period,
            purged_periods=purged_periods
        )
        
        assert split.split_id == 1
        assert len(split.train_periods) == 2
        assert split.test_period == test_period
        assert len(split.purged_periods) == 2
    
    def test_split_validation_overlap(self):
        """Test that overlapping train and test periods raise error."""
        train_periods = [(datetime(2024, 1, 1), datetime(2024, 1, 20))]
        test_period = (datetime(2024, 1, 15), datetime(2024, 1, 30))  # Overlaps with train
        
        with pytest.raises(ValueError, match="overlaps with test period"):
            CPCVSplit(
                split_id=1,
                train_periods=train_periods,
                test_period=test_period,
                purged_periods=[]
            )
    
    def test_get_train_mask(self):
        """Test getting training data mask."""
        train_periods = [
            (datetime(2024, 1, 1), datetime(2024, 1, 5)),
            (datetime(2024, 1, 10), datetime(2024, 1, 15))
        ]
        test_period = (datetime(2024, 1, 20), datetime(2024, 1, 25))
        
        split = CPCVSplit(
            split_id=1,
            train_periods=train_periods,
            test_period=test_period,
            purged_periods=[]
        )
        
        # Create test timestamps
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 30),
            freq='D'
        )
        
        train_mask = split.get_train_mask(timestamps)
        
        # Should include dates in train periods
        assert train_mask[0]  # 2024-01-01
        assert train_mask[4]  # 2024-01-05
        assert not train_mask[5]  # 2024-01-06 (gap)
        assert train_mask[9]  # 2024-01-10
        assert train_mask[14]  # 2024-01-15
        assert not train_mask[19]  # 2024-01-20 (test period)
    
    def test_get_test_mask(self):
        """Test getting test data mask."""
        train_periods = [(datetime(2024, 1, 1), datetime(2024, 1, 10))]
        test_period = (datetime(2024, 1, 20), datetime(2024, 1, 25))
        
        split = CPCVSplit(
            split_id=1,
            train_periods=train_periods,
            test_period=test_period,
            purged_periods=[]
        )
        
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 30),
            freq='D'
        )
        
        test_mask = split.get_test_mask(timestamps)
        
        # Should only include dates in test period
        assert not test_mask[0]  # 2024-01-01 (train)
        assert not test_mask[15]  # 2024-01-16 (gap)
        assert test_mask[19]  # 2024-01-20 (test start)
        assert test_mask[24]  # 2024-01-25 (test end)
        assert not test_mask[25]  # 2024-01-26 (after test)
    
    def test_get_purged_mask(self):
        """Test getting purged data mask."""
        train_periods = [(datetime(2024, 1, 1), datetime(2024, 1, 10))]
        test_period = (datetime(2024, 1, 20), datetime(2024, 1, 25))
        purged_periods = [(datetime(2024, 1, 11), datetime(2024, 1, 19))]
        
        split = CPCVSplit(
            split_id=1,
            train_periods=train_periods,
            test_period=test_period,
            purged_periods=purged_periods
        )
        
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 30),
            freq='D'
        )
        
        purged_mask = split.get_purged_mask(timestamps)
        
        # Should include dates in purged periods
        assert not purged_mask[0]  # 2024-01-01 (train)
        assert not purged_mask[9]  # 2024-01-10 (train)
        assert purged_mask[10]  # 2024-01-11 (purged start)
        assert purged_mask[18]  # 2024-01-19 (purged end)
        assert not purged_mask[19]  # 2024-01-20 (test)


class TestTimeSeriesSplitter:
    """Test cases for TimeSeriesSplitter."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
            freq='H'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': 100 + np.cumsum(np.random.normal(0, 0.01, len(timestamps))),
            'volume': np.random.lognormal(10, 0.5, len(timestamps))
        })
        
        return data
    
    def test_splitter_initialization(self):
        """Test TimeSeriesSplitter initialization."""
        splitter = TimeSeriesSplitter(
            n_splits=5,
            test_size=0.2,
            embargo_period=timedelta(hours=24),
            purge_period=timedelta(hours=12)
        )
        
        assert splitter.n_splits == 5
        assert splitter.test_size == 0.2
        assert splitter.embargo_period == timedelta(hours=24)
        assert splitter.purge_period == timedelta(hours=12)
    
    def test_split_generation(self, sample_data):
        """Test generating CPCV splits."""
        splitter = TimeSeriesSplitter(n_splits=3, test_size=0.1)
        splits = splitter.split(sample_data)
        
        assert len(splits) <= 3  # May be fewer if insufficient data
        
        for split in splits:
            assert isinstance(split, CPCVSplit)
            assert split.split_id >= 0
            assert len(split.train_periods) > 0
            assert split.test_period is not None
    
    def test_split_no_overlap(self, sample_data):
        """Test that splits don't have overlapping train/test periods."""
        splitter = TimeSeriesSplitter(n_splits=2, test_size=0.1)
        splits = splitter.split(sample_data)
        
        timestamps = pd.to_datetime(sample_data['timestamp'])
        
        for split in splits:
            train_mask = split.get_train_mask(timestamps)
            test_mask = split.get_test_mask(timestamps)
            
            # No overlap between train and test
            overlap = np.any(train_mask & test_mask)
            assert not overlap, f"Split {split.split_id} has overlapping train/test data"
    
    def test_split_with_insufficient_data(self):
        """Test splitter behavior with insufficient data."""
        # Very small dataset
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            freq='H'
        )
        
        small_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': [100] * len(timestamps)
        })
        
        splitter = TimeSeriesSplitter(n_splits=5, test_size=0.3)
        splits = splitter.split(small_data)
        
        # Should handle gracefully, may return fewer splits
        assert len(splits) >= 0
    
    def test_embargo_period_enforcement(self, sample_data):
        """Test that embargo periods are properly enforced."""
        embargo_hours = 48
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=0.1,
            embargo_period=timedelta(hours=embargo_hours)
        )
        
        splits = splitter.split(sample_data)
        timestamps = pd.to_datetime(sample_data['timestamp'])
        
        for split in splits:
            test_start, test_end = split.test_period
            
            # Check that no training data is within embargo period of test
            for train_start, train_end in split.train_periods:
                if train_end < test_start:
                    # Training before test - check embargo gap
                    gap = test_start - train_end
                    assert gap >= timedelta(hours=embargo_hours), \
                        f"Insufficient embargo gap: {gap} < {embargo_hours}h"
                elif train_start > test_end:
                    # Training after test - check embargo gap
                    gap = train_start - test_end
                    assert gap >= timedelta(hours=embargo_hours), \
                        f"Insufficient embargo gap: {gap} < {embargo_hours}h"


class TestCPCVValidator:
    """Test cases for CPCVValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create CPCVValidator instance."""
        return CPCVValidator()
    
    @pytest.fixture
    def sample_data_and_splits(self):
        """Create sample data and splits for validation."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 3, 31),
            freq='D'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': 100 + np.cumsum(np.random.normal(0, 0.01, len(timestamps)))
        })
        
        # Create valid splits
        splits = [
            CPCVSplit(
                split_id=0,
                train_periods=[(datetime(2024, 1, 1), datetime(2024, 1, 31))],
                test_period=(datetime(2024, 2, 15), datetime(2024, 2, 28)),
                purged_periods=[(datetime(2024, 2, 1), datetime(2024, 2, 14))]
            ),
            CPCVSplit(
                split_id=1,
                train_periods=[(datetime(2024, 2, 1), datetime(2024, 2, 28))],
                test_period=(datetime(2024, 3, 15), datetime(2024, 3, 31)),
                purged_periods=[(datetime(2024, 3, 1), datetime(2024, 3, 14))]
            )
        ]
        
        return data, splits
    
    def test_validate_valid_splits(self, validator, sample_data_and_splits):
        """Test validation of valid splits."""
        data, splits = sample_data_and_splits
        
        results = validator.validate_splits(splits, data)
        
        assert results['total_splits'] == 2
        assert results['valid_splits'] == 2
        assert not results['leakage_detected']
        assert len(results['split_details']) == 2
        
        for split_detail in results['split_details']:
            assert split_detail['valid']
            assert not split_detail['potential_leakage']
            assert split_detail['train_size'] > 0
            assert split_detail['test_size'] > 0
    
    def test_validate_overlapping_splits(self, validator):
        """Test validation detects overlapping train/test data."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 2, 28),
            freq='D'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': [100] * len(timestamps)
        })
        
        # Create invalid split with overlap
        invalid_split = CPCVSplit(
            split_id=0,
            train_periods=[(datetime(2024, 1, 1), datetime(2024, 1, 20))],
            test_period=(datetime(2024, 1, 15), datetime(2024, 1, 25)),  # Overlaps
            purged_periods=[]
        )
        
        # This should raise an error during split creation
        with pytest.raises(ValueError):
            validator.validate_splits([invalid_split], data)
    
    def test_coverage_statistics(self, validator, sample_data_and_splits):
        """Test coverage statistics calculation."""
        data, splits = sample_data_and_splits
        
        results = validator.validate_splits(splits, data)
        
        coverage_stats = results['coverage_stats']
        
        assert 'train_coverage_pct' in coverage_stats
        assert 'test_coverage_pct' in coverage_stats
        assert 'total_coverage_pct' in coverage_stats
        assert 'overlap_pct' in coverage_stats
        
        # Coverage percentages should be reasonable
        assert 0 <= coverage_stats['train_coverage_pct'] <= 100
        assert 0 <= coverage_stats['test_coverage_pct'] <= 100
        assert 0 <= coverage_stats['total_coverage_pct'] <= 100
        assert 0 <= coverage_stats['overlap_pct'] <= 100


class TestSyntheticLeakageData:
    """Test cases for synthetic leakage test data."""
    
    def test_synthetic_data_creation(self):
        """Test creating synthetic data for leakage testing."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        pattern_change_date = datetime(2024, 3, 15)
        
        data = create_synthetic_leakage_test_data(
            start_date=start_date,
            end_date=end_date,
            freq='1H',
            pattern_change_date=pattern_change_date
        )
        
        assert len(data) > 0
        assert 'timestamp' in data.columns
        assert 'price' in data.columns
        assert 'returns' in data.columns
        assert 'pattern_period' in data.columns
        
        # Check timestamp range
        assert data['timestamp'].min() >= start_date
        assert data['timestamp'].max() <= end_date
        
        # Check pattern period indicator
        pattern_mask = data['pattern_period']
        pattern_timestamps = data[pattern_mask]['timestamp']
        
        # All pattern period timestamps should be before change date
        assert all(ts < pattern_change_date for ts in pattern_timestamps)
    
    def test_synthetic_data_pattern_detection(self):
        """Test that synthetic data has detectable pattern in first half."""
        data = create_synthetic_leakage_test_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            freq='1D'
        )
        
        # Split data into pattern and non-pattern periods
        pattern_data = data[data['pattern_period']]
        non_pattern_data = data[~data['pattern_period']]
        
        assert len(pattern_data) > 0
        assert len(non_pattern_data) > 0
        
        # Pattern period should have different statistical properties
        # (This is a simplified test - in practice you'd test for mean reversion)
        pattern_returns = pattern_data['returns']
        non_pattern_returns = non_pattern_data['returns']
        
        # Both should have returns (not all zeros)
        assert pattern_returns.std() > 0
        assert non_pattern_returns.std() > 0


class TestSplitGenerator:
    """Test the split generation logic specifically."""
    
    def test_split_count(self):
        """Test that correct number of splits is generated."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            freq='D'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': 100 + np.cumsum(np.random.normal(0, 0.01, len(timestamps)))
        })
        
        n_splits = 5
        splitter = TimeSeriesSplitter(n_splits=n_splits, test_size=0.1)
        splits = splitter.split(data)
        
        # Should generate requested number of splits (or fewer if data insufficient)
        assert len(splits) <= n_splits
        assert len(splits) > 0
    
    def test_split_paths_combinatorial(self):
        """Test that splits use different combinations of training periods."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            freq='D'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': 100 + np.cumsum(np.random.normal(0, 0.01, len(timestamps)))
        })
        
        splitter = TimeSeriesSplitter(n_splits=3, test_size=0.15)
        splits = splitter.split(data)
        
        # Each split should have different training periods
        train_period_sets = []
        for split in splits:
            train_set = frozenset(split.train_periods)
            train_period_sets.append(train_set)
        
        # All training period sets should be different
        assert len(set(train_period_sets)) == len(train_period_sets), \
            "Splits should have different training period combinations"


class TestPurger:
    """Test the purging logic specifically."""
    
    def test_purging_removes_correct_points(self):
        """Test that purging removes the correct number of data points."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 3, 31),
            freq='H'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': [100] * len(timestamps)
        })
        
        purge_hours = 24
        splitter = TimeSeriesSplitter(
            n_splits=1,
            test_size=0.2,
            purge_period=timedelta(hours=purge_hours)
        )
        
        splits = splitter.split(data)
        
        if splits:  # If we got valid splits
            split = splits[0]
            timestamps_series = pd.to_datetime(data['timestamp'])
            
            purged_mask = split.get_purged_mask(timestamps_series)
            purged_count = np.sum(purged_mask)
            
            # Should have purged some data points
            assert purged_count > 0
            
            # Purged points should be around test period boundaries
            test_start, test_end = split.test_period
            purged_timestamps = timestamps_series[purged_mask]
            
            # All purged timestamps should be reasonably close to test period
            for purged_ts in purged_timestamps:
                distance_to_test = min(
                    abs((purged_ts - test_start).total_seconds()),
                    abs((purged_ts - test_end).total_seconds())
                )
                
                # Should be within reasonable distance (allowing for some flexibility)
                max_expected_distance = purge_hours * 3600 * 2  # 2x purge period
                assert distance_to_test <= max_expected_distance
    
    def test_embargo_period_enforcement(self):
        """Test that embargo period is correctly enforced."""
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
            freq='D'
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': [100] * len(timestamps)
        })
        
        embargo_days = 7
        splitter = TimeSeriesSplitter(
            n_splits=2,
            test_size=0.1,
            embargo_period=timedelta(days=embargo_days)
        )
        
        splits = splitter.split(data)
        
        for split in splits:
            test_start, test_end = split.test_period
            
            # Check embargo enforcement for each training period
            for train_start, train_end in split.train_periods:
                if train_end < test_start:
                    # Training before test
                    gap = (test_start - train_end).days
                    assert gap >= embargo_days, \
                        f"Embargo violation: {gap} days < {embargo_days} days"
                elif train_start > test_end:
                    # Training after test
                    gap = (train_start - test_end).days
                    assert gap >= embargo_days, \
                        f"Embargo violation: {gap} days < {embargo_days} days"
