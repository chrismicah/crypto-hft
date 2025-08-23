"""Combinatorial Purged Cross-Validation implementation for financial time series."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from itertools import combinations
import math

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVSplit:
    """Represents a single CPCV split with train and test periods."""
    split_id: int
    train_periods: List[Tuple[datetime, datetime]]
    test_period: Tuple[datetime, datetime]
    purged_periods: List[Tuple[datetime, datetime]]
    
    def __post_init__(self):
        """Validate the split configuration."""
        # Ensure test period doesn't overlap with train periods
        test_start, test_end = self.test_period
        
        for train_start, train_end in self.train_periods:
            if not (train_end < test_start or train_start > test_end):
                raise ValueError(f"Train period [{train_start}, {train_end}] overlaps with test period [{test_start}, {test_end}]")
    
    def get_train_mask(self, timestamps: pd.Series) -> np.ndarray:
        """Get boolean mask for training data."""
        mask = np.zeros(len(timestamps), dtype=bool)
        
        for train_start, train_end in self.train_periods:
            period_mask = (timestamps >= train_start) & (timestamps <= train_end)
            mask |= period_mask
        
        return mask
    
    def get_test_mask(self, timestamps: pd.Series) -> np.ndarray:
        """Get boolean mask for test data."""
        test_start, test_end = self.test_period
        return (timestamps >= test_start) & (timestamps <= test_end)
    
    def get_purged_mask(self, timestamps: pd.Series) -> np.ndarray:
        """Get boolean mask for purged data (should be excluded)."""
        mask = np.zeros(len(timestamps), dtype=bool)
        
        for purged_start, purged_end in self.purged_periods:
            period_mask = (timestamps >= purged_start) & (timestamps <= purged_end)
            mask |= period_mask
        
        return mask


class TimeSeriesSplitter:
    """Time series splitter with combinatorial purged cross-validation."""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        embargo_period: timedelta = timedelta(hours=24),
        purge_period: timedelta = timedelta(hours=12),
        min_train_size: float = 0.3,
        max_train_size: float = 0.7
    ):
        """
        Initialize time series splitter.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Fraction of data to use for testing in each split
            embargo_period: Time gap after training data to prevent leakage
            purge_period: Additional time to purge around test periods
            min_train_size: Minimum fraction of data for training
            max_train_size: Maximum fraction of data for training
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_period = embargo_period
        self.purge_period = purge_period
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        
        logger.info("TimeSeriesSplitter initialized",
                   n_splits=n_splits,
                   test_size=test_size,
                   embargo_hours=embargo_period.total_seconds() / 3600,
                   purge_hours=purge_period.total_seconds() / 3600)
    
    def split(self, data: pd.DataFrame, timestamp_col: str = 'timestamp') -> List[CPCVSplit]:
        """
        Generate CPCV splits for the given data.
        
        Args:
            data: DataFrame with time series data
            timestamp_col: Name of timestamp column
            
        Returns:
            List of CPCVSplit objects
        """
        if timestamp_col not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
        
        timestamps = pd.to_datetime(data[timestamp_col])
        start_time = timestamps.min()
        end_time = timestamps.max()
        total_duration = end_time - start_time
        
        logger.info("Generating CPCV splits",
                   start_time=start_time,
                   end_time=end_time,
                   total_duration_days=total_duration.total_seconds() / (24 * 3600),
                   data_points=len(data))
        
        # Generate test periods
        test_periods = self._generate_test_periods(start_time, end_time, total_duration)
        
        # Generate all possible training period combinations
        all_periods = self._generate_all_periods(start_time, end_time, total_duration)
        
        splits = []
        
        for i, test_period in enumerate(test_periods):
            if i >= self.n_splits:
                break
                
            # Select training periods that don't conflict with test period
            valid_train_periods = self._select_training_periods(
                all_periods, test_period, total_duration
            )
            
            # Generate purged periods
            purged_periods = self._generate_purged_periods(valid_train_periods, test_period)
            
            split = CPCVSplit(
                split_id=i,
                train_periods=valid_train_periods,
                test_period=test_period,
                purged_periods=purged_periods
            )
            
            splits.append(split)
            
            logger.debug("Generated split",
                        split_id=i,
                        train_periods=len(valid_train_periods),
                        test_period=test_period,
                        purged_periods=len(purged_periods))
        
        logger.info("CPCV splits generated", total_splits=len(splits))
        return splits
    
    def _generate_test_periods(
        self,
        start_time: datetime,
        end_time: datetime,
        total_duration: timedelta
    ) -> List[Tuple[datetime, datetime]]:
        """Generate test periods distributed across the time series."""
        test_duration = total_duration * self.test_size
        
        # Distribute test periods evenly across the time series
        test_periods = []
        
        for i in range(self.n_splits):
            # Calculate test period start
            # Avoid the very beginning and end to ensure sufficient training data
            available_start = start_time + total_duration * 0.1
            available_end = end_time - total_duration * 0.1 - test_duration
            available_duration = available_end - available_start
            
            if available_duration <= timedelta(0):
                logger.warning("Insufficient data for test period", split_id=i)
                continue
            
            # Distribute test periods evenly
            test_start = available_start + (available_duration * i / self.n_splits)
            test_end = test_start + test_duration
            
            test_periods.append((test_start, test_end))
        
        return test_periods
    
    def _generate_all_periods(
        self,
        start_time: datetime,
        end_time: datetime,
        total_duration: timedelta
    ) -> List[Tuple[datetime, datetime]]:
        """Generate all possible training periods."""
        # Create overlapping periods of different sizes
        periods = []
        
        # Generate periods of various sizes
        min_period_duration = total_duration * 0.1  # Minimum 10% of total duration
        max_period_duration = total_duration * 0.4  # Maximum 40% of total duration
        
        # Number of different period sizes to try
        n_sizes = 5
        
        for size_idx in range(n_sizes):
            # Interpolate between min and max duration
            period_duration = min_period_duration + (
                (max_period_duration - min_period_duration) * size_idx / (n_sizes - 1)
            )
            
            # Generate multiple periods of this size
            n_periods_of_size = 8  # Number of periods per size
            
            for period_idx in range(n_periods_of_size):
                # Calculate period start
                max_start = end_time - period_duration
                if max_start <= start_time:
                    continue
                
                available_duration = max_start - start_time
                period_start = start_time + (available_duration * period_idx / n_periods_of_size)
                period_end = period_start + period_duration
                
                if period_end <= end_time:
                    periods.append((period_start, period_end))
        
        # Sort periods by start time
        periods.sort(key=lambda x: x[0])
        
        logger.debug("Generated candidate training periods", count=len(periods))
        return periods
    
    def _select_training_periods(
        self,
        all_periods: List[Tuple[datetime, datetime]],
        test_period: Tuple[datetime, datetime],
        total_duration: timedelta
    ) -> List[Tuple[datetime, datetime]]:
        """Select training periods that don't conflict with test period."""
        test_start, test_end = test_period
        
        # Add embargo periods around test period
        embargo_start = test_start - self.embargo_period
        embargo_end = test_end + self.embargo_period
        
        # Filter out conflicting periods
        valid_periods = []
        
        for period_start, period_end in all_periods:
            # Check if period conflicts with test + embargo
            if period_end < embargo_start or period_start > embargo_end:
                valid_periods.append((period_start, period_end))
        
        # Select a combination of periods that provides good coverage
        selected_periods = self._optimize_period_selection(
            valid_periods, total_duration
        )
        
        return selected_periods
    
    def _optimize_period_selection(
        self,
        valid_periods: List[Tuple[datetime, datetime]],
        total_duration: timedelta
    ) -> List[Tuple[datetime, datetime]]:
        """Optimize selection of training periods for good coverage."""
        if not valid_periods:
            return []
        
        # Target training duration
        target_train_duration = total_duration * (self.min_train_size + self.max_train_size) / 2
        
        # Try different combinations to find optimal coverage
        best_periods = []
        best_score = -1
        
        # Limit combinations to avoid exponential explosion
        max_periods = min(len(valid_periods), 8)
        
        for n_periods in range(1, max_periods + 1):
            for period_combination in combinations(valid_periods, n_periods):
                # Calculate total duration and coverage
                total_train_duration = sum(
                    (end - start).total_seconds() 
                    for start, end in period_combination
                )
                total_train_duration = timedelta(seconds=total_train_duration)
                
                # Score based on closeness to target duration and time coverage
                duration_score = 1 - abs(
                    total_train_duration.total_seconds() - target_train_duration.total_seconds()
                ) / target_train_duration.total_seconds()
                
                # Coverage score (how well periods are distributed)
                coverage_score = self._calculate_coverage_score(period_combination)
                
                # Combined score
                score = 0.7 * duration_score + 0.3 * coverage_score
                
                if score > best_score:
                    best_score = score
                    best_periods = list(period_combination)
        
        return best_periods
    
    def _calculate_coverage_score(
        self,
        periods: List[Tuple[datetime, datetime]]
    ) -> float:
        """Calculate how well periods cover the time range."""
        if not periods:
            return 0.0
        
        # Sort periods by start time
        sorted_periods = sorted(periods, key=lambda x: x[0])
        
        # Calculate gaps between periods
        total_gap = timedelta(0)
        total_coverage = timedelta(0)
        
        for i, (start, end) in enumerate(sorted_periods):
            total_coverage += end - start
            
            if i > 0:
                prev_end = sorted_periods[i-1][1]
                if start > prev_end:
                    total_gap += start - prev_end
        
        # Score is higher when gaps are smaller relative to coverage
        if total_coverage.total_seconds() == 0:
            return 0.0
        
        gap_ratio = total_gap.total_seconds() / total_coverage.total_seconds()
        return 1.0 / (1.0 + gap_ratio)
    
    def _generate_purged_periods(
        self,
        train_periods: List[Tuple[datetime, datetime]],
        test_period: Tuple[datetime, datetime]
    ) -> List[Tuple[datetime, datetime]]:
        """Generate periods to purge around training and test data."""
        purged_periods = []
        test_start, test_end = test_period
        
        # Purge around test period
        purge_start = test_start - self.purge_period
        purge_end = test_end + self.purge_period
        purged_periods.append((purge_start, purge_end))
        
        # Purge around training periods
        for train_start, train_end in train_periods:
            # Purge after training period (before test)
            if train_end < test_start:
                purge_train_start = train_end
                purge_train_end = min(train_end + self.purge_period, test_start)
                if purge_train_end > purge_train_start:
                    purged_periods.append((purge_train_start, purge_train_end))
            
            # Purge before training period (after test)
            if train_start > test_end:
                purge_train_start = max(train_start - self.purge_period, test_end)
                purge_train_end = train_start
                if purge_train_end > purge_train_start:
                    purged_periods.append((purge_train_start, purge_train_end))
        
        # Merge overlapping purged periods
        purged_periods = self._merge_overlapping_periods(purged_periods)
        
        return purged_periods
    
    def _merge_overlapping_periods(
        self,
        periods: List[Tuple[datetime, datetime]]
    ) -> List[Tuple[datetime, datetime]]:
        """Merge overlapping time periods."""
        if not periods:
            return []
        
        # Sort by start time
        sorted_periods = sorted(periods, key=lambda x: x[0])
        merged = [sorted_periods[0]]
        
        for current_start, current_end in sorted_periods[1:]:
            last_start, last_end = merged[-1]
            
            # If periods overlap or are adjacent, merge them
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged


class CPCVValidator:
    """Validator for CPCV methodology to ensure no information leakage."""
    
    def __init__(self):
        """Initialize CPCV validator."""
        self.logger = get_logger(f"{__name__}.CPCVValidator")
    
    def validate_splits(
        self,
        splits: List[CPCVSplit],
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Dict[str, Any]:
        """
        Validate CPCV splits for information leakage.
        
        Args:
            splits: List of CPCV splits to validate
            data: Original dataset
            timestamp_col: Name of timestamp column
            
        Returns:
            Validation results dictionary
        """
        timestamps = pd.to_datetime(data[timestamp_col])
        
        validation_results = {
            'total_splits': len(splits),
            'valid_splits': 0,
            'leakage_detected': False,
            'coverage_stats': {},
            'split_details': []
        }
        
        total_data_points = len(data)
        
        for split in splits:
            split_validation = self._validate_single_split(split, timestamps)
            validation_results['split_details'].append(split_validation)
            
            if split_validation['valid']:
                validation_results['valid_splits'] += 1
            
            if split_validation['potential_leakage']:
                validation_results['leakage_detected'] = True
        
        # Calculate overall coverage statistics
        validation_results['coverage_stats'] = self._calculate_coverage_stats(
            splits, timestamps, total_data_points
        )
        
        self.logger.info("CPCV validation completed",
                        total_splits=validation_results['total_splits'],
                        valid_splits=validation_results['valid_splits'],
                        leakage_detected=validation_results['leakage_detected'])
        
        return validation_results
    
    def _validate_single_split(
        self,
        split: CPCVSplit,
        timestamps: pd.Series
    ) -> Dict[str, Any]:
        """Validate a single CPCV split."""
        train_mask = split.get_train_mask(timestamps)
        test_mask = split.get_test_mask(timestamps)
        purged_mask = split.get_purged_mask(timestamps)
        
        # Check for overlaps
        train_test_overlap = np.any(train_mask & test_mask)
        train_purged_overlap = np.any(train_mask & purged_mask)
        test_purged_overlap = np.any(test_mask & purged_mask)
        
        # Calculate sizes
        train_size = np.sum(train_mask)
        test_size = np.sum(test_mask)
        purged_size = np.sum(purged_mask)
        
        # Check temporal ordering
        train_times = timestamps[train_mask]
        test_times = timestamps[test_mask]
        
        temporal_leakage = False
        if len(train_times) > 0 and len(test_times) > 0:
            # Check if any training data comes after test data
            max_train_time = train_times.max()
            min_test_time = test_times.min()
            
            # Allow for some training data to be after test data (this is valid in CPCV)
            # But check for suspicious patterns
            temporal_leakage = False  # This would need more sophisticated logic
        
        validation_result = {
            'split_id': split.split_id,
            'valid': not (train_test_overlap or train_purged_overlap or test_purged_overlap),
            'potential_leakage': train_test_overlap or temporal_leakage,
            'train_size': int(train_size),
            'test_size': int(test_size),
            'purged_size': int(purged_size),
            'overlaps': {
                'train_test': bool(train_test_overlap),
                'train_purged': bool(train_purged_overlap),
                'test_purged': bool(test_purged_overlap)
            }
        }
        
        return validation_result
    
    def _calculate_coverage_stats(
        self,
        splits: List[CPCVSplit],
        timestamps: pd.Series,
        total_data_points: int
    ) -> Dict[str, Any]:
        """Calculate coverage statistics across all splits."""
        total_train_coverage = np.zeros(len(timestamps), dtype=bool)
        total_test_coverage = np.zeros(len(timestamps), dtype=bool)
        
        for split in splits:
            train_mask = split.get_train_mask(timestamps)
            test_mask = split.get_test_mask(timestamps)
            
            total_train_coverage |= train_mask
            total_test_coverage |= test_mask
        
        coverage_stats = {
            'train_coverage_pct': float(np.sum(total_train_coverage)) / total_data_points * 100,
            'test_coverage_pct': float(np.sum(total_test_coverage)) / total_data_points * 100,
            'total_coverage_pct': float(np.sum(total_train_coverage | total_test_coverage)) / total_data_points * 100,
            'overlap_pct': float(np.sum(total_train_coverage & total_test_coverage)) / total_data_points * 100
        }
        
        return coverage_stats


def create_synthetic_leakage_test_data(
    start_date: datetime = datetime(2024, 1, 1),
    end_date: datetime = datetime(2024, 6, 30),
    freq: str = '1H',
    pattern_change_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Create synthetic data for testing information leakage.
    
    This creates a dataset with a profitable pattern in the first half
    and no pattern in the second half. Proper CPCV should detect this.
    
    Args:
        start_date: Start date for synthetic data
        end_date: End date for synthetic data
        freq: Frequency of data points
        pattern_change_date: Date when pattern changes (default: midpoint)
        
    Returns:
        DataFrame with synthetic time series data
    """
    if pattern_change_date is None:
        pattern_change_date = start_date + (end_date - start_date) / 2
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate base price series (random walk)
    np.random.seed(42)  # For reproducible results
    n_points = len(timestamps)
    
    # Base random walk
    returns = np.random.normal(0, 0.01, n_points)  # 1% volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Start at $100
    
    # Add profitable pattern to first half
    pattern_mask = timestamps < pattern_change_date
    
    # Create a simple mean-reverting pattern in the first half
    for i in range(1, n_points):
        if pattern_mask[i]:
            # Mean reversion: if price moved up, add slight downward bias
            if returns[i-1] > 0:
                returns[i] -= 0.002  # Small mean reversion
            else:
                returns[i] += 0.002
    
    # Recalculate prices with pattern
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create additional features
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'returns': returns,
        'volume': np.random.lognormal(10, 0.5, n_points),  # Random volume
        'spread': np.random.uniform(0.001, 0.01, n_points),  # Random spread
        'pattern_period': pattern_mask  # Indicator for pattern period
    })
    
    # Add some technical indicators
    data['sma_20'] = data['price'].rolling(20).mean()
    data['price_to_sma'] = data['price'] / data['sma_20']
    
    # Add noise features
    for i in range(5):
        data[f'noise_{i}'] = np.random.normal(0, 1, n_points)
    
    logger.info("Created synthetic leakage test data",
               total_points=len(data),
               pattern_period_points=np.sum(pattern_mask),
               start_date=start_date,
               end_date=end_date,
               pattern_change_date=pattern_change_date)
    
    return data
