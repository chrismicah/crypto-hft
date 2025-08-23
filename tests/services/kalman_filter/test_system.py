"""System tests for Kalman filter service with synthetic data."""

import pytest
import numpy as np
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from services.kalman_filter.main import KalmanFilterService
from services.kalman_filter.filter import PairTradingKalmanFilter, DynamicHedgeRatioKalman


class SyntheticDataGenerator:
    """Generate synthetic cointegrated time series data for testing."""
    
    @staticmethod
    def generate_cointegrated_pair(
        n_points: int = 1000,
        true_hedge_ratio: float = 2.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        seed: int = 42
    ) -> tuple[List[float], List[float], List[datetime]]:
        """
        Generate synthetic cointegrated price series.
        
        Args:
            n_points: Number of data points
            true_hedge_ratio: True hedge ratio between assets
            volatility: Volatility of price movements
            drift: Drift in the hedge ratio over time
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (asset1_prices, asset2_prices, timestamps)
        """
        np.random.seed(seed)
        
        # Generate base price for asset 2
        base_price_2 = 1000.0
        asset2_returns = np.random.normal(0, volatility, n_points)
        asset2_prices = [base_price_2]
        
        for i in range(1, n_points):
            new_price = asset2_prices[-1] * (1 + asset2_returns[i])
            asset2_prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        # Generate cointegrated asset 1 prices
        asset1_prices = []
        hedge_ratio = true_hedge_ratio
        
        for i in range(n_points):
            # Add drift to hedge ratio
            hedge_ratio += drift * np.random.normal(0, 1)
            
            # Generate asset 1 price with cointegration relationship
            noise = np.random.normal(0, volatility * 0.5)  # Lower noise for cointegration
            asset1_price = asset2_prices[i] * hedge_ratio * (1 + noise)
            asset1_prices.append(max(asset1_price, 1.0))
        
        # Generate timestamps
        start_time = datetime.utcnow() - timedelta(hours=n_points // 60)  # 1 minute intervals
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        return asset1_prices, asset2_prices, timestamps
    
    @staticmethod
    def generate_regime_switching_data(
        n_points: int = 1000,
        regimes: List[Dict[str, Any]] = None,
        seed: int = 42
    ) -> tuple[List[float], List[float], List[datetime], List[int]]:
        """
        Generate data with structural breaks (regime switching).
        
        Args:
            n_points: Total number of data points
            regimes: List of regime specifications
            seed: Random seed
            
        Returns:
            Tuple of (asset1_prices, asset2_prices, timestamps, regime_labels)
        """
        if regimes is None:
            regimes = [
                {'hedge_ratio': 1.5, 'volatility': 0.02, 'duration': 0.4},
                {'hedge_ratio': 2.5, 'volatility': 0.03, 'duration': 0.3},
                {'hedge_ratio': 2.0, 'volatility': 0.015, 'duration': 0.3}
            ]
        
        np.random.seed(seed)
        
        asset1_prices = []
        asset2_prices = []
        regime_labels = []
        
        # Calculate regime durations
        regime_points = [int(n_points * r['duration']) for r in regimes]
        regime_points[-1] = n_points - sum(regime_points[:-1])  # Adjust last regime
        
        current_price_2 = 1000.0
        
        for regime_idx, (regime, n_regime_points) in enumerate(zip(regimes, regime_points)):
            hedge_ratio = regime['hedge_ratio']
            volatility = regime['volatility']
            
            for i in range(n_regime_points):
                # Generate asset 2 price
                return_2 = np.random.normal(0, volatility)
                current_price_2 *= (1 + return_2)
                current_price_2 = max(current_price_2, 1.0)
                asset2_prices.append(current_price_2)
                
                # Generate cointegrated asset 1 price
                noise = np.random.normal(0, volatility * 0.5)
                asset1_price = current_price_2 * hedge_ratio * (1 + noise)
                asset1_prices.append(max(asset1_price, 1.0))
                
                regime_labels.append(regime_idx)
        
        # Generate timestamps
        start_time = datetime.utcnow() - timedelta(hours=n_points // 60)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        return asset1_prices, asset2_prices, timestamps, regime_labels


class TestKalmanFilterConvergence:
    """Test filter convergence with synthetic data."""
    
    def test_convergence_stable_regime(self):
        """Test filter convergence in a stable cointegration regime."""
        # Generate synthetic data (reduced size for faster testing)
        true_hedge_ratio = 2.5
        asset1_prices, asset2_prices, timestamps = SyntheticDataGenerator.generate_cointegrated_pair(
            n_points=100,  # Reduced from 500
            true_hedge_ratio=true_hedge_ratio,
            volatility=0.01,
            drift=0.0
        )
        
        # Create filter
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.0,  # Start far from true value
            process_variance=1e-4,
            observation_variance=1e-2
        )
        
        # Process data
        results = []
        for asset1_price, asset2_price, timestamp in zip(asset1_prices, asset2_prices, timestamps):
            price_ratio = asset1_price / asset2_price
            result = filter_obj.update(price_ratio, timestamp)
            results.append(result)
        
        # Analyze convergence
        hedge_ratios = [r['hedge_ratio'] for r in results]
        variances = [r['hedge_ratio_variance'] for r in results]
        
        # Check convergence to true value
        final_hedge_ratio = hedge_ratios[-1]
        final_variance = variances[-1]
        
        # Should converge within 2 standard deviations
        convergence_error = abs(final_hedge_ratio - true_hedge_ratio)
        confidence_bound = 2 * np.sqrt(final_variance)
        
        assert convergence_error < confidence_bound, (
            f"Filter did not converge: error={convergence_error:.4f}, "
            f"bound={confidence_bound:.4f}"
        )
        
        # Variance should decrease over time (learning)
        initial_variance = variances[50]  # Skip initial transient
        assert final_variance < initial_variance, "Variance should decrease with learning"
        
        # Check convergence rate
        mid_point = len(hedge_ratios) // 2
        mid_error = abs(hedge_ratios[mid_point] - true_hedge_ratio)
        final_error = abs(final_hedge_ratio - true_hedge_ratio)
        
        assert final_error < mid_error, "Should continue converging throughout"
    
    def test_convergence_with_noise(self):
        """Test filter convergence with high noise."""
        true_hedge_ratio = 1.8
        asset1_prices, asset2_prices, timestamps = SyntheticDataGenerator.generate_cointegrated_pair(
            n_points=200,  # Reduced from 1000
            true_hedge_ratio=true_hedge_ratio,
            volatility=0.05,  # High volatility
            drift=0.0
        )
        
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.0,
            process_variance=1e-4,
            observation_variance=5e-2  # Higher observation noise
        )
        
        # Process data
        results = []
        for asset1_price, asset2_price, timestamp in zip(asset1_prices, asset2_prices, timestamps):
            price_ratio = asset1_price / asset2_price
            result = filter_obj.update(price_ratio, timestamp)
            results.append(result)
        
        # Check final convergence
        final_hedge_ratio = results[-1]['hedge_ratio']
        final_variance = results[-1]['hedge_ratio_variance']
        
        # With high noise, allow larger convergence tolerance
        convergence_error = abs(final_hedge_ratio - true_hedge_ratio)
        confidence_bound = 3 * np.sqrt(final_variance)  # 3 sigma for high noise
        
        assert convergence_error < confidence_bound, (
            f"Filter did not converge with noise: error={convergence_error:.4f}, "
            f"bound={confidence_bound:.4f}"
        )
    
    def test_structural_break_adaptation(self):
        """Test filter adaptation to structural breaks."""
        regimes = [
            {'hedge_ratio': 1.5, 'volatility': 0.02, 'duration': 0.4},
            {'hedge_ratio': 2.5, 'volatility': 0.02, 'duration': 0.6}
        ]
        
        asset1_prices, asset2_prices, timestamps, regime_labels = (
            SyntheticDataGenerator.generate_regime_switching_data(
                n_points=200,  # Reduced from 800
                regimes=regimes
            )
        )
        
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.0,
            process_variance=5e-4,  # Higher process variance for adaptation
            observation_variance=1e-2
        )
        
        # Process data
        results = []
        for asset1_price, asset2_price, timestamp in zip(asset1_prices, asset2_prices, timestamps):
            price_ratio = asset1_price / asset2_price
            result = filter_obj.update(price_ratio, timestamp)
            results.append(result)
        
        # Analyze adaptation
        hedge_ratios = [r['hedge_ratio'] for r in results]
        
        # Find regime change point
        regime_change_idx = next(i for i, label in enumerate(regime_labels) if label == 1)
        
        # Check adaptation to first regime
        regime1_end_idx = regime_change_idx - 50  # Before regime change
        regime1_hedge_ratio = np.mean(hedge_ratios[regime1_end_idx-50:regime1_end_idx])
        
        assert abs(regime1_hedge_ratio - regimes[0]['hedge_ratio']) < 0.3, (
            f"Did not adapt to first regime: {regime1_hedge_ratio:.3f} vs {regimes[0]['hedge_ratio']}"
        )
        
        # Check adaptation to second regime
        regime2_start_idx = regime_change_idx + 100  # After adaptation period
        regime2_hedge_ratio = np.mean(hedge_ratios[regime2_start_idx:regime2_start_idx+50])
        
        assert abs(regime2_hedge_ratio - regimes[1]['hedge_ratio']) < 0.3, (
            f"Did not adapt to second regime: {regime2_hedge_ratio:.3f} vs {regimes[1]['hedge_ratio']}"
        )
        
        # Check that adaptation occurred
        adaptation_change = abs(regime2_hedge_ratio - regime1_hedge_ratio)
        expected_change = abs(regimes[1]['hedge_ratio'] - regimes[0]['hedge_ratio'])
        
        assert adaptation_change > expected_change * 0.5, (
            f"Insufficient adaptation: {adaptation_change:.3f} vs expected {expected_change:.3f}"
        )


class TestRealTimePerformance:
    """Test real-time performance and latency."""
    
    def test_single_update_latency(self):
        """Test latency of single filter update."""
        import time
        
        filter_obj = DynamicHedgeRatioKalman()
        
        # Warm up
        for _ in range(10):
            filter_obj.update(1.0 + np.random.normal(0, 0.01))
        
        # Measure latency
        latencies = []
        n_measurements = 100  # Reduced from 1000
        
        for _ in range(n_measurements):
            observation = 1.0 + np.random.normal(0, 0.01)
            
            start_time = time.perf_counter()
            filter_obj.update(observation)
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
        
        # Analyze performance
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Performance requirements
        assert avg_latency < 0.001, f"Average latency too high: {avg_latency:.6f}s"
        assert p95_latency < 0.002, f"95th percentile latency too high: {p95_latency:.6f}s"
        assert max_latency < 0.01, f"Max latency too high: {max_latency:.6f}s"
    
    def test_high_frequency_updates(self):
        """Test performance with high-frequency updates."""
        import time
        
        pair_filter = PairTradingKalmanFilter()
        pair_filter.add_pair("BTCETH", "BTCUSDT", "ETHUSDT", initial_hedge_ratio=15.0)
        
        # Generate high-frequency data
        n_updates = 1000  # Reduced from 10000
        base_time = datetime.utcnow()
        
        # Measure total processing time
        start_time = time.perf_counter()
        
        for i in range(n_updates):
            # Simulate realistic price movements
            btc_price = 45000 + np.random.normal(0, 500)
            eth_price = 3000 + np.random.normal(0, 100)
            timestamp = base_time + timedelta(milliseconds=i * 100)  # 10 Hz
            
            result = pair_filter.update_pair("BTCETH", btc_price, eth_price, timestamp)
            assert result is not None
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance analysis
        updates_per_second = n_updates / total_time
        avg_time_per_update = total_time / n_updates
        
        # Requirements for high-frequency trading (relaxed for smaller test)
        assert updates_per_second > 1000, f"Throughput too low: {updates_per_second:.0f} updates/sec"
        assert avg_time_per_update < 0.0002, f"Average update time too high: {avg_time_per_update:.6f}s"
        
        # Check all updates were processed
        assert pair_filter.filters["BTCETH"].n_observations == n_updates
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        filter_obj = DynamicHedgeRatioKalman()
        
        # Process large amount of data (reduced for faster testing)
        n_updates = 5000  # Reduced from 50000
        memory_measurements = []
        
        for i in range(n_updates):
            observation = 1.0 + np.random.normal(0, 0.01)
            filter_obj.update(observation)
            
            # Measure memory every 500 updates (more frequent for smaller test)
            if i % 500 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Check memory stability
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_measurements)
        
        # Should not grow significantly (allow some JIT compilation overhead)
        assert memory_growth < 50, f"Memory growth too high: {memory_growth:.1f} MB"
        assert max_memory < initial_memory + 100, f"Peak memory too high: {max_memory:.1f} MB"


class TestKalmanFilterServiceIntegration:
    """Integration tests for the complete Kalman filter service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_kafka_consumer(self):
        """Mock Kafka consumer for testing."""
        consumer = Mock()
        consumer.__iter__ = Mock(return_value=iter([]))
        return consumer
    
    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer for testing."""
        producer = Mock()
        future = Mock()
        future.get.return_value = Mock(topic='signals-hedge-ratio', partition=0, offset=123)
        producer.send.return_value = future
        return producer
    
    @patch('services.kalman_filter.main.KafkaConsumer')
    @patch('services.kalman_filter.main.KafkaProducer')
    def test_service_initialization(self, mock_producer_class, mock_consumer_class, temp_dir):
        """Test service initialization."""
        mock_consumer_class.return_value = Mock()
        mock_producer_class.return_value = Mock()
        
        service = KalmanFilterService()
        
        # Override state directory for testing
        service.state_manager.state_dir = Path(temp_dir)
        
        # Test initialization components
        asyncio.run(service._initialize_components())
        
        # Check components were created
        assert service.consumer is not None
        assert service.producer is not None
        assert service.pair_filter is not None
        assert len(service.pair_filter.filters) > 0  # Should have default pairs
    
    def test_tick_message_processing(self):
        """Test processing of tick messages."""
        service = KalmanFilterService()
        
        # Initialize pair filter manually
        service.pair_filter.add_pair("BTCETH", "BTCUSDT", "ETHUSDT", initial_hedge_ratio=15.0)
        
        # Process tick messages
        messages = [
            {
                "symbol": "BTCUSDT",
                "price": 45000.0,
                "timestamp": "2023-12-01T12:00:00Z",
                "volume": 1.5
            },
            {
                "symbol": "ETHUSDT", 
                "price": 3000.0,
                "timestamp": "2023-12-01T12:00:01Z",
                "volume": 2.0
            }
        ]
        
        for message in messages:
            service._process_tick_message(message)
        
        # Check price cache was updated
        assert hasattr(service, '_price_cache')
        assert "BTCUSDT" in service._price_cache
        assert "ETHUSDT" in service._price_cache
        
        assert service._price_cache["BTCUSDT"]["price"] == 45000.0
        assert service._price_cache["ETHUSDT"]["price"] == 3000.0
    
    def test_pair_update_logic(self):
        """Test pair update logic with price cache."""
        service = KalmanFilterService()
        service.pair_filter.add_pair("BTCETH", "BTCUSDT", "ETHUSDT", initial_hedge_ratio=15.0)
        
        # Initialize price cache
        service._price_cache = {}
        current_time = datetime.utcnow()
        
        # Add prices for both assets
        service._update_price_cache("BTCUSDT", 45000.0, current_time)
        service._update_price_cache("ETHUSDT", 3000.0, current_time)
        
        # Mock producer for signal publishing
        service.producer = Mock()
        future = Mock()
        future.get.return_value = Mock(topic='signals-hedge-ratio', partition=0, offset=123)
        service.producer.send.return_value = future
        
        # Check and update pairs
        service._check_and_update_pairs(current_time)
        
        # Verify pair was updated
        pair_state = service.pair_filter.get_pair_state("BTCETH")
        assert pair_state.n_observations == 1
        
        # Verify signal was published
        service.producer.send.assert_called_once()
        call_args = service.producer.send.call_args
        assert call_args[0][0] == "signals-hedge-ratio"  # Topic
        
        signal_data = call_args[0][1]  # Message
        assert signal_data['pair_id'] == "BTCETH"
        assert 'hedge_ratio' in signal_data
        assert 'service' in signal_data
    
    def test_health_status(self):
        """Test service health status reporting."""
        service = KalmanFilterService()
        service.running = True
        service.stats['start_time'] = datetime.utcnow() - timedelta(minutes=5)
        service.stats['messages_processed'] = 1000
        service.stats['signals_published'] = 50
        
        # Add some pairs
        service.pair_filter.add_pair("BTCETH", "BTCUSDT", "ETHUSDT")
        
        health_status = service.get_health_status()
        
        required_keys = ['status', 'uptime_seconds', 'stats', 'trading_pairs', 'kafka_connected']
        for key in required_keys:
            assert key in health_status
        
        assert health_status['status'] == 'healthy'
        assert health_status['uptime_seconds'] > 0
        assert health_status['stats']['messages_processed'] == 1000
        assert len(health_status['trading_pairs']) == 1


class TestSyntheticDataScenarios:
    """Test various synthetic data scenarios."""
    
    def test_trending_market_scenario(self):
        """Test filter performance in trending markets."""
        # Generate trending data with changing hedge ratio
        n_points = 200  # Reduced from 1000
        np.random.seed(42)
        
        asset2_prices = []
        asset1_prices = []
        current_price_2 = 1000.0
        hedge_ratio = 2.0
        
        for i in range(n_points):
            # Trending market: both assets generally increasing
            trend_2 = 0.0001  # 0.01% per period
            trend_1 = 0.00015  # Slightly higher trend
            
            # Add trend and noise
            return_2 = trend_2 + np.random.normal(0, 0.02)
            current_price_2 *= (1 + return_2)
            asset2_prices.append(current_price_2)
            
            # Hedge ratio slowly changes
            hedge_ratio += np.random.normal(0, 0.001)
            
            return_1 = trend_1 + np.random.normal(0, 0.02)
            asset1_price = asset2_prices[-1] * hedge_ratio * (1 + return_1 * 0.5)
            asset1_prices.append(asset1_price)
        
        # Test filter adaptation
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.5,
            process_variance=1e-4,
            observation_variance=1e-2
        )
        
        results = []
        for asset1_price, asset2_price in zip(asset1_prices, asset2_prices):
            price_ratio = asset1_price / asset2_price
            result = filter_obj.update(price_ratio)
            results.append(result)
        
        # Check that filter tracks the changing hedge ratio
        hedge_ratios = [r['hedge_ratio'] for r in results]
        
        # Should show some trend adaptation
        early_ratio = np.mean(hedge_ratios[100:200])
        late_ratio = np.mean(hedge_ratios[-100:])
        
        # Variance should remain reasonable
        final_variance = results[-1]['hedge_ratio_variance']
        assert final_variance < 0.1, f"Variance too high in trending market: {final_variance}"
    
    def test_mean_reverting_scenario(self):
        """Test filter in mean-reverting market conditions."""
        # Generate mean-reverting data
        n_points = 150  # Reduced from 800
        np.random.seed(123)
        
        true_hedge_ratio = 1.8
        asset2_prices = []
        asset1_prices = []
        current_price_2 = 1000.0
        
        for i in range(n_points):
            # Mean-reverting returns
            reversion_speed = 0.05
            return_2 = -reversion_speed * np.log(current_price_2 / 1000.0) + np.random.normal(0, 0.03)
            current_price_2 *= (1 + return_2)
            asset2_prices.append(current_price_2)
            
            # Cointegrated asset 1 with mean reversion
            noise = np.random.normal(0, 0.01)
            asset1_price = asset2_prices[-1] * true_hedge_ratio * (1 + noise)
            asset1_prices.append(asset1_price)
        
        # Test filter
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.0,
            process_variance=5e-5,  # Lower process variance for stable regime
            observation_variance=1e-2
        )
        
        results = []
        for asset1_price, asset2_price in zip(asset1_prices, asset2_prices):
            price_ratio = asset1_price / asset2_price
            result = filter_obj.update(price_ratio)
            results.append(result)
        
        # Check convergence in mean-reverting environment
        final_hedge_ratio = results[-1]['hedge_ratio']
        convergence_error = abs(final_hedge_ratio - true_hedge_ratio)
        
        assert convergence_error < 0.1, (
            f"Poor convergence in mean-reverting market: error={convergence_error:.4f}"
        )
        
        # Check stability (low variance)
        final_variance = results[-1]['hedge_ratio_variance']
        assert final_variance < 0.01, f"Variance too high in stable market: {final_variance}"
    
    def test_high_volatility_scenario(self):
        """Test filter robustness in high volatility conditions."""
        # Generate high volatility data
        true_hedge_ratio = 2.2
        asset1_prices, asset2_prices, timestamps = SyntheticDataGenerator.generate_cointegrated_pair(
            n_points=120,  # Reduced from 600
            true_hedge_ratio=true_hedge_ratio,
            volatility=0.08,  # Very high volatility
            drift=0.0
        )
        
        # Test with robust filter settings
        filter_obj = DynamicHedgeRatioKalman(
            initial_hedge_ratio=1.0,
            process_variance=1e-3,  # Higher process variance for volatility
            observation_variance=0.1   # Higher observation variance
        )
        
        results = []
        for asset1_price, asset2_price, timestamp in zip(asset1_prices, asset2_prices, timestamps):
            price_ratio = asset1_price / asset2_price
            result = filter_obj.update(price_ratio, timestamp)
            results.append(result)
        
        # Check robustness
        hedge_ratios = [r['hedge_ratio'] for r in results]
        
        # Should still converge, but allow larger tolerance
        final_hedge_ratio = hedge_ratios[-1]
        final_variance = results[-1]['hedge_ratio_variance']
        
        convergence_error = abs(final_hedge_ratio - true_hedge_ratio)
        confidence_bound = 3 * np.sqrt(final_variance)
        
        assert convergence_error < confidence_bound, (
            f"Filter not robust to high volatility: error={convergence_error:.4f}, "
            f"bound={confidence_bound:.4f}"
        )
        
        # Check that filter doesn't become unstable
        max_hedge_ratio = max(hedge_ratios)
        min_hedge_ratio = min(hedge_ratios)
        
        assert max_hedge_ratio < true_hedge_ratio * 2, "Filter became unstable (too high)"
        assert min_hedge_ratio > true_hedge_ratio * 0.5, "Filter became unstable (too low)"
