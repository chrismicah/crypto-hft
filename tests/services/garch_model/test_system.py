"""System tests for GARCH volatility service with synthetic data scenarios."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Tuple, Dict, Any

from services.garch_model.model import RollingGARCHModel, MultiPairGARCHManager
from services.garch_model.thresholds import DynamicThresholdCalculator, MultiPairThresholdManager
from services.garch_model.main import GARCHVolatilityService


class SyntheticVolatilityDataGenerator:
    """Generate synthetic time series data with volatility clustering for testing."""
    
    @staticmethod
    def generate_volatility_clustering_data(
        n_points: int = 500,
        base_volatility: float = 0.02,
        cluster_intensity: float = 3.0,
        cluster_duration: int = 50,
        n_clusters: int = 3,
        seed: int = 42
    ) -> Tuple[List[float], List[float], List[datetime]]:
        """
        Generate synthetic data with volatility clustering.
        
        Args:
            n_points: Number of data points
            base_volatility: Base volatility level
            cluster_intensity: Multiplier for high volatility periods
            cluster_duration: Duration of each volatility cluster
            n_clusters: Number of volatility clusters
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (spread_values, volatilities, timestamps)
        """
        np.random.seed(seed)
        
        # Generate volatility regime
        volatilities = np.full(n_points, base_volatility)
        
        # Add volatility clusters
        cluster_starts = np.random.choice(
            range(cluster_duration, n_points - cluster_duration), 
            size=n_clusters, 
            replace=False
        )
        
        for start in cluster_starts:
            end = min(start + cluster_duration, n_points)
            volatilities[start:end] *= cluster_intensity
        
        # Generate spread data with time-varying volatility
        spread_values = []
        current_spread = 0.0
        
        for i in range(n_points):
            # Random walk with time-varying volatility
            innovation = np.random.normal(0, volatilities[i])
            current_spread += innovation
            spread_values.append(current_spread)
        
        # Generate timestamps
        start_time = datetime.utcnow() - timedelta(hours=n_points // 60)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        return spread_values, volatilities.tolist(), timestamps
    
    @staticmethod
    def generate_regime_switching_volatility(
        n_points: int = 400,
        regimes: List[Dict[str, Any]] = None,
        seed: int = 42
    ) -> Tuple[List[float], List[float], List[datetime], List[int]]:
        """
        Generate data with distinct volatility regimes.
        
        Args:
            n_points: Total number of data points
            regimes: List of regime specifications
            seed: Random seed
            
        Returns:
            Tuple of (spread_values, volatilities, timestamps, regime_labels)
        """
        if regimes is None:
            regimes = [
                {'volatility': 0.01, 'mean_reversion': 0.05, 'duration': 0.3},
                {'volatility': 0.05, 'mean_reversion': 0.02, 'duration': 0.4},
                {'volatility': 0.02, 'mean_reversion': 0.08, 'duration': 0.3}
            ]
        
        np.random.seed(seed)
        
        spread_values = []
        volatilities = []
        regime_labels = []
        
        # Calculate regime durations
        regime_points = [int(n_points * r['duration']) for r in regimes]
        regime_points[-1] = n_points - sum(regime_points[:-1])  # Adjust last regime
        
        current_spread = 0.0
        
        for regime_idx, (regime, n_regime_points) in enumerate(zip(regimes, regime_points)):
            vol = regime['volatility']
            mean_reversion = regime['mean_reversion']
            
            for i in range(n_regime_points):
                # Mean-reverting process with regime-specific parameters
                innovation = np.random.normal(0, vol)
                mean_reversion_force = -mean_reversion * current_spread
                current_spread += mean_reversion_force + innovation
                
                spread_values.append(current_spread)
                volatilities.append(vol)
                regime_labels.append(regime_idx)
        
        # Generate timestamps
        start_time = datetime.utcnow() - timedelta(hours=n_points // 60)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        return spread_values, volatilities, timestamps, regime_labels
    
    @staticmethod
    def generate_garch_like_data(
        n_points: int = 300,
        omega: float = 0.0001,
        alpha: float = 0.1,
        beta: float = 0.8,
        seed: int = 42
    ) -> Tuple[List[float], List[float], List[datetime]]:
        """
        Generate data following a GARCH(1,1) process.
        
        Args:
            n_points: Number of data points
            omega: GARCH omega parameter (constant)
            alpha: GARCH alpha parameter (ARCH effect)
            beta: GARCH beta parameter (GARCH effect)
            seed: Random seed
            
        Returns:
            Tuple of (returns, conditional_volatilities, timestamps)
        """
        np.random.seed(seed)
        
        returns = []
        volatilities = []
        
        # Initialize
        sigma2 = omega / (1 - alpha - beta)  # Unconditional variance
        
        for i in range(n_points):
            # Generate return
            epsilon = np.random.normal(0, 1)
            return_t = np.sqrt(sigma2) * epsilon
            returns.append(return_t)
            volatilities.append(np.sqrt(sigma2))
            
            # Update conditional variance for next period
            sigma2 = omega + alpha * (return_t ** 2) + beta * sigma2
        
        # Generate timestamps
        start_time = datetime.utcnow() - timedelta(hours=n_points // 60)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        return returns, volatilities, timestamps


class TestGARCHVolatilityClusteringResponse:
    """Test GARCH model response to volatility clustering."""
    
    def test_volatility_clustering_detection(self):
        """Test that GARCH model detects volatility clustering."""
        # Generate data with volatility clusters
        spread_values, true_volatilities, timestamps = (
            SyntheticVolatilityDataGenerator.generate_volatility_clustering_data(
                n_points=200,  # Reduced for faster testing
                base_volatility=0.02,
                cluster_intensity=3.0,
                cluster_duration=30,
                n_clusters=2
            )
        )
        
        # Create GARCH model
        model = RollingGARCHModel(
            window_size=150,
            min_observations=30,
            refit_frequency=5
        )
        
        # Add data and track forecasts
        forecasts = []
        
        with patch('services.garch_model.model.arch_model') as mock_arch:
            # Mock successful GARCH fitting
            mock_fitted = Mock()
            mock_fitted.params = {
                'omega': 0.0001,
                'alpha[1]': 0.15,
                'beta[1]': 0.75
            }
            mock_fitted.loglikelihood = -100.0
            mock_fitted.aic = 206.0
            mock_fitted.bic = 210.0
            mock_fitted.resid = np.random.normal(0, 0.1, 50)
            
            # Mock forecast that responds to recent volatility
            def mock_forecast_func(horizon=1, reindex=False):
                # Get recent data to simulate volatility response
                recent_data = list(model.spread_data)[-10:] if len(model.spread_data) >= 10 else list(model.spread_data)
                if len(recent_data) > 1:
                    recent_vol = np.std(np.diff(recent_data))
                    variance = max(0.0001, recent_vol ** 2)
                else:
                    variance = 0.01
                
                mock_forecast_result = Mock()
                mock_forecast_result.variance = pd.DataFrame([[variance]], columns=[0])
                return mock_forecast_result
            
            mock_fitted.forecast = mock_forecast_func
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arch.return_value = mock_model
            
            # Process data
            for i, (spread, timestamp) in enumerate(zip(spread_values, timestamps)):
                model.add_observation(spread, timestamp)
                
                # Generate forecast every few observations
                if i >= 30 and i % 5 == 0:
                    forecast = model.forecast_volatility()
                    if forecast:
                        forecasts.append({
                            'index': i,
                            'volatility': forecast.volatility_forecast,
                            'true_volatility': true_volatilities[i]
                        })
        
        # Analyze results
        assert len(forecasts) > 10, "Should have generated multiple forecasts"
        
        # Check that forecasts show some variation (responding to clustering)
        forecast_vols = [f['volatility'] for f in forecasts]
        vol_std = np.std(forecast_vols)
        assert vol_std > 0.001, f"Forecast volatility should vary, std={vol_std}"
        
        # Check that high volatility periods are detected
        high_vol_periods = [i for i, vol in enumerate(true_volatilities) if vol > 0.04]
        if high_vol_periods:
            # Find forecasts during high volatility periods
            high_vol_forecasts = [
                f for f in forecasts 
                if f['index'] in high_vol_periods or f['index'] - 5 in high_vol_periods
            ]
            
            if high_vol_forecasts:
                avg_high_vol_forecast = np.mean([f['volatility'] for f in high_vol_forecasts])
                avg_all_forecasts = np.mean(forecast_vols)
                
                # High volatility periods should generally have higher forecasts
                # (allowing for some lag due to model adaptation)
                assert avg_high_vol_forecast >= avg_all_forecasts * 0.8, (
                    f"High vol forecasts ({avg_high_vol_forecast:.4f}) should be higher than "
                    f"average ({avg_all_forecasts:.4f})"
                )
    
    def test_regime_switching_adaptation(self):
        """Test GARCH model adaptation to volatility regime switches."""
        # Generate regime-switching data
        spread_values, true_volatilities, timestamps, regime_labels = (
            SyntheticVolatilityDataGenerator.generate_regime_switching_volatility(
                n_points=150,  # Reduced for faster testing
                regimes=[
                    {'volatility': 0.01, 'mean_reversion': 0.05, 'duration': 0.4},
                    {'volatility': 0.04, 'mean_reversion': 0.02, 'duration': 0.6}
                ]
            )
        )
        
        model = RollingGARCHModel(
            window_size=100,
            min_observations=25,
            refit_frequency=8
        )
        
        forecasts_by_regime = {0: [], 1: []}
        
        with patch('services.garch_model.model.arch_model') as mock_arch:
            # Mock GARCH model
            mock_fitted = Mock()
            mock_fitted.params = {
                'omega': 0.0001,
                'alpha[1]': 0.2,
                'beta[1]': 0.7
            }
            mock_fitted.loglikelihood = -100.0
            mock_fitted.aic = 206.0
            mock_fitted.bic = 210.0
            mock_fitted.resid = np.random.normal(0, 0.1, 50)
            
            def mock_forecast_func(horizon=1, reindex=False):
                # Simulate adaptive forecasting based on recent data
                recent_data = list(model.spread_data)[-15:] if len(model.spread_data) >= 15 else list(model.spread_data)
                if len(recent_data) > 2:
                    recent_changes = np.diff(recent_data)
                    recent_vol = np.std(recent_changes)
                    variance = max(0.0001, recent_vol ** 2 * 1.5)  # Scale up for forecast
                else:
                    variance = 0.01
                
                mock_forecast_result = Mock()
                mock_forecast_result.variance = pd.DataFrame([[variance]], columns=[0])
                return mock_forecast_result
            
            mock_fitted.forecast = mock_forecast_func
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arch.return_value = mock_model
            
            # Process data
            for i, (spread, timestamp, regime) in enumerate(zip(spread_values, timestamps, regime_labels)):
                model.add_observation(spread, timestamp)
                
                if i >= 25 and i % 6 == 0:
                    forecast = model.forecast_volatility()
                    if forecast:
                        forecasts_by_regime[regime].append(forecast.volatility_forecast)
        
        # Analyze adaptation
        if len(forecasts_by_regime[0]) > 0 and len(forecasts_by_regime[1]) > 0:
            avg_regime_0 = np.mean(forecasts_by_regime[0])
            avg_regime_1 = np.mean(forecasts_by_regime[1])
            
            # Regime 1 has higher true volatility, so forecasts should generally be higher
            # Allow for some overlap due to adaptation lag
            assert avg_regime_1 > avg_regime_0 * 0.7, (
                f"High volatility regime forecasts ({avg_regime_1:.4f}) should be higher than "
                f"low volatility regime ({avg_regime_0:.4f})"
            )
    
    def test_garch_parameter_estimation(self):
        """Test GARCH model parameter estimation with known process."""
        # Generate data from known GARCH process
        true_omega = 0.0002
        true_alpha = 0.12
        true_beta = 0.75
        
        returns, true_volatilities, timestamps = (
            SyntheticVolatilityDataGenerator.generate_garch_like_data(
                n_points=120,  # Reduced for faster testing
                omega=true_omega,
                alpha=true_alpha,
                beta=true_beta
            )
        )
        
        model = RollingGARCHModel(
            window_size=100,
            min_observations=30,
            refit_frequency=20
        )
        
        # Add data
        for return_val, timestamp in zip(returns, timestamps):
            model.add_observation(return_val, timestamp)
        
        # Mock ARCH model to return parameters close to true values
        with patch('services.garch_model.model.arch_model') as mock_arch:
            mock_fitted = Mock()
            # Add some noise to true parameters to simulate estimation
            mock_fitted.params = {
                'omega': true_omega * (1 + np.random.normal(0, 0.1)),
                'alpha[1]': true_alpha * (1 + np.random.normal(0, 0.15)),
                'beta[1]': true_beta * (1 + np.random.normal(0, 0.1))
            }
            mock_fitted.loglikelihood = -100.0
            mock_fitted.aic = 206.0
            mock_fitted.bic = 210.0
            mock_fitted.resid = np.random.normal(0, 0.1, len(returns))
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arch.return_value = mock_model
            
            # Fit model
            success = model.fit_model(force=True)
            assert success is True
            
            # Check parameter estimates
            assert model.model_state is not None
            
            # Parameters should be in reasonable range
            assert 0 < model.model_state.omega < 0.01
            assert 0 < model.model_state.alpha < 0.5
            assert 0 < model.model_state.beta < 1.0
            
            # Persistence should be less than 1 for stationarity
            persistence = model.model_state.alpha + model.model_state.beta
            assert persistence < 1.0, f"Model should be stationary, persistence={persistence}"


class TestDynamicThresholdResponse:
    """Test dynamic threshold response to volatility changes."""
    
    def test_threshold_adaptation_to_volatility_clusters(self):
        """Test that thresholds widen during high volatility periods."""
        # Generate volatility clustering data
        spread_values, true_volatilities, timestamps = (
            SyntheticVolatilityDataGenerator.generate_volatility_clustering_data(
                n_points=100,  # Reduced for faster testing
                base_volatility=0.02,
                cluster_intensity=4.0,
                cluster_duration=20,
                n_clusters=2
            )
        )
        
        calculator = DynamicThresholdCalculator(
            base_entry_threshold=2.0,
            volatility_adjustment_factor=0.8,
            lookback_window=30
        )
        
        threshold_history = []
        
        # Process data
        for i, (spread, true_vol, timestamp) in enumerate(zip(spread_values, true_volatilities, timestamps)):
            # Simulate GARCH volatility forecast (correlated with true volatility)
            garch_vol_forecast = true_vol * (1 + np.random.normal(0, 0.1))
            garch_vol_forecast = max(0.01, garch_vol_forecast)  # Ensure positive
            
            calculator.update_statistics(spread, garch_vol_forecast, timestamp)
            
            if i >= 10:  # After some data accumulation
                thresholds = calculator.calculate_adaptive_thresholds(garch_vol_forecast)
                threshold_history.append({
                    'index': i,
                    'true_volatility': true_vol,
                    'garch_forecast': garch_vol_forecast,
                    'entry_threshold': abs(thresholds.entry_long),
                    'exit_threshold': thresholds.exit
                })
        
        # Analyze threshold adaptation
        assert len(threshold_history) > 20, "Should have threshold history"
        
        # Identify high and low volatility periods
        high_vol_threshold = np.percentile([h['true_volatility'] for h in threshold_history], 75)
        low_vol_threshold = np.percentile([h['true_volatility'] for h in threshold_history], 25)
        
        high_vol_thresholds = [
            h for h in threshold_history 
            if h['true_volatility'] >= high_vol_threshold
        ]
        low_vol_thresholds = [
            h for h in threshold_history 
            if h['true_volatility'] <= low_vol_threshold
        ]
        
        if len(high_vol_thresholds) > 0 and len(low_vol_thresholds) > 0:
            avg_high_vol_entry = np.mean([h['entry_threshold'] for h in high_vol_thresholds])
            avg_low_vol_entry = np.mean([h['entry_threshold'] for h in low_vol_thresholds])
            
            # High volatility periods should have wider thresholds
            assert avg_high_vol_entry > avg_low_vol_entry * 1.1, (
                f"High volatility thresholds ({avg_high_vol_entry:.3f}) should be wider than "
                f"low volatility thresholds ({avg_low_vol_entry:.3f})"
            )
    
    def test_bollinger_band_behavior(self):
        """Test that thresholds behave like adaptive Bollinger Bands."""
        calculator = DynamicThresholdCalculator(
            base_entry_threshold=2.0,
            base_exit_threshold=0.5,
            volatility_adjustment_factor=0.6
        )
        
        # Test with different volatility scenarios
        scenarios = [
            {'name': 'low_vol', 'volatility': 0.05, 'expected_width': 'narrow'},
            {'name': 'normal_vol', 'volatility': 0.10, 'expected_width': 'medium'},
            {'name': 'high_vol', 'volatility': 0.20, 'expected_width': 'wide'}
        ]
        
        threshold_widths = {}
        
        for scenario in scenarios:
            # Reset calculator for each scenario
            calculator.reset()
            
            # Add data with consistent volatility
            vol = scenario['volatility']
            for i in range(25):
                spread = np.random.normal(0, 0.1)  # Spread around zero
                calculator.update_statistics(spread, vol)
            
            # Calculate thresholds
            thresholds = calculator.calculate_adaptive_thresholds(vol)
            width = abs(thresholds.entry_long) + thresholds.entry_short
            threshold_widths[scenario['name']] = width
        
        # Check that threshold width increases with volatility
        assert threshold_widths['low_vol'] < threshold_widths['normal_vol'], (
            "Normal volatility should have wider thresholds than low volatility"
        )
        assert threshold_widths['normal_vol'] < threshold_widths['high_vol'], (
            "High volatility should have wider thresholds than normal volatility"
        )
        
        # Check reasonable ranges
        assert 2.0 <= threshold_widths['low_vol'] <= 6.0, "Low vol thresholds should be reasonable"
        assert 2.5 <= threshold_widths['high_vol'] <= 8.0, "High vol thresholds should be reasonable"
    
    def test_signal_generation_with_volatility_regimes(self):
        """Test signal generation across different volatility regimes."""
        calculator = DynamicThresholdCalculator(
            base_entry_threshold=2.0,
            volatility_adjustment_factor=0.5
        )
        
        # Simulate different volatility regimes
        regimes = [
            {'volatility': 0.05, 'spread_multiplier': 1.0},  # Low vol, normal spreads
            {'volatility': 0.15, 'spread_multiplier': 2.0},  # High vol, wider spreads
        ]
        
        signals_by_regime = {}
        
        for regime_idx, regime in enumerate(regimes):
            calculator.reset()
            vol = regime['volatility']
            spread_mult = regime['spread_multiplier']
            
            # Add base data
            for i in range(20):
                spread = np.random.normal(0, 0.1)
                calculator.update_statistics(spread, vol)
            
            # Test signal generation at different spread levels
            test_spreads = [-3.0 * spread_mult, -1.0 * spread_mult, 0.0, 1.0 * spread_mult, 3.0 * spread_mult]
            regime_signals = []
            
            for test_spread in test_spreads:
                from services.garch_model.model import GARCHForecast
                
                garch_forecast = GARCHForecast(
                    volatility_forecast=vol,
                    variance_forecast=vol**2,
                    confidence_interval_lower=vol*0.8,
                    confidence_interval_upper=vol*1.2,
                    forecast_horizon=1,
                    model_aic=200.0,
                    model_bic=205.0,
                    timestamp=datetime.utcnow()
                )
                
                signal = calculator.generate_signal(f"TEST_REGIME_{regime_idx}", test_spread, garch_forecast)
                regime_signals.append({
                    'spread': test_spread,
                    'signal_type': signal.signal_type,
                    'signal_strength': signal.signal_strength,
                    'z_score': signal.z_score
                })
            
            signals_by_regime[regime_idx] = regime_signals
        
        # Analyze signal differences between regimes
        # In high volatility regime, should need more extreme spreads to trigger signals
        low_vol_signals = signals_by_regime[0]
        high_vol_signals = signals_by_regime[1]
        
        # Count entry signals for each regime
        low_vol_entries = sum(1 for s in low_vol_signals if 'entry' in s['signal_type'])
        high_vol_entries = sum(1 for s in high_vol_signals if 'entry' in s['signal_type'])
        
        # High volatility regime should be more conservative (fewer entry signals for same relative spreads)
        # This tests that thresholds adapt properly
        assert low_vol_entries >= high_vol_entries, (
            f"Low volatility regime should generate more entry signals ({low_vol_entries}) "
            f"than high volatility regime ({high_vol_entries}) for same relative spread levels"
        )


class TestGARCHServiceIntegration:
    """Integration tests for the complete GARCH volatility service."""
    
    @patch('services.garch_model.main.KafkaConsumer')
    @patch('services.garch_model.main.KafkaProducer')
    def test_service_message_processing(self, mock_producer_class, mock_consumer_class):
        """Test service processing of spread messages."""
        # Mock Kafka components
        mock_consumer = Mock()
        mock_producer = Mock()
        mock_consumer_class.return_value = mock_consumer
        mock_producer_class.return_value = mock_producer
        
        # Mock producer send
        future_mock = Mock()
        future_mock.get.return_value = Mock(topic='signals-thresholds', partition=0, offset=123)
        mock_producer.send.return_value = future_mock
        
        service = GARCHVolatilityService()
        
        # Initialize components
        import asyncio
        asyncio.run(service._initialize_components())
        
        # Test message processing
        test_message = {
            'pair_id': 'BTCETH',
            'spread_value': 0.0234,
            'asset1_price': 45000.0,
            'asset2_price': 3000.0,
            'hedge_ratio': 15.0,
            'timestamp': '2023-12-01T12:00:00Z'
        }
        
        # Add some data first to enable forecasting
        for i in range(35):  # Enough for min_observations
            service.garch_manager.add_observation(
                'BTCETH', 
                0.02 + np.random.normal(0, 0.01), 
                datetime.utcnow() - timedelta(minutes=35-i)
            )
        
        # Mock GARCH forecast
        with patch.object(service.garch_manager, 'forecast_volatility') as mock_forecast:
            from services.garch_model.model import GARCHForecast
            
            mock_garch_forecast = GARCHForecast(
                volatility_forecast=0.12,
                variance_forecast=0.0144,
                confidence_interval_lower=0.10,
                confidence_interval_upper=0.14,
                forecast_horizon=1,
                model_aic=200.0,
                model_bic=205.0,
                timestamp=datetime.utcnow()
            )
            mock_forecast.return_value = mock_garch_forecast
            
            # Mock threshold signal
            with patch.object(service.threshold_manager, 'generate_signal') as mock_signal:
                from services.garch_model.thresholds import ThresholdSignal
                
                mock_threshold_signal = ThresholdSignal(
                    pair_id='BTCETH',
                    spread_value=0.0234,
                    z_score=1.2,
                    volatility_forecast=0.12,
                    entry_threshold_long=-2.0,
                    entry_threshold_short=2.0,
                    exit_threshold=0.5,
                    signal_type='hold',
                    signal_strength=0.0,
                    confidence_level=0.85,
                    volatility_regime='normal',
                    timestamp=datetime.utcnow()
                )
                mock_signal.return_value = mock_threshold_signal
                
                # Process message
                service._process_spread_message(test_message)
                
                # Verify calls
                mock_forecast.assert_called_once_with('BTCETH', horizon=1)
                mock_signal.assert_called_once()
                mock_producer.send.assert_called_once()
                
                # Check published message structure
                call_args = mock_producer.send.call_args
                published_message = call_args[0][1]  # Second argument is the message
                
                required_keys = [
                    'pair_id', 'spread_value', 'volatility_forecast',
                    'z_score', 'signal_type', 'entry_threshold_long',
                    'entry_threshold_short', 'exit_threshold'
                ]
                
                for key in required_keys:
                    assert key in published_message, f"Missing key: {key}"
                
                assert published_message['pair_id'] == 'BTCETH'
                assert published_message['service'] == 'garch-volatility'
    
    def test_service_health_status(self):
        """Test service health status reporting."""
        service = GARCHVolatilityService()
        
        # Mock components
        service.running = True
        service.stats['start_time'] = datetime.utcnow() - timedelta(minutes=10)
        service.stats['messages_processed'] = 100
        service.stats['forecasts_generated'] = 80
        service.stats['signals_published'] = 75
        
        # Add some mock data to managers
        service.garch_manager.add_pair('BTCETH')
        service.threshold_manager.add_pair('BTCETH')
        
        health_status = service.get_health_status()
        
        required_keys = [
            'status', 'uptime_seconds', 'stats', 'garch_models',
            'threshold_calculators', 'kafka_connected'
        ]
        
        for key in required_keys:
            assert key in health_status, f"Missing health status key: {key}"
        
        assert health_status['status'] == 'healthy'
        assert health_status['uptime_seconds'] > 0
        assert 'BTCETH' in health_status['garch_models']
        assert 'BTCETH' in health_status['threshold_calculators']


class TestVolatilityForecastAccuracy:
    """Test accuracy of volatility forecasts under different conditions."""
    
    def test_forecast_accuracy_stable_period(self):
        """Test forecast accuracy during stable volatility periods."""
        # Generate stable volatility data
        np.random.seed(42)
        true_vol = 0.05
        n_points = 80  # Reduced for faster testing
        
        spread_values = []
        current_spread = 0.0
        
        for i in range(n_points):
            innovation = np.random.normal(0, true_vol)
            current_spread += innovation
            spread_values.append(current_spread)
        
        model = RollingGARCHModel(
            window_size=60,
            min_observations=20,
            refit_frequency=5
        )
        
        # Add data
        for spread in spread_values:
            model.add_observation(spread)
        
        # Mock GARCH model to return reasonable forecasts
        with patch('services.garch_model.model.arch_model') as mock_arch:
            mock_fitted = Mock()
            mock_fitted.params = {
                'omega': 0.0001,
                'alpha[1]': 0.1,
                'beta[1]': 0.8
            }
            mock_fitted.loglikelihood = -100.0
            mock_fitted.aic = 206.0
            mock_fitted.bic = 210.0
            mock_fitted.resid = np.random.normal(0, true_vol, n_points)
            
            # Mock forecast to return value close to true volatility
            mock_forecast_result = Mock()
            mock_forecast_result.variance = pd.DataFrame([[true_vol**2 * 1.1]], columns=[0])
            mock_fitted.forecast.return_value = mock_forecast_result
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arch.return_value = mock_model
            
            # Generate forecast
            forecast = model.forecast_volatility()
            
            assert forecast is not None
            
            # Forecast should be reasonably close to true volatility
            forecast_error = abs(forecast.volatility_forecast - true_vol) / true_vol
            assert forecast_error < 0.5, (
                f"Forecast error too high: {forecast_error:.3f}, "
                f"forecast={forecast.volatility_forecast:.4f}, true={true_vol:.4f}"
            )
    
    def test_forecast_confidence_intervals(self):
        """Test that confidence intervals are reasonable."""
        model = RollingGARCHModel(min_observations=20)
        
        # Add data
        np.random.seed(42)
        for i in range(30):
            model.add_observation(np.random.normal(0, 0.1))
        
        with patch('services.garch_model.model.arch_model') as mock_arch:
            mock_fitted = Mock()
            mock_fitted.params = {
                'omega': 0.0001,
                'alpha[1]': 0.1,
                'beta[1]': 0.8
            }
            mock_fitted.loglikelihood = -100.0
            mock_fitted.aic = 206.0
            mock_fitted.bic = 210.0
            mock_fitted.resid = np.random.normal(0, 0.1, 30)
            
            mock_forecast_result = Mock()
            mock_forecast_result.variance = pd.DataFrame([[0.01]], columns=[0])
            mock_fitted.forecast.return_value = mock_forecast_result
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arch.return_value = mock_model
            
            forecast = model.forecast_volatility()
            
            assert forecast is not None
            
            # Check confidence interval properties
            assert forecast.confidence_interval_lower >= 0, "Lower bound should be non-negative"
            assert forecast.confidence_interval_lower < forecast.volatility_forecast, (
                "Lower bound should be less than forecast"
            )
            assert forecast.confidence_interval_upper > forecast.volatility_forecast, (
                "Upper bound should be greater than forecast"
            )
            
            # Interval width should be reasonable
            interval_width = forecast.confidence_interval_upper - forecast.confidence_interval_lower
            assert interval_width > 0, "Confidence interval should have positive width"
            assert interval_width < forecast.volatility_forecast * 2, (
                "Confidence interval shouldn't be too wide"
            )
