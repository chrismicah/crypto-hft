"""
Unit tests for OnChain Signal Generator.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from services.onchain_data.signals.signal_generator import (
    OnChainSignalGenerator, ModelConfig
)
from services.onchain_data.models import (
    OnChainFeatureSet, OnChainSignal, SignalStrength, OnChainAlert
)


class TestOnChainSignalGenerator:
    """Test suite for OnChainSignalGenerator."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return ModelConfig(
            model_type="random_forest",
            n_estimators=10,  # Small for testing
            min_training_samples=5,
            test_size=0.2
        )
    
    @pytest.fixture
    def signal_generator(self, config):
        """Signal generator instance."""
        return OnChainSignalGenerator(config)
    
    @pytest.fixture
    def sample_feature_sets(self):
        """Generate sample feature sets for testing."""
        feature_sets = []
        base_time = datetime.utcnow()
        
        for i in range(20):  # Create 20 samples
            timestamp = base_time + timedelta(hours=i)
            
            # Create varying features to simulate real data
            bullish_score = 0.5 + 0.3 * np.sin(i * 0.1)
            bearish_score = 0.5 - 0.3 * np.sin(i * 0.1)
            momentum_score = 0.1 * np.cos(i * 0.2)
            
            feature_set = OnChainFeatureSet(
                symbol="BTC",
                timestamp=timestamp,
                exchange_flow_ma_7d=1000 + 100 * np.sin(i * 0.1),
                exchange_flow_ma_30d=1000 + 50 * np.cos(i * 0.05),
                exchange_flow_momentum=momentum_score,
                whale_activity_ma_7d=50 + 10 * np.sin(i * 0.15),
                whale_activity_volatility=5 + 2 * np.abs(np.sin(i * 0.2)),
                onchain_bullish_score=bullish_score,
                onchain_bearish_score=bearish_score,
                onchain_momentum_score=momentum_score
            )
            
            # Add some raw metrics
            feature_set.raw_metrics = {
                f"feature_{j}": np.random.normal(0, 1) for j in range(10)
            }
            
            feature_sets.append(feature_set)
        
        return feature_sets
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data."""
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=10),
            end=datetime.utcnow() + timedelta(days=10),
            freq='H'
        )
        
        # Generate realistic price movement
        returns = np.random.normal(0, 0.01, len(dates))
        prices = 50000 * np.exp(np.cumsum(returns))  # Starting at $50k
        
        price_data = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        return price_data
    
    def test_model_creation(self, signal_generator):
        """Test model creation with different types."""
        # Test XGBoost
        model = signal_generator._create_model("xgboost", classification=True)
        assert model is not None
        
        # Test Random Forest
        model = signal_generator._create_model("random_forest", classification=False)
        assert model is not None
        
        # Test invalid model type
        with pytest.raises(ValueError):
            signal_generator._create_model("invalid_model", classification=True)
    
    def test_prepare_training_data(self, signal_generator, sample_feature_sets, sample_price_data):
        """Test training data preparation."""
        # Test with price data
        features, labels, feature_names = signal_generator.prepare_training_data(
            sample_feature_sets, sample_price_data
        )
        
        assert features.shape[0] == len(sample_feature_sets)
        assert features.shape[1] > 0
        assert len(labels) == features.shape[0]
        assert len(feature_names) == features.shape[1]
        
        # Check label types
        unique_labels = set(labels)
        valid_labels = {s.value for s in SignalStrength}
        assert unique_labels.issubset(valid_labels)
        
        # Test without price data (using composite scores)
        features_no_price, labels_no_price, _ = signal_generator.prepare_training_data(
            sample_feature_sets
        )
        
        assert features_no_price.shape[0] == len(sample_feature_sets)
        assert len(labels_no_price) == features_no_price.shape[0]
    
    def test_train_models(self, signal_generator, sample_feature_sets, sample_price_data):
        """Test model training."""
        # Train models
        metrics = signal_generator.train_models(
            sample_feature_sets, sample_price_data, validate=True
        )
        
        assert signal_generator.is_fitted
        assert signal_generator.classification_model is not None
        assert signal_generator.regression_model is not None
        assert len(signal_generator.feature_columns) > 0
        
        # Check metrics
        assert 'classification_accuracy' in metrics
        assert 'regression_mse' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
        
        # Check feature importance
        assert len(signal_generator.feature_importance) > 0
    
    def test_insufficient_training_data(self, signal_generator):
        """Test handling of insufficient training data."""
        # Create very few samples
        feature_sets = [
            OnChainFeatureSet(
                symbol="BTC",
                timestamp=datetime.utcnow(),
                onchain_bullish_score=0.5,
                onchain_bearish_score=0.3,
                onchain_momentum_score=0.1
            )
            for _ in range(2)  # Less than min_training_samples
        ]
        
        with pytest.raises(ValueError, match="Insufficient training samples"):
            signal_generator.train_models(feature_sets)
    
    def test_generate_signal(self, signal_generator, sample_feature_sets, sample_price_data):
        """Test signal generation."""
        # Train first
        signal_generator.train_models(sample_feature_sets, sample_price_data, validate=False)
        
        # Generate signal
        feature_set = sample_feature_sets[0]
        signal = signal_generator.generate_signal(feature_set, "BTC")
        
        assert signal is not None
        assert isinstance(signal, OnChainSignal)
        assert signal.symbol == "BTC"
        assert signal.timestamp == feature_set.timestamp
        assert isinstance(signal.strength, SignalStrength)
        assert 0 <= signal.confidence <= 1
        assert -1 <= signal.score <= 1
        assert signal.signal_type == "onchain_ml"
        assert len(signal.features_used) > 0
    
    def test_generate_signal_untrained(self, signal_generator, sample_feature_sets):
        """Test signal generation with untrained models."""
        feature_set = sample_feature_sets[0]
        signal = signal_generator.generate_signal(feature_set, "BTC")
        
        assert signal is None
    
    def test_generate_signal_low_confidence(self, signal_generator, sample_feature_sets):
        """Test signal generation with low confidence."""
        # Mock the models to return low confidence
        signal_generator.is_fitted = True
        signal_generator.feature_columns = ["feature_0", "feature_1"]
        
        mock_classifier = Mock()
        mock_classifier.predict.return_value = [0]
        mock_classifier.predict_proba.return_value = [[0.6, 0.3, 0.1]]  # Low max probability
        
        mock_regressor = Mock()
        mock_regressor.predict.return_value = [0.5]
        
        signal_generator.classification_model = mock_classifier
        signal_generator.regression_model = mock_regressor
        signal_generator.label_encoder = Mock()
        signal_generator.label_encoder.inverse_transform.return_value = [SignalStrength.NEUTRAL.value]
        
        # Set high confidence threshold
        signal_generator.config.confidence_threshold = 0.8
        
        feature_set = sample_feature_sets[0]
        signal = signal_generator.generate_signal(feature_set, "BTC")
        
        assert signal is None  # Should be None due to low confidence
    
    def test_generate_batch_signals(self, signal_generator, sample_feature_sets, sample_price_data):
        """Test batch signal generation."""
        # Train first
        signal_generator.train_models(sample_feature_sets, sample_price_data, validate=False)
        
        # Generate batch signals
        signals = signal_generator.generate_batch_signals(sample_feature_sets[:5], "BTC")
        
        assert isinstance(signals, list)
        assert len(signals) <= 5  # Some might be filtered out due to confidence
        
        for signal in signals:
            assert isinstance(signal, OnChainSignal)
            assert signal.symbol == "BTC"
    
    def test_generate_alerts(self, signal_generator):
        """Test alert generation from signals."""
        # Create mock signals
        signals = [
            OnChainSignal(
                signal_id="test_1",
                symbol="BTC",
                timestamp=datetime.utcnow(),
                signal_type="onchain_ml",
                strength=SignalStrength.VERY_BULLISH,
                confidence=0.95,
                score=0.85  # Above default threshold
            ),
            OnChainSignal(
                signal_id="test_2",
                symbol="BTC",
                timestamp=datetime.utcnow(),
                signal_type="onchain_ml",
                strength=SignalStrength.NEUTRAL,
                confidence=0.6,
                score=0.1  # Below threshold
            )
        ]
        
        alerts = signal_generator.generate_alerts(signals)
        
        assert isinstance(alerts, list)
        assert len(alerts) == 1  # Only first signal should generate alert
        
        alert = alerts[0]
        assert isinstance(alert, OnChainAlert)
        assert alert.symbol == "BTC"
        assert alert.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert "Strong On-Chain Signal" in alert.title
    
    def test_save_and_load_models(self, signal_generator, sample_feature_sets, sample_price_data):
        """Test model saving and loading."""
        # Train models
        signal_generator.train_models(sample_feature_sets, sample_price_data, validate=False)
        
        # Save models
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            filepath = tmp_file.name
        
        try:
            signal_generator.save_models(filepath)
            assert os.path.exists(filepath)
            
            # Create new generator and load models
            new_generator = OnChainSignalGenerator(signal_generator.config)
            new_generator.load_models(filepath)
            
            assert new_generator.is_fitted
            assert new_generator.classification_model is not None
            assert new_generator.regression_model is not None
            assert new_generator.feature_columns == signal_generator.feature_columns
            
            # Test that loaded model can generate signals
            feature_set = sample_feature_sets[0]
            signal = new_generator.generate_signal(feature_set, "BTC")
            assert signal is not None
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_get_model_summary(self, signal_generator, sample_feature_sets, sample_price_data):
        """Test model summary generation."""
        # Test untrained model
        summary = signal_generator.get_model_summary()
        assert summary["status"] == "not_trained"
        
        # Train and test again
        signal_generator.train_models(sample_feature_sets, sample_price_data, validate=False)
        
        summary = signal_generator.get_model_summary()
        assert summary["status"] == "trained"
        assert summary["model_type"] == signal_generator.config.model_type
        assert summary["num_features"] > 0
        assert "feature_columns" in summary
        assert "feature_importance" in summary
        assert "model_metrics" in summary
    
    def test_contributing_metrics_identification(self, signal_generator):
        """Test identification of contributing metrics."""
        # Create feature set with specific patterns
        feature_set = OnChainFeatureSet(
            symbol="BTC",
            timestamp=datetime.utcnow(),
            exchange_flow_momentum=0.15,  # Above threshold
            whale_activity_ma_7d=100,     # Positive
            onchain_bullish_score=0.8,   # High bullish
            onchain_bearish_score=0.1    # Low bearish
        )
        
        metrics = signal_generator._get_contributing_metrics(feature_set)
        
        assert isinstance(metrics, list)
        assert len(metrics) <= 5  # Limited to top 5
        
        # Should include relevant metric types based on feature values
        from services.onchain_data.models import OnChainMetricType
        assert OnChainMetricType.EXCHANGE_NET_FLOW in metrics
        assert OnChainMetricType.WHALE_TRANSACTIONS in metrics
    
    def test_recommended_actions(self, signal_generator):
        """Test recommended action generation."""
        # Test bullish signal
        bullish_signal = OnChainSignal(
            signal_id="test_bullish",
            symbol="BTC",
            timestamp=datetime.utcnow(),
            signal_type="onchain_ml",
            strength=SignalStrength.VERY_BULLISH,
            confidence=0.9,
            score=0.85
        )
        
        action = signal_generator._get_recommended_action(bullish_signal)
        assert "BUY" in action.upper()
        
        # Test bearish signal
        bearish_signal = OnChainSignal(
            signal_id="test_bearish",
            symbol="BTC",
            timestamp=datetime.utcnow(),
            signal_type="onchain_ml",
            strength=SignalStrength.BEARISH,
            confidence=0.8,
            score=-0.6
        )
        
        action = signal_generator._get_recommended_action(bearish_signal)
        assert "SELL" in action.upper()
        
        # Test neutral signal
        neutral_signal = OnChainSignal(
            signal_id="test_neutral",
            symbol="BTC",
            timestamp=datetime.utcnow(),
            signal_type="onchain_ml",
            strength=SignalStrength.NEUTRAL,
            confidence=0.7,
            score=0.05
        )
        
        action = signal_generator._get_recommended_action(neutral_signal)
        assert "HOLD" in action.upper()


if __name__ == "__main__":
    pytest.main([__file__])
