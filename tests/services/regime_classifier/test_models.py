"""
Unit tests for regime classification models.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from services.regime_classifier.models.regime_models import (
    MarketRegime, RegimeClassification, RegimeConfidence, RegimeTransition,
    MarketFeatures, StrategyParameters, RegimeAlert, RegimeModelConfig,
    DEFAULT_STRATEGY_PARAMETERS, get_confidence_level, is_high_volatility_regime,
    is_trending_regime, get_regime_risk_level
)


class TestMarketFeatures:
    """Test MarketFeatures data model."""
    
    def test_market_features_creation(self):
        """Test creating MarketFeatures instance."""
        timestamp = datetime.now()
        features = MarketFeatures(
            timestamp=timestamp,
            returns_1h=0.01,
            volatility_1h=0.02,
            volume_ratio_1h=1.5,
            rsi_14=65.0
        )
        
        assert features.timestamp == timestamp
        assert features.returns_1h == 0.01
        assert features.volatility_1h == 0.02
        assert features.volume_ratio_1h == 1.5
        assert features.rsi_14 == 65.0
    
    def test_to_array_conversion(self):
        """Test converting features to numpy array."""
        features = MarketFeatures(
            timestamp=datetime.now(),
            returns_1h=0.01,
            returns_4h=0.02,
            returns_24h=0.03,
            returns_7d=0.04,
            volatility_1h=0.1,
            volatility_4h=0.2,
            volatility_24h=0.3,
            volatility_7d=0.4
        )
        
        array = features.to_array()
        assert isinstance(array, np.ndarray)
        assert len(array) == 23  # Total number of features
        assert array[0] == 0.01  # returns_1h
        assert array[4] == 0.1   # volatility_1h
    
    def test_to_dict_conversion(self):
        """Test converting features to dictionary."""
        timestamp = datetime.now()
        features = MarketFeatures(
            timestamp=timestamp,
            returns_1h=0.01,
            rsi_14=65.0
        )
        
        feature_dict = features.to_dict()
        assert isinstance(feature_dict, dict)
        assert feature_dict['timestamp'] == timestamp.isoformat()
        assert feature_dict['returns_1h'] == 0.01
        assert feature_dict['rsi_14'] == 65.0


class TestRegimeClassification:
    """Test RegimeClassification data model."""
    
    def test_regime_classification_creation(self):
        """Test creating RegimeClassification instance."""
        timestamp = datetime.now()
        classification = RegimeClassification(
            timestamp=timestamp,
            regime=MarketRegime.LOW_VOL_BULL,
            confidence=0.85,
            confidence_level=RegimeConfidence.HIGH
        )
        
        assert classification.timestamp == timestamp
        assert classification.regime == MarketRegime.LOW_VOL_BULL
        assert classification.confidence == 0.85
        assert classification.confidence_level == RegimeConfidence.HIGH
    
    def test_to_dict_conversion(self):
        """Test converting classification to dictionary."""
        timestamp = datetime.now()
        features = MarketFeatures(timestamp=timestamp, returns_1h=0.01)
        
        classification = RegimeClassification(
            timestamp=timestamp,
            regime=MarketRegime.HIGH_VOL_BEAR,
            confidence=0.75,
            confidence_level=RegimeConfidence.HIGH,
            features=features,
            previous_regime=MarketRegime.LOW_VOL_RANGE,
            regime_duration=timedelta(hours=2)
        )
        
        result_dict = classification.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['regime'] == 'high_vol_bear'
        assert result_dict['confidence'] == 0.75
        assert result_dict['previous_regime'] == 'low_vol_range'
        assert result_dict['regime_duration'] == 7200.0  # 2 hours in seconds
        assert 'features' in result_dict


class TestRegimeTransition:
    """Test RegimeTransition data model."""
    
    def test_regime_transition_creation(self):
        """Test creating RegimeTransition instance."""
        timestamp = datetime.now()
        duration = timedelta(hours=3)
        
        transition = RegimeTransition(
            timestamp=timestamp,
            from_regime=MarketRegime.STABLE_RANGE,
            to_regime=MarketRegime.HIGH_VOL_BULL,
            transition_probability=0.8,
            duration_in_previous=duration,
            is_gradual=False,
            price_change_during_transition=0.05
        )
        
        assert transition.timestamp == timestamp
        assert transition.from_regime == MarketRegime.STABLE_RANGE
        assert transition.to_regime == MarketRegime.HIGH_VOL_BULL
        assert transition.transition_probability == 0.8
        assert transition.duration_in_previous == duration
        assert transition.is_gradual is False
        assert transition.price_change_during_transition == 0.05
    
    def test_to_dict_conversion(self):
        """Test converting transition to dictionary."""
        timestamp = datetime.now()
        duration = timedelta(minutes=90)
        
        transition = RegimeTransition(
            timestamp=timestamp,
            from_regime=MarketRegime.LOW_VOL_BULL,
            to_regime=MarketRegime.CRISIS,
            transition_probability=0.9,
            duration_in_previous=duration
        )
        
        result_dict = transition.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['from_regime'] == 'low_vol_bull'
        assert result_dict['to_regime'] == 'crisis'
        assert result_dict['transition_probability'] == 0.9
        assert result_dict['duration_in_previous'] == 5400.0  # 90 minutes in seconds


class TestStrategyParameters:
    """Test StrategyParameters data model."""
    
    def test_strategy_parameters_creation(self):
        """Test creating StrategyParameters instance."""
        params = StrategyParameters(
            regime=MarketRegime.HIGH_VOL_RANGE,
            entry_z_score=2.5,
            exit_z_score=0.8,
            max_position_size=0.6,
            kelly_fraction=0.15
        )
        
        assert params.regime == MarketRegime.HIGH_VOL_RANGE
        assert params.entry_z_score == 2.5
        assert params.exit_z_score == 0.8
        assert params.max_position_size == 0.6
        assert params.kelly_fraction == 0.15
    
    def test_default_parameters_exist(self):
        """Test that default parameters exist for all regimes."""
        for regime in MarketRegime:
            assert regime in DEFAULT_STRATEGY_PARAMETERS
            params = DEFAULT_STRATEGY_PARAMETERS[regime]
            assert isinstance(params, StrategyParameters)
            assert params.regime == regime
    
    def test_to_dict_conversion(self):
        """Test converting parameters to dictionary."""
        params = StrategyParameters(
            regime=MarketRegime.TRENDING_UP,
            entry_z_score=1.8,
            max_position_size=1.0
        )
        
        result_dict = params.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['regime'] == 'trending_up'
        assert result_dict['entry_z_score'] == 1.8
        assert result_dict['max_position_size'] == 1.0


class TestRegimeAlert:
    """Test RegimeAlert data model."""
    
    def test_regime_alert_creation(self):
        """Test creating RegimeAlert instance."""
        timestamp = datetime.now()
        alert = RegimeAlert(
            timestamp=timestamp,
            alert_type="regime_change",
            regime=MarketRegime.CRISIS,
            previous_regime=MarketRegime.HIGH_VOL_BEAR,
            confidence=0.95,
            message="Crisis regime detected",
            severity="critical",
            suggested_actions=["Reduce positions", "Activate emergency protocols"],
            risk_level="critical"
        )
        
        assert alert.timestamp == timestamp
        assert alert.alert_type == "regime_change"
        assert alert.regime == MarketRegime.CRISIS
        assert alert.previous_regime == MarketRegime.HIGH_VOL_BEAR
        assert alert.confidence == 0.95
        assert alert.severity == "critical"
        assert len(alert.suggested_actions) == 2
    
    def test_to_dict_conversion(self):
        """Test converting alert to dictionary."""
        timestamp = datetime.now()
        alert = RegimeAlert(
            timestamp=timestamp,
            alert_type="high_volatility",
            regime=MarketRegime.HIGH_VOL_BULL,
            confidence=0.8,
            message="High volatility detected",
            severity="medium"
        )
        
        result_dict = alert.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['alert_type'] == "high_volatility"
        assert result_dict['regime'] == 'high_vol_bull'
        assert result_dict['confidence'] == 0.8
        assert result_dict['severity'] == "medium"


class TestRegimeModelConfig:
    """Test RegimeModelConfig data model."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = RegimeModelConfig()
        
        assert config.n_components == 6
        assert config.covariance_type == "full"
        assert config.n_iter == 100
        assert config.tol == 1e-2
        assert config.min_confidence_threshold == 0.3
    
    def test_custom_config_creation(self):
        """Test creating custom configuration."""
        config = RegimeModelConfig(
            n_components=8,
            covariance_type="diag",
            training_window=3000,
            retrain_frequency=72
        )
        
        assert config.n_components == 8
        assert config.covariance_type == "diag"
        assert config.training_window == 3000
        assert config.retrain_frequency == 72


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_confidence_level(self):
        """Test confidence level mapping."""
        assert get_confidence_level(0.1) == RegimeConfidence.VERY_LOW
        assert get_confidence_level(0.4) == RegimeConfidence.LOW
        assert get_confidence_level(0.6) == RegimeConfidence.MEDIUM
        assert get_confidence_level(0.8) == RegimeConfidence.HIGH
        assert get_confidence_level(0.95) == RegimeConfidence.VERY_HIGH
    
    def test_is_high_volatility_regime(self):
        """Test high volatility regime detection."""
        assert is_high_volatility_regime(MarketRegime.HIGH_VOL_BULL) is True
        assert is_high_volatility_regime(MarketRegime.HIGH_VOL_BEAR) is True
        assert is_high_volatility_regime(MarketRegime.HIGH_VOL_RANGE) is True
        assert is_high_volatility_regime(MarketRegime.CRISIS) is True
        
        assert is_high_volatility_regime(MarketRegime.LOW_VOL_BULL) is False
        assert is_high_volatility_regime(MarketRegime.STABLE_RANGE) is False
    
    def test_is_trending_regime(self):
        """Test trending regime detection."""
        assert is_trending_regime(MarketRegime.LOW_VOL_BULL) is True
        assert is_trending_regime(MarketRegime.HIGH_VOL_BEAR) is True
        assert is_trending_regime(MarketRegime.TRENDING_UP) is True
        assert is_trending_regime(MarketRegime.TRENDING_DOWN) is True
        
        assert is_trending_regime(MarketRegime.STABLE_RANGE) is False
        assert is_trending_regime(MarketRegime.LOW_VOL_RANGE) is False
    
    def test_get_regime_risk_level(self):
        """Test regime risk level mapping."""
        assert get_regime_risk_level(MarketRegime.STABLE_RANGE) == "very_low"
        assert get_regime_risk_level(MarketRegime.LOW_VOL_BULL) == "low"
        assert get_regime_risk_level(MarketRegime.HIGH_VOL_BULL) == "medium"
        assert get_regime_risk_level(MarketRegime.RECOVERY) == "high"
        assert get_regime_risk_level(MarketRegime.CRISIS) == "very_high"


class TestMarketRegimeEnum:
    """Test MarketRegime enum."""
    
    def test_all_regimes_defined(self):
        """Test that all expected regimes are defined."""
        expected_regimes = [
            'LOW_VOL_BULL', 'LOW_VOL_BEAR', 'LOW_VOL_RANGE',
            'HIGH_VOL_BULL', 'HIGH_VOL_BEAR', 'HIGH_VOL_RANGE',
            'STABLE_RANGE', 'TRENDING_UP', 'TRENDING_DOWN',
            'CRISIS', 'RECOVERY', 'UNKNOWN'
        ]
        
        actual_regimes = [regime.name for regime in MarketRegime]
        
        for expected in expected_regimes:
            assert expected in actual_regimes
    
    def test_regime_values(self):
        """Test regime string values."""
        assert MarketRegime.LOW_VOL_BULL.value == "low_vol_bull"
        assert MarketRegime.HIGH_VOL_BEAR.value == "high_vol_bear"
        assert MarketRegime.CRISIS.value == "crisis"
        assert MarketRegime.UNKNOWN.value == "unknown"


if __name__ == "__main__":
    pytest.main([__file__])
