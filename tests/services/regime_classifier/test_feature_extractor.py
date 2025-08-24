"""
Unit tests for feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.regime_classifier.features.feature_extractor import (
    FeatureExtractor, TechnicalIndicators, MarketMicrostructureFeatures,
    VolatilityFeatures, TrendFeatures
)
from services.regime_classifier.models.regime_models import MarketFeatures


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        # Create test price data with known pattern
        prices = np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106])
        
        rsi = TechnicalIndicators.rsi(prices, period=14)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = np.array([100, 101, 102])  # Less than required period
        
        rsi = TechnicalIndicators.rsi(prices, period=14)
        
        # Should return default value
        assert rsi == 50.0
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        # Create trending price data
        prices = np.array([100 + i * 0.5 for i in range(50)])
        
        macd_signal = TechnicalIndicators.macd(prices)
        
        assert isinstance(macd_signal, float)
        # For uptrending data, MACD signal should be positive
        assert macd_signal >= 0
    
    def test_bollinger_position_calculation(self):
        """Test Bollinger Bands position calculation."""
        # Create price data with known pattern
        prices = np.array([100] * 15 + [105] * 5 + [95] * 5)  # Stable, then high, then low
        
        # Test position at high price
        position_high = TechnicalIndicators.bollinger_position(prices[:20])
        assert 0.5 < position_high <= 1.0  # Should be in upper half
        
        # Test position at low price
        position_low = TechnicalIndicators.bollinger_position(prices)
        assert 0.0 <= position_low < 0.5  # Should be in lower half
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        # Create OHLC data
        high = np.array([101, 102, 103, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105, 106, 107])
        low = np.array([99, 100, 101, 100, 99, 98, 97, 98, 99, 100, 101, 102, 103, 104, 105])
        close = np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106])
        
        atr = TechnicalIndicators.atr(high, low, close, period=14)
        
        assert isinstance(atr, float)
        assert atr >= 0  # ATR should be non-negative


class TestMarketMicrostructureFeatures:
    """Test market microstructure feature calculations."""
    
    def test_bid_ask_spread_calculation(self):
        """Test bid-ask spread calculation."""
        bid_prices = np.array([99.5, 99.6, 99.7])
        ask_prices = np.array([100.5, 100.6, 100.7])
        
        spread = MarketMicrostructureFeatures.calculate_bid_ask_spread(bid_prices, ask_prices)
        
        assert isinstance(spread, float)
        assert spread > 0  # Spread should be positive
        assert spread < 1  # Should be reasonable percentage
    
    def test_order_book_imbalance_calculation(self):
        """Test order book imbalance calculation."""
        bid_volumes = np.array([100, 200, 150])
        ask_volumes = np.array([80, 120, 100])
        
        imbalance = MarketMicrostructureFeatures.calculate_order_book_imbalance(bid_volumes, ask_volumes)
        
        assert isinstance(imbalance, float)
        assert -1 <= imbalance <= 1  # Imbalance should be normalized
        assert imbalance > 0  # More bid volume, so positive imbalance
    
    def test_trade_intensity_calculation(self):
        """Test trade intensity calculation."""
        now = datetime.now()
        trade_times = [
            now - timedelta(minutes=30),
            now - timedelta(minutes=20),
            now - timedelta(minutes=10),
            now - timedelta(minutes=5),
            now
        ]
        
        intensity = MarketMicrostructureFeatures.calculate_trade_intensity(trade_times, window_minutes=60)
        
        assert isinstance(intensity, float)
        assert intensity >= 0  # Should be non-negative
        assert intensity == 5 / 60  # 5 trades in 60 minutes


class TestVolatilityFeatures:
    """Test volatility feature calculations."""
    
    def test_realized_volatility_calculation(self):
        """Test realized volatility calculation."""
        # Create returns with known volatility
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        
        vol = VolatilityFeatures.realized_volatility(returns, annualize=False)
        
        assert isinstance(vol, float)
        assert vol > 0  # Volatility should be positive
        assert 0.01 < vol < 0.05  # Should be reasonable range
    
    def test_volatility_clustering_calculation(self):
        """Test volatility clustering measurement."""
        # Create returns with volatility clustering
        returns = []
        for i in range(100):
            if i < 50:
                returns.append(np.random.normal(0, 0.01))  # Low vol period
            else:
                returns.append(np.random.normal(0, 0.03))  # High vol period
        
        returns = np.array(returns)
        clustering = VolatilityFeatures.volatility_clustering(returns, window=20)
        
        assert isinstance(clustering, float)
        assert -1 <= clustering <= 1  # Correlation should be normalized
    
    def test_volatility_regime_indicator(self):
        """Test volatility regime indicator."""
        # Create low volatility returns
        low_vol_returns = np.random.normal(0, 0.005, 50)
        indicator_low = VolatilityFeatures.volatility_regime_indicator(low_vol_returns)
        
        # Create high volatility returns
        high_vol_returns = np.random.normal(0, 0.05, 50)
        indicator_high = VolatilityFeatures.volatility_regime_indicator(high_vol_returns)
        
        assert 0 <= indicator_low <= 1
        assert 0 <= indicator_high <= 1
        assert indicator_high > indicator_low  # High vol should have higher indicator


class TestTrendFeatures:
    """Test trend feature calculations."""
    
    def test_trend_strength_calculation(self):
        """Test trend strength calculation."""
        # Create uptrending prices
        uptrend_prices = np.array([100 + i * 0.5 for i in range(30)])
        trend_strength_up = TrendFeatures.trend_strength(uptrend_prices, window=20)
        
        # Create downtrending prices
        downtrend_prices = np.array([100 - i * 0.5 for i in range(30)])
        trend_strength_down = TrendFeatures.trend_strength(downtrend_prices, window=20)
        
        # Create sideways prices
        sideways_prices = np.array([100 + np.random.normal(0, 0.1) for _ in range(30)])
        trend_strength_sideways = TrendFeatures.trend_strength(sideways_prices, window=20)
        
        assert isinstance(trend_strength_up, float)
        assert isinstance(trend_strength_down, float)
        assert isinstance(trend_strength_sideways, float)
        
        # Uptrend should be positive, downtrend negative
        assert trend_strength_up > 0
        assert trend_strength_down < 0
        assert abs(trend_strength_sideways) < abs(trend_strength_up)
    
    def test_trend_consistency_calculation(self):
        """Test trend consistency calculation."""
        # Create consistent uptrend
        consistent_prices = np.array([100 + i * 0.1 for i in range(30)])
        consistency_high = TrendFeatures.trend_consistency(consistent_prices, window=20)
        
        # Create random walk
        random_prices = np.array([100])
        for _ in range(29):
            random_prices = np.append(random_prices, random_prices[-1] + np.random.choice([-0.1, 0.1]))
        consistency_low = TrendFeatures.trend_consistency(random_prices, window=20)
        
        assert 0 <= consistency_high <= 1
        assert 0 <= consistency_low <= 1
        assert consistency_high > consistency_low
    
    def test_support_resistance_strength(self):
        """Test support and resistance strength calculation."""
        # Create price data with clear support/resistance
        prices = np.array([100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 102, 101, 100, 99, 98, 99, 100])
        
        support_strength, resistance_strength = TrendFeatures.support_resistance_strength(prices, window=15)
        
        assert isinstance(support_strength, float)
        assert isinstance(resistance_strength, float)
        assert support_strength >= 0
        assert resistance_strength >= 0


class TestFeatureExtractor:
    """Test main FeatureExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        self.price_data = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
            'volume': 1000 + np.random.randn(100) * 100
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        for i in range(len(self.price_data)):
            row = self.price_data.iloc[i]
            min_price = min(row['open'], row['close'])
            max_price = max(row['open'], row['close'])
            self.price_data.iloc[i, self.price_data.columns.get_loc('low')] = min(row['low'], min_price)
            self.price_data.iloc[i, self.price_data.columns.get_loc('high')] = max(row['high'], max_price)
    
    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_features(self.price_data)
        
        assert isinstance(features, MarketFeatures)
        assert features.timestamp is not None
        
        # Check that some features are populated
        assert isinstance(features.returns_1h, float)
        assert isinstance(features.volatility_1h, float)
        assert isinstance(features.volume_ratio_1h, float)
        assert isinstance(features.rsi_14, float)
    
    def test_extract_features_with_external_data(self):
        """Test feature extraction with external data."""
        external_data = {
            'funding_rate': 0.0001,
            'open_interest_change': 0.05,
            'fear_greed_index': 45.0,
            'social_sentiment': -0.2
        }
        
        features = self.extractor.extract_features(
            self.price_data,
            external_data=external_data
        )
        
        assert features.funding_rate == 0.0001
        assert features.open_interest_change == 0.05
        assert features.fear_greed_index == 45.0
        assert features.social_sentiment == -0.2
    
    def test_extract_features_with_orderbook_data(self):
        """Test feature extraction with order book data."""
        orderbook_data = {
            'bid_prices': [99.5, 99.4, 99.3],
            'ask_prices': [100.5, 100.6, 100.7],
            'bid_volumes': [100, 200, 150],
            'ask_volumes': [80, 120, 100],
            'trade_times': [datetime.now() - timedelta(minutes=i) for i in range(10)]
        }
        
        features = self.extractor.extract_features(
            self.price_data,
            orderbook_data=orderbook_data
        )
        
        assert features.bid_ask_spread > 0
        assert -1 <= features.order_book_imbalance <= 1
        assert features.trade_intensity >= 0
    
    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        # Create training data
        feature_arrays = []
        for _ in range(10):
            features = self.extractor.extract_features(self.price_data)
            feature_arrays.append(features.to_array())
        
        # Fit scaler
        self.extractor.fit_scaler(feature_arrays)
        assert self.extractor.is_fitted
        
        # Test scaling
        test_features = self.extractor.extract_features(self.price_data)
        scaled_features = self.extractor.scale_features(test_features.to_array())
        
        assert isinstance(scaled_features, np.ndarray)
        assert len(scaled_features) == len(test_features.to_array())
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        feature_names = self.extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'returns_1h' in feature_names
        assert 'volatility_1h' in feature_names
        assert 'rsi_14' in feature_names
    
    def test_extract_features_empty_data(self):
        """Test feature extraction with empty data."""
        empty_data = pd.DataFrame()
        
        features = self.extractor.extract_features(empty_data)
        
        # Should return default MarketFeatures
        assert isinstance(features, MarketFeatures)
        assert features.returns_1h == 0.0
        assert features.volatility_1h == 0.0
    
    def test_extract_features_insufficient_data(self):
        """Test feature extraction with insufficient data."""
        # Create very small dataset
        small_data = self.price_data.head(5)
        
        features = self.extractor.extract_features(small_data)
        
        # Should still return MarketFeatures, but some values might be defaults
        assert isinstance(features, MarketFeatures)
        assert features.timestamp is not None
    
    @patch('services.regime_classifier.features.feature_extractor.talib')
    def test_extract_features_talib_fallback(self, mock_talib):
        """Test feature extraction when talib functions fail."""
        # Mock talib to raise exceptions
        mock_talib.RSI.side_effect = Exception("talib error")
        mock_talib.MACD.side_effect = Exception("talib error")
        mock_talib.BBANDS.side_effect = Exception("talib error")
        
        features = self.extractor.extract_features(self.price_data)
        
        # Should still work with fallback calculations
        assert isinstance(features, MarketFeatures)
        assert isinstance(features.rsi_14, float)
        assert 0 <= features.rsi_14 <= 100
    
    def test_feature_importance_mock(self):
        """Test feature importance extraction with mock model."""
        # Create mock model with feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = np.random.rand(23)  # 23 features
        
        importance = self.extractor.get_feature_importance(mock_model)
        
        assert isinstance(importance, dict)
        assert len(importance) == 23
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_feature_importance_no_attributes(self):
        """Test feature importance with model that has no importance attributes."""
        mock_model = Mock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        importance = self.extractor.get_feature_importance(mock_model)
        
        assert importance == {}


if __name__ == "__main__":
    pytest.main([__file__])
