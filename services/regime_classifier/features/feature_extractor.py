"""
Time series feature extraction for market regime classification.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
import warnings
warnings.filterwarnings('ignore')

from ..models.regime_models import MarketFeatures
from ...common.logger import get_logger


class TechnicalIndicators:
    """Calculate technical indicators for feature extraction."""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0
        
        try:
            rsi_values = talib.RSI(prices.astype(float), timeperiod=period)
            return float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else 50.0
        except:
            # Fallback calculation
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Calculate MACD signal."""
        if len(prices) < slow + signal:
            return 0.0
        
        try:
            macd_line, macd_signal, _ = talib.MACD(prices.astype(float), 
                                                  fastperiod=fast, 
                                                  slowperiod=slow, 
                                                  signalperiod=signal)
            return float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0
        except:
            # Fallback calculation
            ema_fast = pd.Series(prices).ewm(span=fast).mean()
            ema_slow = pd.Series(prices).ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            return float(macd_signal.iloc[-1])
    
    @staticmethod
    def bollinger_position(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> float:
        """Calculate position within Bollinger Bands (0 = lower band, 1 = upper band)."""
        if len(prices) < period:
            return 0.5
        
        try:
            upper, middle, lower = talib.BBANDS(prices.astype(float), 
                                               timeperiod=period, 
                                               nbdevup=std_dev, 
                                               nbdevdn=std_dev)
            
            current_price = prices[-1]
            upper_val = upper[-1]
            lower_val = lower[-1]
            
            if np.isnan(upper_val) or np.isnan(lower_val) or upper_val == lower_val:
                return 0.5
            
            position = (current_price - lower_val) / (upper_val - lower_val)
            return max(0.0, min(1.0, position))
        except:
            # Fallback calculation
            window = prices[-period:]
            mean_price = np.mean(window)
            std_price = np.std(window)
            
            upper_band = mean_price + std_dev * std_price
            lower_band = mean_price - std_dev * std_price
            
            if upper_band == lower_band:
                return 0.5
            
            position = (prices[-1] - lower_band) / (upper_band - lower_band)
            return max(0.0, min(1.0, position))
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return 0.0
        
        try:
            atr_values = talib.ATR(high.astype(float), low.astype(float), close.astype(float), timeperiod=period)
            return float(atr_values[-1]) if not np.isnan(atr_values[-1]) else 0.0
        except:
            # Fallback calculation
            tr_list = []
            for i in range(1, len(close)):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr_list.append(tr)
            
            if len(tr_list) >= period:
                return np.mean(tr_list[-period:])
            return 0.0


class MarketMicrostructureFeatures:
    """Extract market microstructure features."""
    
    @staticmethod
    def calculate_bid_ask_spread(bid_prices: np.ndarray, ask_prices: np.ndarray) -> float:
        """Calculate average bid-ask spread."""
        if len(bid_prices) == 0 or len(ask_prices) == 0:
            return 0.0
        
        spreads = (ask_prices - bid_prices) / ((ask_prices + bid_prices) / 2)
        return float(np.mean(spreads[spreads > 0]))
    
    @staticmethod
    def calculate_order_book_imbalance(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
        """Calculate order book imbalance."""
        if len(bid_volumes) == 0 or len(ask_volumes) == 0:
            return 0.0
        
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        
        if total_bid + total_ask == 0:
            return 0.0
        
        return (total_bid - total_ask) / (total_bid + total_ask)
    
    @staticmethod
    def calculate_trade_intensity(trade_times: List[datetime], window_minutes: int = 60) -> float:
        """Calculate trade intensity (trades per minute)."""
        if len(trade_times) < 2:
            return 0.0
        
        # Count trades in the last window
        cutoff_time = trade_times[-1] - timedelta(minutes=window_minutes)
        recent_trades = [t for t in trade_times if t >= cutoff_time]
        
        return len(recent_trades) / window_minutes


class VolatilityFeatures:
    """Extract volatility-based features."""
    
    @staticmethod
    def realized_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate realized volatility."""
        if len(returns) < 2:
            return 0.0
        
        vol = np.std(returns[~np.isnan(returns)])
        
        if annualize:
            # Annualize assuming hourly data
            vol *= np.sqrt(24 * 365)
        
        return float(vol)
    
    @staticmethod
    def garch_volatility(returns: np.ndarray) -> float:
        """Estimate GARCH volatility (simplified)."""
        if len(returns) < 10:
            return TechnicalIndicators.realized_volatility(returns)
        
        try:
            from arch import arch_model
            
            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = fitted_model.conditional_volatility
            return float(conditional_vol.iloc[-1] / 100)
        except:
            # Fallback to realized volatility
            return VolatilityFeatures.realized_volatility(returns)
    
    @staticmethod
    def volatility_clustering(returns: np.ndarray, window: int = 20) -> float:
        """Measure volatility clustering."""
        if len(returns) < window * 2:
            return 0.0
        
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(window).std()
        
        # Measure autocorrelation in volatility
        vol_autocorr = rolling_vol.autocorr(lag=1)
        
        return float(vol_autocorr) if not np.isnan(vol_autocorr) else 0.0
    
    @staticmethod
    def volatility_regime_indicator(returns: np.ndarray, threshold: float = 0.02) -> float:
        """Indicator of high/low volatility regime."""
        if len(returns) < 10:
            return 0.5
        
        current_vol = np.std(returns[-10:])  # Recent volatility
        historical_vol = np.std(returns)     # Historical volatility
        
        if historical_vol == 0:
            return 0.5
        
        vol_ratio = current_vol / historical_vol
        
        # Return value between 0 (low vol) and 1 (high vol)
        return min(1.0, max(0.0, (vol_ratio - 0.5) * 2))


class TrendFeatures:
    """Extract trend-based features."""
    
    @staticmethod
    def trend_strength(prices: np.ndarray, window: int = 20) -> float:
        """Calculate trend strength using linear regression."""
        if len(prices) < window:
            return 0.0
        
        recent_prices = prices[-window:]
        x = np.arange(len(recent_prices))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
        
        # Weight by R-squared to account for trend quality
        trend_strength = normalized_slope * (r_value ** 2)
        
        return float(trend_strength)
    
    @staticmethod
    def trend_consistency(prices: np.ndarray, window: int = 20) -> float:
        """Measure trend consistency."""
        if len(prices) < window:
            return 0.0
        
        returns = np.diff(prices[-window:])
        
        if len(returns) == 0:
            return 0.0
        
        # Count consistent direction moves
        positive_moves = np.sum(returns > 0)
        negative_moves = np.sum(returns < 0)
        total_moves = len(returns)
        
        # Consistency is the proportion of moves in the dominant direction
        consistency = max(positive_moves, negative_moves) / total_moves
        
        return float(consistency)
    
    @staticmethod
    def support_resistance_strength(prices: np.ndarray, window: int = 50) -> Tuple[float, float]:
        """Calculate support and resistance strength."""
        if len(prices) < window:
            return 0.0, 0.0
        
        recent_prices = prices[-window:]
        
        # Find local minima (support) and maxima (resistance)
        try:
            min_peaks, _ = find_peaks(-recent_prices, distance=5)
            max_peaks, _ = find_peaks(recent_prices, distance=5)
            
            support_levels = recent_prices[min_peaks] if len(min_peaks) > 0 else []
            resistance_levels = recent_prices[max_peaks] if len(max_peaks) > 0 else []
            
            current_price = prices[-1]
            
            # Calculate distance to nearest support/resistance
            support_strength = 0.0
            resistance_strength = 0.0
            
            if len(support_levels) > 0:
                nearest_support = max(support_levels[support_levels <= current_price]) if any(support_levels <= current_price) else min(support_levels)
                support_strength = (current_price - nearest_support) / current_price if current_price > 0 else 0.0
            
            if len(resistance_levels) > 0:
                nearest_resistance = min(resistance_levels[resistance_levels >= current_price]) if any(resistance_levels >= current_price) else max(resistance_levels)
                resistance_strength = (nearest_resistance - current_price) / current_price if current_price > 0 else 0.0
            
            return float(support_strength), float(resistance_strength)
        except:
            return 0.0, 0.0


class FeatureExtractor:
    """Main feature extraction class for market regime classification."""
    
    def __init__(self, 
                 lookback_periods: List[int] = [1, 4, 24, 168],
                 use_technical_indicators: bool = True,
                 use_microstructure: bool = True,
                 use_volatility_features: bool = True,
                 use_trend_features: bool = True):
        
        self.lookback_periods = lookback_periods
        self.use_technical_indicators = use_technical_indicators
        self.use_microstructure = use_microstructure
        self.use_volatility_features = use_volatility_features
        self.use_trend_features = use_trend_features
        
        self.logger = get_logger("feature_extractor")
        
        # Feature scalers
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def extract_features(self, 
                        price_data: pd.DataFrame,
                        volume_data: Optional[pd.DataFrame] = None,
                        orderbook_data: Optional[Dict[str, Any]] = None,
                        external_data: Optional[Dict[str, Any]] = None) -> MarketFeatures:
        """
        Extract comprehensive market features for regime classification.
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume data
            orderbook_data: Optional order book data
            external_data: Optional external data (funding rates, sentiment, etc.)
            
        Returns:
            MarketFeatures object
        """
        
        try:
            timestamp = price_data.index[-1] if hasattr(price_data.index[-1], 'to_pydatetime') else datetime.now()
            
            # Initialize features
            features = MarketFeatures(timestamp=timestamp)
            
            # Extract price-based features
            if 'close' in price_data.columns:
                features = self._extract_price_features(features, price_data)
            
            # Extract volatility features
            if self.use_volatility_features:
                features = self._extract_volatility_features(features, price_data)
            
            # Extract volume features
            if volume_data is not None or 'volume' in price_data.columns:
                features = self._extract_volume_features(features, price_data, volume_data)
            
            # Extract technical indicators
            if self.use_technical_indicators:
                features = self._extract_technical_features(features, price_data)
            
            # Extract microstructure features
            if self.use_microstructure and orderbook_data:
                features = self._extract_microstructure_features(features, orderbook_data)
            
            # Extract external features
            if external_data:
                features = self._extract_external_features(features, external_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return MarketFeatures(timestamp=datetime.now())
    
    def _extract_price_features(self, features: MarketFeatures, price_data: pd.DataFrame) -> MarketFeatures:
        """Extract price-based features."""
        try:
            close_prices = price_data['close'].values
            
            # Calculate returns for different periods
            for period in self.lookback_periods:
                if len(close_prices) > period:
                    period_return = (close_prices[-1] - close_prices[-period-1]) / close_prices[-period-1]
                    
                    if period == 1:
                        features.returns_1h = float(period_return)
                    elif period == 4:
                        features.returns_4h = float(period_return)
                    elif period == 24:
                        features.returns_24h = float(period_return)
                    elif period == 168:
                        features.returns_7d = float(period_return)
            
            # Calculate cross-asset correlation (if BTC data available)
            if 'btc_close' in price_data.columns:
                btc_returns = np.diff(price_data['btc_close'].values[-50:])
                asset_returns = np.diff(close_prices[-50:])
                
                if len(btc_returns) > 10 and len(asset_returns) > 10:
                    min_len = min(len(btc_returns), len(asset_returns))
                    correlation = np.corrcoef(btc_returns[-min_len:], asset_returns[-min_len:])[0, 1]
                    features.btc_correlation = float(correlation) if not np.isnan(correlation) else 0.0
            
            # Calculate market beta (simplified)
            if len(close_prices) > 30:
                returns = np.diff(close_prices[-30:]) / close_prices[-31:-1]
                market_returns = returns  # Simplified - would use market index
                
                if np.std(market_returns) > 0:
                    beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
                    features.market_beta = float(beta) if not np.isnan(beta) else 1.0
            
        except Exception as e:
            self.logger.warning(f"Error extracting price features: {e}")
        
        return features
    
    def _extract_volatility_features(self, features: MarketFeatures, price_data: pd.DataFrame) -> MarketFeatures:
        """Extract volatility features."""
        try:
            close_prices = price_data['close'].values
            
            # Calculate volatility for different periods
            for period in self.lookback_periods:
                if len(close_prices) > period:
                    returns = np.diff(close_prices[-period-1:]) / close_prices[-period-1:-1]
                    volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized
                    
                    if period == 1:
                        features.volatility_1h = float(volatility)
                    elif period == 4:
                        features.volatility_4h = float(volatility)
                    elif period == 24:
                        features.volatility_24h = float(volatility)
                    elif period == 168:
                        features.volatility_7d = float(volatility)
            
        except Exception as e:
            self.logger.warning(f"Error extracting volatility features: {e}")
        
        return features
    
    def _extract_volume_features(self, features: MarketFeatures, 
                                price_data: pd.DataFrame, 
                                volume_data: Optional[pd.DataFrame] = None) -> MarketFeatures:
        """Extract volume features."""
        try:
            volume_column = 'volume'
            if volume_column in price_data.columns:
                volumes = price_data[volume_column].values
            elif volume_data is not None and 'volume' in volume_data.columns:
                volumes = volume_data['volume'].values
            else:
                return features
            
            # Calculate volume ratios for different periods
            for period in [1, 4, 24]:
                if len(volumes) > period:
                    current_volume = volumes[-1]
                    avg_volume = np.mean(volumes[-period-1:-1])
                    
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        
                        if period == 1:
                            features.volume_ratio_1h = float(volume_ratio)
                        elif period == 4:
                            features.volume_ratio_4h = float(volume_ratio)
                        elif period == 24:
                            features.volume_ratio_24h = float(volume_ratio)
            
        except Exception as e:
            self.logger.warning(f"Error extracting volume features: {e}")
        
        return features
    
    def _extract_technical_features(self, features: MarketFeatures, price_data: pd.DataFrame) -> MarketFeatures:
        """Extract technical indicator features."""
        try:
            close_prices = price_data['close'].values
            
            # RSI
            features.rsi_14 = TechnicalIndicators.rsi(close_prices, 14)
            
            # MACD Signal
            features.macd_signal = TechnicalIndicators.macd(close_prices)
            
            # Bollinger Bands position
            features.bollinger_position = TechnicalIndicators.bollinger_position(close_prices)
            
        except Exception as e:
            self.logger.warning(f"Error extracting technical features: {e}")
        
        return features
    
    def _extract_microstructure_features(self, features: MarketFeatures, orderbook_data: Dict[str, Any]) -> MarketFeatures:
        """Extract market microstructure features."""
        try:
            # Bid-ask spread
            if 'bid_prices' in orderbook_data and 'ask_prices' in orderbook_data:
                bid_prices = np.array(orderbook_data['bid_prices'])
                ask_prices = np.array(orderbook_data['ask_prices'])
                features.bid_ask_spread = MarketMicrostructureFeatures.calculate_bid_ask_spread(bid_prices, ask_prices)
            
            # Order book imbalance
            if 'bid_volumes' in orderbook_data and 'ask_volumes' in orderbook_data:
                bid_volumes = np.array(orderbook_data['bid_volumes'])
                ask_volumes = np.array(orderbook_data['ask_volumes'])
                features.order_book_imbalance = MarketMicrostructureFeatures.calculate_order_book_imbalance(bid_volumes, ask_volumes)
            
            # Trade intensity
            if 'trade_times' in orderbook_data:
                trade_times = orderbook_data['trade_times']
                features.trade_intensity = MarketMicrostructureFeatures.calculate_trade_intensity(trade_times)
            
        except Exception as e:
            self.logger.warning(f"Error extracting microstructure features: {e}")
        
        return features
    
    def _extract_external_features(self, features: MarketFeatures, external_data: Dict[str, Any]) -> MarketFeatures:
        """Extract external features."""
        try:
            # Funding rate
            if 'funding_rate' in external_data:
                features.funding_rate = float(external_data['funding_rate'])
            
            # Open interest change
            if 'open_interest_change' in external_data:
                features.open_interest_change = float(external_data['open_interest_change'])
            
            # Fear & Greed Index
            if 'fear_greed_index' in external_data:
                features.fear_greed_index = float(external_data['fear_greed_index'])
            
            # Social sentiment
            if 'social_sentiment' in external_data:
                features.social_sentiment = float(external_data['social_sentiment'])
            
        except Exception as e:
            self.logger.warning(f"Error extracting external features: {e}")
        
        return features
    
    def fit_scaler(self, feature_arrays: List[np.ndarray]):
        """Fit the feature scaler on training data."""
        if feature_arrays:
            all_features = np.vstack(feature_arrays)
            self.scaler.fit(all_features)
            self.is_fitted = True
            self.logger.info("Feature scaler fitted on training data")
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler."""
        if not self.is_fitted:
            self.logger.warning("Scaler not fitted, returning unscaled features")
            return features
        
        return self.scaler.transform(features.reshape(1, -1)).flatten()
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'returns_1h', 'returns_4h', 'returns_24h', 'returns_7d',
            'volatility_1h', 'volatility_4h', 'volatility_24h', 'volatility_7d',
            'volume_ratio_1h', 'volume_ratio_4h', 'volume_ratio_24h',
            'rsi_14', 'macd_signal', 'bollinger_position',
            'bid_ask_spread', 'order_book_imbalance', 'trade_intensity',
            'btc_correlation', 'market_beta',
            'funding_rate', 'open_interest_change',
            'fear_greed_index', 'social_sentiment'
        ]
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return {}
            
            feature_names = self.get_feature_names()
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
