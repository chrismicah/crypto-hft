"""
Feature engineering for on-chain data signals.
Transforms raw on-chain metrics into ML-ready features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta

from ..models import (
    OnChainDataPoint, OnChainMetrics, ExchangeFlowMetrics,
    WhaleActivity, NetworkHealthMetrics, OnChainFeatureSet,
    OnChainMetricType
)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    lookback_windows: List[int] = None  # [7, 14, 30] days
    ma_windows: List[int] = None        # [3, 7, 14] days for moving averages
    volatility_windows: List[int] = None # [7, 14] days for volatility
    zscore_windows: List[int] = None    # [30, 60] days for z-score normalization
    
    # Feature scaling
    use_robust_scaling: bool = True
    clip_outliers: bool = True
    outlier_threshold: float = 3.0
    
    # Technical indicators
    use_technical_indicators: bool = True
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [7, 14, 30]
        if self.ma_windows is None:
            self.ma_windows = [3, 7, 14]
        if self.volatility_windows is None:
            self.volatility_windows = [7, 14]
        if self.zscore_windows is None:
            self.zscore_windows = [30, 60]


class OnChainFeatureEngineer:
    """
    Feature engineering pipeline for on-chain data.
    
    Transforms raw on-chain metrics into features suitable for ML models:
    - Time-based features (moving averages, volatility, momentum)
    - Statistical features (z-scores, percentiles, anomaly detection)
    - Technical indicators adapted for on-chain data
    - Cross-asset features and ratios
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.logger = logging.getLogger(__name__)
        
        # Scalers for feature normalization
        self.scaler = RobustScaler() if config.use_robust_scaling else StandardScaler()
        self.fitted_scalers = {}
        
        # Historical data cache for incremental processing
        self.data_cache = {}
    
    def create_dataframe_from_metrics(
        self,
        metrics_list: List[OnChainMetrics]
    ) -> pd.DataFrame:
        """
        Convert list of OnChainMetrics to pandas DataFrame.
        
        Args:
            metrics_list: List of OnChainMetrics objects
            
        Returns:
            DataFrame with timestamp index and metric columns
        """
        if not metrics_list:
            return pd.DataFrame()
        
        data = []
        for metric in metrics_list:
            row = {
                'timestamp': metric.timestamp,
                'symbol': metric.symbol,
                'exchange_inflow': metric.exchange_inflow_btc,
                'exchange_outflow': metric.exchange_outflow_btc,
                'exchange_net_flow': metric.exchange_net_flow_btc,
                'active_addresses': metric.active_addresses_24h,
                'transaction_count': metric.transaction_count_24h,
                'transaction_volume': metric.transaction_volume_usd,
                'whale_transactions': metric.whale_transaction_count,
                'large_tx_volume': metric.large_transaction_volume,
                'hash_rate': metric.hash_rate_7d_ma,
                'supply_1y_plus': metric.supply_1y_plus,
                'supply_short_term': metric.supply_short_term,
                'realized_cap': metric.realized_cap,
                'mvrv_ratio': metric.mvrv_ratio,
                'nvt_ratio': metric.nvt_ratio,
                'total_value_locked': metric.total_value_locked
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Fill missing values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features (moving averages, volatility, momentum).
        
        Args:
            df: Input DataFrame with raw metrics
            
        Returns:
            DataFrame with additional time-based features
        """
        result_df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().all():
                continue
                
            # Moving averages
            for window in self.config.ma_windows:
                result_df[f'{col}_ma_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Moving average momentum
                result_df[f'{col}_ma_momentum_{window}d'] = (
                    df[col] / result_df[f'{col}_ma_{window}d'] - 1
                )
            
            # Volatility (rolling standard deviation)
            for window in self.config.volatility_windows:
                result_df[f'{col}_volatility_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Coefficient of variation (volatility relative to mean)
                mean_val = df[col].rolling(window=window, min_periods=1).mean()
                result_df[f'{col}_cv_{window}d'] = (
                    result_df[f'{col}_volatility_{window}d'] / mean_val
                ).replace([np.inf, -np.inf], np.nan)
            
            # Price momentum (rate of change)
            for window in self.config.lookback_windows:
                result_df[f'{col}_momentum_{window}d'] = df[col].pct_change(periods=window)
                
                # Acceleration (second derivative)
                result_df[f'{col}_acceleration_{window}d'] = (
                    result_df[f'{col}_momentum_{window}d'].diff()
                )
            
            # Z-scores (standardized values)
            for window in self.config.zscore_windows:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                result_df[f'{col}_zscore_{window}d'] = (
                    (df[col] - rolling_mean) / rolling_std
                ).replace([np.inf, -np.inf], np.nan)
        
        return result_df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features (percentiles, anomaly scores).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional statistical features
        """
        result_df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().all():
                continue
            
            # Rolling percentiles
            for window in [30, 60, 90]:
                result_df[f'{col}_percentile_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
            
            # Anomaly detection (distance from rolling median)
            for window in [14, 30]:
                rolling_median = df[col].rolling(window=window, min_periods=1).median()
                rolling_mad = df[col].rolling(window=window, min_periods=1).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                result_df[f'{col}_anomaly_score_{window}d'] = (
                    np.abs(df[col] - rolling_median) / rolling_mad
                ).replace([np.inf, -np.inf], np.nan)
            
            # Trend detection (linear regression slope)
            for window in [7, 14, 30]:
                def calculate_trend(series):
                    if len(series) < 2:
                        return 0
                    x = np.arange(len(series))
                    slope, _, _, _, _ = stats.linregress(x, series)
                    return slope
                
                result_df[f'{col}_trend_{window}d'] = df[col].rolling(
                    window=window, min_periods=2
                ).apply(calculate_trend)
        
        return result_df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators adapted for on-chain data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        if not self.config.use_technical_indicators:
            return df
        
        result_df = df.copy()
        
        # Apply technical indicators to key metrics
        key_metrics = ['exchange_net_flow', 'active_addresses', 'whale_transactions']
        
        for metric in key_metrics:
            if metric not in df.columns or df[metric].isnull().all():
                continue
            
            series = df[metric].fillna(method='ffill')
            
            # RSI adapted for on-chain metrics
            if len(series) >= self.config.rsi_period:
                result_df[f'{metric}_rsi'] = ta.momentum.RSIIndicator(
                    close=series, window=self.config.rsi_period
                ).rsi()
            
            # MACD for trend analysis
            if len(series) >= self.config.macd_slow:
                macd = ta.trend.MACD(
                    close=series,
                    window_fast=self.config.macd_fast,
                    window_slow=self.config.macd_slow,
                    window_sign=self.config.macd_signal
                )
                result_df[f'{metric}_macd'] = macd.macd()
                result_df[f'{metric}_macd_signal'] = macd.macd_signal()
                result_df[f'{metric}_macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands for volatility
            if len(series) >= 20:
                bollinger = ta.volatility.BollingerBands(close=series, window=20)
                result_df[f'{metric}_bb_upper'] = bollinger.bollinger_hband()
                result_df[f'{metric}_bb_lower'] = bollinger.bollinger_lband()
                result_df[f'{metric}_bb_position'] = (
                    (series - bollinger.bollinger_lband()) / 
                    (bollinger.bollinger_hband() - bollinger.bollinger_lband())
                )
        
        return result_df
    
    def create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that combine multiple metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cross-asset features
        """
        result_df = df.copy()
        
        # Exchange flow ratios
        if 'exchange_inflow' in df.columns and 'exchange_outflow' in df.columns:
            result_df['flow_ratio'] = (
                df['exchange_outflow'] / df['exchange_inflow']
            ).replace([np.inf, -np.inf], np.nan)
            
            # Flow pressure indicator
            result_df['flow_pressure'] = (
                (df['exchange_outflow'] - df['exchange_inflow']) / 
                (df['exchange_outflow'] + df['exchange_inflow'])
            ).replace([np.inf, -np.inf], np.nan)
        
        # Network activity ratios
        if 'active_addresses' in df.columns and 'transaction_count' in df.columns:
            result_df['tx_per_address'] = (
                df['transaction_count'] / df['active_addresses']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Whale activity relative to network
        if 'whale_transactions' in df.columns and 'transaction_count' in df.columns:
            result_df['whale_dominance'] = (
                df['whale_transactions'] / df['transaction_count']
            ).replace([np.inf, -np.inf], np.nan)
        
        # HODLer behavior
        if 'supply_1y_plus' in df.columns and 'supply_short_term' in df.columns:
            result_df['hodl_ratio'] = (
                df['supply_1y_plus'] / df['supply_short_term']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Market valuation metrics
        if 'mvrv_ratio' in df.columns and 'nvt_ratio' in df.columns:
            result_df['valuation_composite'] = (
                df['mvrv_ratio'] * df['nvt_ratio']
            ).replace([np.inf, -np.inf], np.nan)
        
        return result_df
    
    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite scores combining multiple features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame with composite scores
        """
        result_df = df.copy()
        
        # Bullish score: combination of positive indicators
        bullish_features = []
        
        # Exchange outflows (bullish)
        if 'exchange_net_flow' in df.columns:
            bullish_features.append(df['exchange_net_flow'].fillna(0))
        
        # High whale activity (can be bullish)
        if 'whale_transactions' in df.columns:
            whale_zscore = (df['whale_transactions'] - df['whale_transactions'].mean()) / df['whale_transactions'].std()
            bullish_features.append(whale_zscore.fillna(0))
        
        # Network growth
        if 'active_addresses' in df.columns:
            addr_momentum = df['active_addresses'].pct_change(7).fillna(0)
            bullish_features.append(addr_momentum)
        
        if bullish_features:
            result_df['onchain_bullish_score'] = np.mean(bullish_features, axis=0)
        
        # Bearish score: combination of negative indicators
        bearish_features = []
        
        # Exchange inflows (bearish)
        if 'exchange_net_flow' in df.columns:
            bearish_features.append(-df['exchange_net_flow'].fillna(0))
        
        # Decreasing network activity
        if 'active_addresses' in df.columns:
            addr_decline = -df['active_addresses'].pct_change(7).fillna(0)
            bearish_features.append(addr_decline)
        
        if bearish_features:
            result_df['onchain_bearish_score'] = np.mean(bearish_features, axis=0)
        
        # Momentum score
        momentum_features = []
        for col in ['exchange_net_flow', 'active_addresses', 'whale_transactions']:
            if f'{col}_momentum_7d' in df.columns:
                momentum_features.append(df[f'{col}_momentum_7d'].fillna(0))
        
        if momentum_features:
            result_df['onchain_momentum_score'] = np.mean(momentum_features, axis=0)
        
        return result_df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using robust scaling.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with normalized features
        """
        result_df = df.copy()
        
        # Get numeric columns (exclude metadata columns)
        exclude_cols = ['symbol']
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
        
        if not numeric_cols:
            return result_df
        
        # Handle outliers if configured
        if self.config.clip_outliers:
            for col in numeric_cols:
                if col in result_df.columns:
                    # Clip outliers using IQR method
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    result_df[col] = result_df[col].clip(lower_bound, upper_bound)
        
        # Apply scaling
        if fit:
            if numeric_cols:
                self.scaler.fit(result_df[numeric_cols].fillna(0))
                self.fitted_scalers['main'] = self.scaler
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            scaled_values = self.scaler.transform(result_df[numeric_cols].fillna(0))
            result_df[numeric_cols] = scaled_values
        
        return result_df
    
    def engineer_features(
        self,
        metrics_list: List[OnChainMetrics],
        normalize: bool = True,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            metrics_list: List of OnChainMetrics objects
            normalize: Whether to normalize features
            fit_scaler: Whether to fit the scaler
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info(f"Engineering features for {len(metrics_list)} metric records")
        
        # Convert to DataFrame
        df = self.create_dataframe_from_metrics(metrics_list)
        
        if df.empty:
            self.logger.warning("No data to process")
            return df
        
        # Apply feature engineering steps
        df = self.create_time_based_features(df)
        self.logger.debug("Created time-based features")
        
        df = self.create_statistical_features(df)
        self.logger.debug("Created statistical features")
        
        df = self.create_technical_indicators(df)
        self.logger.debug("Created technical indicators")
        
        df = self.create_cross_asset_features(df)
        self.logger.debug("Created cross-asset features")
        
        df = self.create_composite_scores(df)
        self.logger.debug("Created composite scores")
        
        # Normalize features if requested
        if normalize:
            df = self.normalize_features(df, fit=fit_scaler)
            self.logger.debug("Normalized features")
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        self.logger.info(f"Feature engineering complete: {df.shape[1]} features, {df.shape[0]} records")
        return df
    
    def create_feature_sets(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[OnChainFeatureSet]:
        """
        Convert DataFrame to OnChainFeatureSet objects.
        
        Args:
            df: DataFrame with engineered features
            symbol: Asset symbol
            
        Returns:
            List of OnChainFeatureSet objects
        """
        feature_sets = []
        
        for timestamp, row in df.iterrows():
            # Extract key composite scores
            bullish_score = row.get('onchain_bullish_score', 0)
            bearish_score = row.get('onchain_bearish_score', 0)
            momentum_score = row.get('onchain_momentum_score', 0)
            
            # Extract flow features
            flow_ma_7d = row.get('exchange_net_flow_ma_7d', 0)
            flow_ma_30d = row.get('exchange_net_flow_ma_30d', 0)
            flow_momentum = row.get('exchange_net_flow_momentum_7d', 0)
            
            # Extract whale features
            whale_ma_7d = row.get('whale_transactions_ma_7d', 0)
            whale_volatility = row.get('whale_transactions_volatility_7d', 0)
            
            # Create feature set
            feature_set = OnChainFeatureSet(
                symbol=symbol,
                timestamp=timestamp,
                exchange_flow_ma_7d=flow_ma_7d,
                exchange_flow_ma_30d=flow_ma_30d,
                exchange_flow_momentum=flow_momentum,
                whale_activity_ma_7d=whale_ma_7d,
                whale_activity_volatility=whale_volatility,
                onchain_bullish_score=bullish_score,
                onchain_bearish_score=bearish_score,
                onchain_momentum_score=momentum_score
            )
            
            # Store all numeric features in raw_metrics
            for col, value in row.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_set.raw_metrics[col] = float(value)
            
            feature_sets.append(feature_set)
        
        return feature_sets
