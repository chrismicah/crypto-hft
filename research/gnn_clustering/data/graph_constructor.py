"""
Graph construction from multi-asset crypto market data.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import ccxt
import asyncio
import logging
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import networkx as nx

from ..models.graph_models import GraphSnapshot


@dataclass
class MarketDataPoint:
    """Single market data observation."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    market_cap: Optional[float] = None
    
    def get_returns(self, previous: 'MarketDataPoint') -> float:
        """Calculate returns from previous data point."""
        if previous.close == 0:
            return 0.0
        return (self.close - previous.close) / previous.close
    
    def get_log_returns(self, previous: 'MarketDataPoint') -> float:
        """Calculate log returns from previous data point."""
        if previous.close <= 0 or self.close <= 0:
            return 0.0
        return np.log(self.close / previous.close)


class CryptoDataCollector:
    """
    Collects multi-asset crypto market data for graph construction.
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        max_retries: int = 3
    ):
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # Default symbols (top crypto assets)
        if symbols is None:
            self.symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SHIB/USDT',
                'MATIC/USDT', 'UNI/USDT', 'LINK/USDT', 'ATOM/USDT', 'LTC/USDT',
                'BCH/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'FIL/USDT'
            ]
        else:
            self.symbols = symbols
        
        # Initialize exchange
        self.exchange = getattr(ccxt, exchange_id)({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
    
    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> Dict[str, List[MarketDataPoint]]:
        """
        Collect historical data for all symbols.
        
        Args:
            start_date: Start of data collection period
            end_date: End of data collection period
            limit: Maximum number of candles per request
            
        Returns:
            Dictionary mapping symbols to market data points
        """
        self.logger.info(f"Collecting data for {len(self.symbols)} symbols from {start_date} to {end_date}")
        
        all_data = {}
        
        for symbol in self.symbols:
            try:
                data_points = await self._collect_symbol_data(
                    symbol, start_date, end_date, limit
                )
                all_data[symbol] = data_points
                self.logger.info(f"Collected {len(data_points)} data points for {symbol}")
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Failed to collect data for {symbol}: {e}")
                all_data[symbol] = []
        
        return all_data
    
    async def _collect_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int
    ) -> List[MarketDataPoint]:
        """Collect data for a single symbol."""
        data_points = []
        current_start = start_date
        
        while current_start < end_date:
            for attempt in range(self.max_retries):
                try:
                    # Convert to timestamp
                    since = int(current_start.timestamp() * 1000)
                    
                    # Fetch OHLCV data
                    ohlcv = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.exchange.fetch_ohlcv(
                            symbol, self.timeframe, since=since, limit=limit
                        )
                    )
                    
                    if not ohlcv:
                        break
                    
                    # Convert to MarketDataPoint objects
                    for candle in ohlcv:
                        timestamp = datetime.fromtimestamp(candle[0] / 1000)
                        
                        if timestamp > end_date:
                            break
                        
                        data_point = MarketDataPoint(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(candle[1]),
                            high=float(candle[2]),
                            low=float(candle[3]),
                            close=float(candle[4]),
                            volume=float(candle[5])
                        )
                        data_points.append(data_point)
                    
                    # Update start time for next batch
                    if ohlcv:
                        last_timestamp = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                        current_start = last_timestamp + timedelta(hours=1)
                    else:
                        break
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return data_points
    
    def get_aligned_dataframe(
        self,
        market_data: Dict[str, List[MarketDataPoint]]
    ) -> pd.DataFrame:
        """
        Convert market data to aligned DataFrame.
        
        Args:
            market_data: Dictionary of symbol -> data points
            
        Returns:
            DataFrame with timestamps as index and symbols as columns
        """
        # Create DataFrame for each symbol
        symbol_dfs = {}
        
        for symbol, data_points in market_data.items():
            if not data_points:
                continue
            
            df_data = []
            for dp in data_points:
                df_data.append({
                    'timestamp': dp.timestamp,
                    'close': dp.close,
                    'volume': dp.volume,
                    'returns': 0.0,  # Will be calculated
                    'log_returns': 0.0
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Add volatility features
            df['volatility_5'] = df['returns'].rolling(5).std()
            df['volatility_24'] = df['returns'].rolling(24).std()
            
            # Add momentum features
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_24'] = df['close'] / df['close'].shift(24) - 1
            
            # Clean symbol name for column
            clean_symbol = symbol.replace('/', '_')
            symbol_dfs[clean_symbol] = df
        
        # Align all DataFrames to common timestamp index
        if not symbol_dfs:
            return pd.DataFrame()
        
        # Find common time range
        start_times = [df.index.min() for df in symbol_dfs.values()]
        end_times = [df.index.max() for df in symbol_dfs.values()]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        # Create aligned DataFrame
        aligned_data = {}
        common_index = None
        
        for symbol, df in symbol_dfs.items():
            df_filtered = df.loc[common_start:common_end]
            
            if common_index is None:
                common_index = df_filtered.index
            else:
                # Take intersection of indices
                common_index = common_index.intersection(df_filtered.index)
        
        # Build final aligned DataFrame
        for symbol, df in symbol_dfs.items():
            df_filtered = df.loc[common_index]
            
            for col in ['close', 'volume', 'returns', 'log_returns', 'volatility_5', 'volatility_24', 'momentum_5', 'momentum_24']:
                aligned_data[f"{symbol}_{col}"] = df_filtered[col]
        
        aligned_df = pd.DataFrame(aligned_data, index=common_index)
        
        # Forward fill and backward fill missing values
        aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
        
        return aligned_df


class GraphConstructor:
    """
    Constructs dynamic graphs from multi-asset market data.
    """
    
    def __init__(
        self,
        correlation_window: int = 24,  # Hours for correlation calculation
        edge_threshold: float = 0.3,   # Minimum correlation for edge
        max_edges: int = 200,          # Maximum edges in graph
        feature_set: str = "comprehensive"  # "basic", "technical", "comprehensive"
    ):
        self.correlation_window = correlation_window
        self.edge_threshold = edge_threshold
        self.max_edges = max_edges
        self.feature_set = feature_set
        self.logger = logging.getLogger(__name__)
        
        # Feature scalers
        self.scalers = {}
    
    def construct_graph_sequence(
        self,
        aligned_df: pd.DataFrame,
        window_size: int = 100,  # Number of timestamps per graph
        stride: int = 1          # Stride between graphs
    ) -> List[GraphSnapshot]:
        """
        Construct sequence of graph snapshots from time series data.
        
        Args:
            aligned_df: Aligned DataFrame with all symbols and features
            window_size: Number of time periods to use for each graph
            stride: Step size between consecutive graphs
            
        Returns:
            List of GraphSnapshot objects
        """
        if aligned_df.empty:
            return []
        
        self.logger.info(f"Constructing graph sequence from {len(aligned_df)} timestamps")
        
        # Extract symbol list
        symbols = self._extract_symbols(aligned_df)
        if len(symbols) < 2:
            self.logger.warning("Need at least 2 symbols for graph construction")
            return []
        
        graphs = []
        
        # Create graphs with sliding window
        for start_idx in range(0, len(aligned_df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            window_data = aligned_df.iloc[start_idx:end_idx]
            timestamp = window_data.index[-1]  # Use last timestamp as graph timestamp
            
            try:
                graph = self._construct_single_graph(window_data, symbols, timestamp)
                graphs.append(graph)
                
                if len(graphs) % 100 == 0:
                    self.logger.info(f"Constructed {len(graphs)} graphs")
                    
            except Exception as e:
                self.logger.warning(f"Failed to construct graph at {timestamp}: {e}")
                continue
        
        self.logger.info(f"Successfully constructed {len(graphs)} graph snapshots")
        return graphs
    
    def _extract_symbols(self, aligned_df: pd.DataFrame) -> List[str]:
        """Extract unique symbols from DataFrame columns."""
        symbols = set()
        
        for col in aligned_df.columns:
            symbol = col.split('_')[0]  # Symbol is before first underscore
            symbols.add(symbol)
        
        return sorted(list(symbols))
    
    def _construct_single_graph(
        self,
        window_data: pd.DataFrame,
        symbols: List[str],
        timestamp: datetime
    ) -> GraphSnapshot:
        """Construct single graph snapshot from window data."""
        
        # Extract node features
        node_features = self._extract_node_features(window_data, symbols)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(window_data, symbols)
        
        # Calculate volatility vector
        volatility_vector = self._calculate_volatility_vector(window_data, symbols)
        
        # Construct edges based on correlations
        edge_index, edge_weights = self._construct_edges(correlation_matrix)
        
        # Convert to tensors
        node_features_tensor = torch.FloatTensor(node_features)
        edge_index_tensor = torch.LongTensor(edge_index)
        edge_weights_tensor = torch.FloatTensor(edge_weights)
        
        return GraphSnapshot(
            timestamp=timestamp,
            node_features=node_features_tensor,
            edge_index=edge_index_tensor,
            edge_weights=edge_weights_tensor,
            symbols=symbols,
            correlation_matrix=correlation_matrix,
            volatility_vector=volatility_vector
        )
    
    def _extract_node_features(
        self,
        window_data: pd.DataFrame,
        symbols: List[str]
    ) -> np.ndarray:
        """Extract node features for each symbol."""
        features = []
        
        for symbol in symbols:
            symbol_features = []
            
            # Price-based features
            if f"{symbol}_close" in window_data.columns:
                close_prices = window_data[f"{symbol}_close"].values
                
                # Current price (normalized)
                if len(close_prices) > 0:
                    current_price = close_prices[-1]
                    price_norm = current_price / close_prices[0] if close_prices[0] > 0 else 1.0
                    symbol_features.append(price_norm)
                else:
                    symbol_features.append(1.0)
                
                # Price momentum features
                if len(close_prices) >= 5:
                    short_momentum = close_prices[-1] / close_prices[-5] - 1
                    symbol_features.append(short_momentum)
                else:
                    symbol_features.append(0.0)
                
                if len(close_prices) >= 24:
                    long_momentum = close_prices[-1] / close_prices[-24] - 1
                    symbol_features.append(long_momentum)
                else:
                    symbol_features.append(0.0)
            else:
                symbol_features.extend([1.0, 0.0, 0.0])
            
            # Volatility features
            if f"{symbol}_volatility_5" in window_data.columns:
                vol_5 = window_data[f"{symbol}_volatility_5"].iloc[-1]
                symbol_features.append(vol_5 if not pd.isna(vol_5) else 0.0)
            else:
                symbol_features.append(0.0)
            
            if f"{symbol}_volatility_24" in window_data.columns:
                vol_24 = window_data[f"{symbol}_volatility_24"].iloc[-1]
                symbol_features.append(vol_24 if not pd.isna(vol_24) else 0.0)
            else:
                symbol_features.append(0.0)
            
            # Volume features
            if f"{symbol}_volume" in window_data.columns:
                volumes = window_data[f"{symbol}_volume"].values
                if len(volumes) > 0:
                    avg_volume = np.mean(volumes)
                    current_volume = volumes[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    symbol_features.append(np.log(volume_ratio + 1))  # Log transform
                else:
                    symbol_features.append(0.0)
            else:
                symbol_features.append(0.0)
            
            # Returns-based features
            if f"{symbol}_returns" in window_data.columns:
                returns = window_data[f"{symbol}_returns"].values
                if len(returns) > 1:
                    # Recent return
                    recent_return = returns[-1] if not pd.isna(returns[-1]) else 0.0
                    symbol_features.append(recent_return)
                    
                    # Return volatility
                    return_vol = np.std(returns[~pd.isna(returns)])
                    symbol_features.append(return_vol)
                    
                    # Skewness and kurtosis
                    from scipy import stats
                    clean_returns = returns[~pd.isna(returns)]
                    if len(clean_returns) > 3:
                        skew = stats.skew(clean_returns)
                        kurt = stats.kurtosis(clean_returns)
                        symbol_features.extend([skew, kurt])
                    else:
                        symbol_features.extend([0.0, 0.0])
                else:
                    symbol_features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                symbol_features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Technical indicators (if comprehensive features)
            if self.feature_set == "comprehensive":
                symbol_features.extend(self._calculate_technical_features(window_data, symbol))
            
            features.append(symbol_features)
        
        features_array = np.array(features, dtype=np.float32)
        
        # Handle NaN and infinite values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale features
        if features_array.shape[0] > 1:  # Need multiple samples for scaling
            if 'node_features' not in self.scalers:
                from sklearn.preprocessing import RobustScaler
                self.scalers['node_features'] = RobustScaler()
                features_array = self.scalers['node_features'].fit_transform(features_array)
            else:
                features_array = self.scalers['node_features'].transform(features_array)
        
        return features_array
    
    def _calculate_technical_features(
        self,
        window_data: pd.DataFrame,
        symbol: str
    ) -> List[float]:
        """Calculate technical indicator features."""
        features = []
        
        if f"{symbol}_close" in window_data.columns:
            prices = window_data[f"{symbol}_close"].values
            
            # RSI-like momentum
            if len(prices) >= 14:
                gains = np.maximum(np.diff(prices), 0)
                losses = np.maximum(-np.diff(prices), 0)
                avg_gain = np.mean(gains[-13:])
                avg_loss = np.mean(losses[-13:])
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features.append(rsi / 100.0)  # Normalize to [0, 1]
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # Price position in recent range
            if len(prices) >= 20:
                recent_high = np.max(prices[-20:])
                recent_low = np.min(prices[-20:])
                current_price = prices[-1]
                
                if recent_high > recent_low:
                    price_position = (current_price - recent_low) / (recent_high - recent_low)
                    features.append(price_position)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
        else:
            features.extend([0.5, 0.5])
        
        return features
    
    def _calculate_correlation_matrix(
        self,
        window_data: pd.DataFrame,
        symbols: List[str]
    ) -> np.ndarray:
        """Calculate correlation matrix between symbols."""
        
        # Extract return series for each symbol
        return_series = {}
        
        for symbol in symbols:
            if f"{symbol}_returns" in window_data.columns:
                returns = window_data[f"{symbol}_returns"].values
                # Clean returns (remove NaN)
                clean_returns = returns[~pd.isna(returns)]
                if len(clean_returns) >= 5:  # Need minimum data for correlation
                    return_series[symbol] = clean_returns
        
        # Calculate pairwise correlations
        n_symbols = len(symbols)
        correlation_matrix = np.eye(n_symbols)  # Start with identity matrix
        
        symbol_indices = {sym: i for i, sym in enumerate(symbols)}
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if sym1 in return_series and sym2 in return_series:
                    try:
                        # Align return series
                        returns1 = return_series[sym1]
                        returns2 = return_series[sym2]
                        
                        min_len = min(len(returns1), len(returns2))
                        if min_len >= 5:
                            corr, _ = pearsonr(returns1[-min_len:], returns2[-min_len:])
                            
                            if not np.isnan(corr):
                                correlation_matrix[i, j] = corr
                                correlation_matrix[j, i] = corr
                    
                    except Exception:
                        # Keep default correlation of 0
                        pass
        
        return correlation_matrix
    
    def _calculate_volatility_vector(
        self,
        window_data: pd.DataFrame,
        symbols: List[str]
    ) -> np.ndarray:
        """Calculate volatility for each symbol."""
        volatilities = []
        
        for symbol in symbols:
            if f"{symbol}_returns" in window_data.columns:
                returns = window_data[f"{symbol}_returns"].values
                clean_returns = returns[~pd.isna(returns)]
                
                if len(clean_returns) >= 5:
                    vol = np.std(clean_returns)
                    volatilities.append(vol)
                else:
                    volatilities.append(0.0)
            else:
                volatilities.append(0.0)
        
        return np.array(volatilities)
    
    def _construct_edges(
        self,
        correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct edges based on correlation matrix."""
        
        # Get upper triangle indices (avoid self-loops and duplicate edges)
        n_nodes = correlation_matrix.shape[0]
        i_indices, j_indices = np.triu_indices(n_nodes, k=1)
        
        # Get correlations for these pairs
        correlations = correlation_matrix[i_indices, j_indices]
        
        # Filter by threshold
        strong_correlations = np.abs(correlations) >= self.edge_threshold
        
        if np.sum(strong_correlations) == 0:
            # If no strong correlations, take top correlations
            n_edges = min(self.max_edges // 2, len(correlations))
            top_indices = np.argsort(np.abs(correlations))[-n_edges:]
            strong_correlations = np.zeros_like(correlations, dtype=bool)
            strong_correlations[top_indices] = True
        
        # Select edges
        selected_i = i_indices[strong_correlations]
        selected_j = j_indices[strong_correlations]
        selected_weights = correlations[strong_correlations]
        
        # Create bidirectional edges
        edge_index = np.vstack([
            np.concatenate([selected_i, selected_j]),
            np.concatenate([selected_j, selected_i])
        ])
        
        edge_weights = np.concatenate([selected_weights, selected_weights])
        
        # Limit total number of edges
        if len(edge_weights) > self.max_edges:
            # Keep strongest edges
            sorted_indices = np.argsort(np.abs(edge_weights))[-self.max_edges:]
            edge_index = edge_index[:, sorted_indices]
            edge_weights = edge_weights[sorted_indices]
        
        return edge_index, edge_weights
