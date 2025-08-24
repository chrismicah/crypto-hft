"""
Data collectors for LLM analyst system.
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import aiohttp
import json
from dataclasses import asdict

from ..models.anomaly_models import (
    TimeSeriesData, MarketContext, SystemMetrics, TradingMetrics,
    DataSource, format_data_for_llm
)
from ...common.db.client import DatabaseClient
from ...common.logger import get_logger


class BaseDataCollector:
    """Base class for data collectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"data_collector.{name}")
        self._cache = {}
        self._cache_ttl = {}
    
    def _is_cache_valid(self, key: str, ttl_seconds: int = 300) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache or key not in self._cache_ttl:
            return False
        
        return (datetime.now() - self._cache_ttl[key]).total_seconds() < ttl_seconds
    
    def _set_cache(self, key: str, data: Any):
        """Set cached data."""
        self._cache[key] = data
        self._cache_ttl[key] = datetime.now()
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached data."""
        return self._cache.get(key)


class PnLDataCollector(BaseDataCollector):
    """Collector for PnL and trading performance data."""
    
    def __init__(self, db_client: DatabaseClient):
        super().__init__("pnl")
        self.db_client = db_client
    
    async def collect_pnl_data(
        self,
        start_time: datetime,
        end_time: datetime,
        granularity: str = "1h"
    ) -> TimeSeriesData:
        """Collect PnL data from database."""
        cache_key = f"pnl_{start_time}_{end_time}_{granularity}"
        
        if self._is_cache_valid(cache_key):
            return self._get_cache(cache_key)
        
        try:
            # Query portfolio PnL data
            query = """
            SELECT timestamp, portfolio_value, realized_pnl, unrealized_pnl
            FROM portfolio_pnl
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """
            
            with self.db_client.get_session() as session:
                result = session.execute(query, (start_time, end_time)).fetchall()
            
            if not result:
                self.logger.warning(f"No PnL data found for period {start_time} to {end_time}")
                return TimeSeriesData([], [], [])
            
            # Process data
            timestamps = []
            values = []
            labels = []
            
            for row in result:
                timestamps.append(row[0])
                total_pnl = (row[2] or 0) + (row[3] or 0)  # realized + unrealized
                values.append(float(total_pnl))
                labels.append(f"Portfolio: ${row[1]:.2f}")
            
            # Aggregate by granularity if needed
            if granularity != "raw":
                timestamps, values, labels = self._aggregate_data(
                    timestamps, values, labels, granularity
                )
            
            pnl_data = TimeSeriesData(
                timestamps=timestamps,
                values=values,
                labels=labels,
                metadata={
                    'source': 'database',
                    'granularity': granularity,
                    'total_points': len(values)
                }
            )
            
            self._set_cache(cache_key, pnl_data)
            return pnl_data
            
        except Exception as e:
            self.logger.error(f"Error collecting PnL data: {e}")
            return TimeSeriesData([], [], [])
    
    async def collect_trading_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[TradingMetrics]:
        """Collect detailed trading metrics."""
        try:
            # Query trades and calculate metrics
            trades_query = """
            SELECT timestamp, symbol, side, quantity, price, pnl
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """
            
            with self.db_client.get_session() as session:
                trades = session.execute(trades_query, (start_time, end_time)).fetchall()
            
            if not trades:
                return []
            
            # Calculate metrics by hour
            metrics = []
            current_hour = start_time.replace(minute=0, second=0, microsecond=0)
            
            while current_hour <= end_time:
                hour_end = current_hour + timedelta(hours=1)
                
                # Filter trades for this hour
                hour_trades = [t for t in trades if current_hour <= t[0] < hour_end]
                
                if hour_trades:
                    # Calculate metrics
                    pnls = [float(t[5]) for t in hour_trades if t[5] is not None]
                    
                    total_pnl = sum(pnls)
                    winning_trades = len([p for p in pnls if p > 0])
                    total_trades = len(pnls)
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    # Calculate Sharpe ratio (simplified)
                    if len(pnls) > 1:
                        sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
                    else:
                        sharpe = 0
                    
                    # Calculate drawdown (simplified)
                    cumulative_pnl = np.cumsum(pnls)
                    running_max = np.maximum.accumulate(cumulative_pnl)
                    drawdown = np.min(cumulative_pnl - running_max)
                    
                    metrics.append(TradingMetrics(
                        timestamp=current_hour,
                        pnl=total_pnl,
                        cumulative_pnl=float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0,
                        sharpe_ratio=sharpe,
                        max_drawdown=abs(drawdown),
                        win_rate=win_rate,
                        avg_trade_duration=1.0,  # Placeholder
                        total_trades=total_trades,
                        active_positions=0  # Would need position tracking
                    ))
                
                current_hour = hour_end
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
            return []
    
    def _aggregate_data(
        self,
        timestamps: List[datetime],
        values: List[float],
        labels: List[str],
        granularity: str
    ) -> Tuple[List[datetime], List[float], List[str]]:
        """Aggregate data by granularity."""
        if granularity == "raw":
            return timestamps, values, labels
        
        # Create DataFrame for easier aggregation
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'label': labels
        })
        df.set_index('timestamp', inplace=True)
        
        # Resample based on granularity
        if granularity == "1h":
            resampled = df.resample('1H').agg({
                'value': 'sum',
                'label': 'last'
            })
        elif granularity == "1d":
            resampled = df.resample('1D').agg({
                'value': 'sum',
                'label': 'last'
            })
        else:
            return timestamps, values, labels
        
        # Convert back to lists
        return (
            resampled.index.tolist(),
            resampled['value'].tolist(),
            resampled['label'].tolist()
        )


class MarketDataCollector(BaseDataCollector):
    """Collector for market data from external APIs."""
    
    def __init__(self):
        super().__init__("market")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_market_context(self, timestamp: datetime) -> MarketContext:
        """Collect market context for a specific timestamp."""
        cache_key = f"market_context_{timestamp.date()}"
        
        if self._is_cache_valid(cache_key, ttl_seconds=3600):  # 1 hour cache
            return self._get_cache(cache_key)
        
        try:
            # Collect various market data points
            btc_price, eth_price = await self._get_crypto_prices()
            volatility = await self._get_market_volatility()
            funding_rates = await self._get_funding_rates()
            volumes = await self._get_24h_volumes()
            fear_greed = await self._get_fear_greed_index()
            news = await self._get_major_news(timestamp)
            
            context = MarketContext(
                timestamp=timestamp,
                btc_price=btc_price,
                eth_price=eth_price,
                market_volatility=volatility,
                funding_rates=funding_rates,
                volume_24h=volumes,
                fear_greed_index=fear_greed,
                major_news=news,
                macro_events=[]  # Would need additional API
            )
            
            self._set_cache(cache_key, context)
            return context
            
        except Exception as e:
            self.logger.error(f"Error collecting market context: {e}")
            return MarketContext(
                timestamp=timestamp,
                btc_price=0.0,
                eth_price=0.0,
                market_volatility=0.0,
                funding_rates={},
                volume_24h={}
            )
    
    async def collect_price_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> TimeSeriesData:
        """Collect price data for a symbol."""
        cache_key = f"price_{symbol}_{start_time}_{end_time}_{interval}"
        
        if self._is_cache_valid(cache_key):
            return self._get_cache(cache_key)
        
        try:
            # Use Binance API for price data
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.replace('/', ''),
                'interval': interval,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'limit': 1000
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    timestamps = []
                    values = []
                    labels = []
                    
                    for candle in data:
                        timestamps.append(datetime.fromtimestamp(candle[0] / 1000))
                        values.append(float(candle[4]))  # Close price
                        labels.append(f"{symbol}: ${float(candle[4]):.2f}")
                    
                    price_data = TimeSeriesData(
                        timestamps=timestamps,
                        values=values,
                        labels=labels,
                        metadata={
                            'symbol': symbol,
                            'interval': interval,
                            'source': 'binance'
                        }
                    )
                    
                    self._set_cache(cache_key, price_data)
                    return price_data
                else:
                    self.logger.error(f"Failed to fetch price data: {response.status}")
                    return TimeSeriesData([], [], [])
                    
        except Exception as e:
            self.logger.error(f"Error collecting price data for {symbol}: {e}")
            return TimeSeriesData([], [], [])
    
    async def _get_crypto_prices(self) -> Tuple[float, float]:
        """Get current BTC and ETH prices."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get BTC price
            async with self.session.get(url, params={'symbol': 'BTCUSDT'}) as response:
                btc_data = await response.json()
                btc_price = float(btc_data['price'])
            
            # Get ETH price
            async with self.session.get(url, params={'symbol': 'ETHUSDT'}) as response:
                eth_data = await response.json()
                eth_price = float(eth_data['price'])
            
            return btc_price, eth_price
            
        except Exception as e:
            self.logger.error(f"Error getting crypto prices: {e}")
            return 0.0, 0.0
    
    async def _get_market_volatility(self) -> float:
        """Calculate market volatility."""
        try:
            # Get 24h ticker data for major symbols
            url = "https://api.binance.com/api/v3/ticker/24hr"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                # Calculate average price change percentage
                major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
                price_changes = []
                
                for ticker in data:
                    if ticker['symbol'] in major_symbols:
                        price_changes.append(abs(float(ticker['priceChangePercent'])))
                
                return np.mean(price_changes) if price_changes else 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {e}")
            return 0.0
    
    async def _get_funding_rates(self) -> Dict[str, float]:
        """Get funding rates for major perpetual contracts."""
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                funding_rates = {}
                major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
                
                for item in data:
                    if item['symbol'] in major_symbols:
                        funding_rates[item['symbol']] = float(item['lastFundingRate'])
                
                return funding_rates
                
        except Exception as e:
            self.logger.error(f"Error getting funding rates: {e}")
            return {}
    
    async def _get_24h_volumes(self) -> Dict[str, float]:
        """Get 24h trading volumes."""
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                volumes = {}
                major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
                
                for ticker in data:
                    if ticker['symbol'] in major_symbols:
                        volumes[ticker['symbol']] = float(ticker['quoteVolume'])
                
                return volumes
                
        except Exception as e:
            self.logger.error(f"Error getting 24h volumes: {e}")
            return {}
    
    async def _get_fear_greed_index(self) -> Optional[float]:
        """Get Fear & Greed Index."""
        try:
            url = "https://api.alternative.me/fng/"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                if data['data']:
                    return float(data['data'][0]['value'])
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting Fear & Greed Index: {e}")
            return None
    
    async def _get_major_news(self, timestamp: datetime) -> List[str]:
        """Get major crypto news (placeholder - would need news API)."""
        # This would integrate with news APIs like NewsAPI, CryptoCompare, etc.
        # For now, return empty list
        return []


class SystemMetricsCollector(BaseDataCollector):
    """Collector for system performance metrics."""
    
    def __init__(self):
        super().__init__("system")
    
    async def collect_system_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[SystemMetrics]:
        """Collect system metrics from monitoring systems."""
        try:
            # This would integrate with Prometheus, Grafana, or other monitoring
            # For now, simulate some metrics
            metrics = []
            current_time = start_time
            
            while current_time <= end_time:
                # Simulate metrics (in production, query from Prometheus)
                metrics.append(SystemMetrics(
                    timestamp=current_time,
                    cpu_usage=np.random.uniform(20, 80),
                    memory_usage=np.random.uniform(40, 90),
                    network_latency=np.random.uniform(1, 10),
                    kafka_lag={'market-data': np.random.randint(0, 100)},
                    active_connections=np.random.randint(50, 200),
                    error_rate=np.random.uniform(0, 0.05),
                    response_time_p95=np.random.uniform(10, 100)
                ))
                
                current_time += timedelta(minutes=5)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return []


class LogCollector(BaseDataCollector):
    """Collector for system and application logs."""
    
    def __init__(self, log_directory: str = "logs"):
        super().__init__("logs")
        self.log_directory = log_directory
    
    async def collect_error_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        log_level: str = "ERROR"
    ) -> List[Dict[str, Any]]:
        """Collect error logs from the specified time period."""
        try:
            import os
            import glob
            
            logs = []
            
            # Find log files in the directory
            log_pattern = os.path.join(self.log_directory, "*.log")
            log_files = glob.glob(log_pattern)
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if log_level.upper() in line.upper():
                                # Parse log line (simplified)
                                logs.append({
                                    'timestamp': datetime.now(),  # Would parse from log
                                    'level': log_level,
                                    'message': line.strip(),
                                    'source': os.path.basename(log_file)
                                })
                except Exception as e:
                    self.logger.warning(f"Error reading log file {log_file}: {e}")
            
            # Filter by time range
            filtered_logs = [
                log for log in logs
                if start_time <= log['timestamp'] <= end_time
            ]
            
            return filtered_logs
            
        except Exception as e:
            self.logger.error(f"Error collecting logs: {e}")
            return []
    
    async def collect_kafka_logs(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Collect Kafka-related logs."""
        # This would parse Kafka logs for connection issues, lag, etc.
        return []


class DataCollectorOrchestrator:
    """Orchestrates data collection from multiple sources."""
    
    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client
        self.logger = get_logger("data_collector_orchestrator")
        
        # Initialize collectors
        self.pnl_collector = PnLDataCollector(db_client)
        self.market_collector = MarketDataCollector()
        self.system_collector = SystemMetricsCollector()
        self.log_collector = LogCollector()
    
    async def collect_all_data(
        self,
        start_time: datetime,
        end_time: datetime,
        data_sources: List[DataSource]
    ) -> Dict[str, Any]:
        """Collect data from all specified sources."""
        collected_data = {}
        
        try:
            async with self.market_collector:
                # Collect data based on requested sources
                if DataSource.PNL_DATA in data_sources:
                    self.logger.info("Collecting PnL data...")
                    collected_data['pnl_data'] = await self.pnl_collector.collect_pnl_data(
                        start_time, end_time
                    )
                    collected_data['trading_metrics'] = await self.pnl_collector.collect_trading_metrics(
                        start_time, end_time
                    )
                
                if DataSource.MARKET_DATA in data_sources:
                    self.logger.info("Collecting market data...")
                    collected_data['market_context'] = await self.market_collector.collect_market_context(
                        end_time
                    )
                    
                    # Collect price data for major symbols
                    major_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                    price_data = {}
                    
                    for symbol in major_symbols:
                        price_data[symbol] = await self.market_collector.collect_price_data(
                            symbol, start_time, end_time
                        )
                    
                    collected_data['price_data'] = price_data
                
                if DataSource.SYSTEM_LOGS in data_sources:
                    self.logger.info("Collecting system metrics...")
                    collected_data['system_metrics'] = await self.system_collector.collect_system_metrics(
                        start_time, end_time
                    )
                    collected_data['error_logs'] = await self.log_collector.collect_error_logs(
                        start_time, end_time
                    )
                
                if DataSource.FUNDING_RATES in data_sources:
                    # Funding rates are included in market context
                    pass
            
            self.logger.info(f"Data collection completed. Sources: {[s.value for s in data_sources]}")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error in data collection orchestration: {e}")
            return collected_data
    
    async def get_data_summary(self, collected_data: Dict[str, Any]) -> str:
        """Generate a summary of collected data for LLM consumption."""
        summary_parts = []
        
        # PnL data summary
        if 'pnl_data' in collected_data:
            pnl_data = collected_data['pnl_data']
            if pnl_data.values:
                stats = pnl_data.get_statistics()
                summary_parts.append(f"PnL Data: {len(pnl_data.values)} points, "
                                   f"Total PnL: ${stats['mean']:.2f}, "
                                   f"Volatility: ${stats['std']:.2f}")
        
        # Market context summary
        if 'market_context' in collected_data:
            context = collected_data['market_context']
            summary_parts.append(f"Market Context: BTC ${context.btc_price:.0f}, "
                                f"ETH ${context.eth_price:.0f}, "
                                f"Volatility: {context.market_volatility:.2f}%")
        
        # System metrics summary
        if 'system_metrics' in collected_data:
            metrics = collected_data['system_metrics']
            if metrics:
                avg_cpu = np.mean([m.cpu_usage for m in metrics])
                avg_memory = np.mean([m.memory_usage for m in metrics])
                summary_parts.append(f"System Metrics: {len(metrics)} points, "
                                    f"Avg CPU: {avg_cpu:.1f}%, "
                                    f"Avg Memory: {avg_memory:.1f}%")
        
        # Error logs summary
        if 'error_logs' in collected_data:
            logs = collected_data['error_logs']
            summary_parts.append(f"Error Logs: {len(logs)} errors found")
        
        return "\n".join(summary_parts) if summary_parts else "No data collected"
