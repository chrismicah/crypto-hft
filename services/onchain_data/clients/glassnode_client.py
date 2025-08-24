"""
Glassnode API client for on-chain metrics.
https://docs.glassnode.com/api
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
import time
import json

from ..models import (
    OnChainDataPoint, OnChainMetricType, OnChainDataSource,
    OnChainMetrics, ExchangeFlowMetrics, WhaleActivity, NetworkHealthMetrics
)


class GlassnodeAPIError(Exception):
    """Glassnode API specific errors."""
    pass


class GlassnodeRateLimitError(GlassnodeAPIError):
    """Rate limit exceeded error."""
    pass


class GlassnodeClient:
    """
    Async client for Glassnode API.
    
    Provides access to on-chain metrics including:
    - Exchange flows
    - Network activity
    - Whale transactions
    - Market indicators
    """
    
    BASE_URL = "https://api.glassnode.com"
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_minute: int = 100
    ):
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_per_minute = rate_limit_per_minute
        
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.request_times: List[float] = []
        
        # Metric mappings
        self.metric_endpoints = {
            OnChainMetricType.EXCHANGE_INFLOW: "distribution/exchange_inflow",
            OnChainMetricType.EXCHANGE_OUTFLOW: "distribution/exchange_outflow", 
            OnChainMetricType.EXCHANGE_NET_FLOW: "distribution/exchange_net_flow",
            OnChainMetricType.ACTIVE_ADDRESSES: "addresses/active_count",
            OnChainMetricType.TRANSACTION_COUNT: "transactions/count",
            OnChainMetricType.TRANSACTION_VOLUME: "transactions/transfers_volume_sum",
            OnChainMetricType.NETWORK_FEE: "fees/volume_sum",
            OnChainMetricType.HASH_RATE: "mining/hash_rate_mean",
            OnChainMetricType.WHALE_TRANSACTIONS: "transactions/transfers_volume_large_sum",
            OnChainMetricType.LARGE_TRANSACTIONS: "transactions/transfers_volume_large_count",
            OnChainMetricType.LONG_TERM_HOLDERS: "supply/lth_sum",
            OnChainMetricType.SHORT_TERM_HOLDERS: "supply/sth_sum",
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    "X-API-KEY": self.api_key,
                    "User-Agent": "HFT-CryptoBot/1.0"
                }
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _check_rate_limit(self):
        """Check if we're within rate limits."""
        now = time.time()
        
        # Remove requests older than 1 minute
        cutoff = now - 60
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Check if we can make another request
        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                raise GlassnodeRateLimitError(f"Rate limit exceeded. Wait {sleep_time:.1f} seconds")
        
        self.request_times.append(now)
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with retry logic.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            GlassnodeAPIError: On API errors
            GlassnodeRateLimitError: On rate limit exceeded
        """
        await self._ensure_session()
        
        url = f"{self.BASE_URL}/v1/metrics/{endpoint}"
        params = params or {}
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check rate limits
                self._check_rate_limit()
                
                self.logger.debug(f"Making request to {url} with params: {params}")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Successful response: {len(data) if isinstance(data, list) else 'single'} items")
                        return data
                    
                    elif response.status == 429:
                        # Rate limit exceeded
                        error_text = await response.text()
                        self.logger.warning(f"Rate limit exceeded: {error_text}")
                        raise GlassnodeRateLimitError("Rate limit exceeded")
                    
                    elif response.status == 401:
                        error_text = await response.text()
                        self.logger.error(f"Authentication failed: {error_text}")
                        raise GlassnodeAPIError(f"Authentication failed: {error_text}")
                    
                    elif response.status == 400:
                        error_text = await response.text()
                        self.logger.error(f"Bad request: {error_text}")
                        raise GlassnodeAPIError(f"Bad request: {error_text}")
                    
                    else:
                        error_text = await response.text()
                        self.logger.warning(f"Request failed with status {response.status}: {error_text}")
                        
                        if attempt == self.max_retries:
                            raise GlassnodeAPIError(f"Request failed: {response.status} {error_text}")
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    raise GlassnodeAPIError(f"Request failed after {self.max_retries} retries: {e}")
            
            except GlassnodeRateLimitError:
                # Don't retry rate limit errors immediately
                raise
            
            # Wait before retrying
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def get_metric(
        self,
        metric_type: OnChainMetricType,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "1h",
        currency: str = "USD"
    ) -> List[OnChainDataPoint]:
        """
        Get a specific on-chain metric.
        
        Args:
            metric_type: Type of metric to fetch
            symbol: Asset symbol (BTC, ETH, etc.)
            since: Start timestamp
            until: End timestamp  
            interval: Data interval (1h, 1d, etc.)
            currency: Currency for value metrics
            
        Returns:
            List of data points
        """
        if metric_type not in self.metric_endpoints:
            raise GlassnodeAPIError(f"Unsupported metric type: {metric_type}")
        
        endpoint = self.metric_endpoints[metric_type]
        
        params = {
            "a": symbol.upper(),
            "i": interval,
            "c": currency.upper()
        }
        
        if since:
            params["s"] = int(since.timestamp())
        if until:
            params["u"] = int(until.timestamp())
        
        data = await self._make_request(endpoint, params)
        
        # Convert response to data points
        data_points = []
        
        if isinstance(data, list):
            for item in data:
                if "t" in item and "v" in item:
                    timestamp = datetime.fromtimestamp(item["t"])
                    value = item["v"]
                    
                    if value is not None:
                        data_points.append(OnChainDataPoint(
                            metric_type=metric_type,
                            symbol=symbol.upper(),
                            timestamp=timestamp,
                            value=float(value),
                            source=OnChainDataSource.GLASSNODE,
                            metadata={"interval": interval, "currency": currency}
                        ))
        
        self.logger.info(f"Fetched {len(data_points)} data points for {metric_type} ({symbol})")
        return data_points
    
    async def get_exchange_flows(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        interval: str = "1h"
    ) -> List[ExchangeFlowMetrics]:
        """
        Get exchange flow metrics.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp
            until: End timestamp
            interval: Data interval
            
        Returns:
            List of exchange flow metrics
        """
        # Fetch inflow and outflow data
        inflows = await self.get_metric(
            OnChainMetricType.EXCHANGE_INFLOW,
            symbol=symbol,
            since=since,
            until=until,
            interval=interval
        )
        
        outflows = await self.get_metric(
            OnChainMetricType.EXCHANGE_OUTFLOW,
            symbol=symbol,
            since=since,
            until=until,
            interval=interval
        )
        
        # Combine into flow metrics
        flow_metrics = []
        
        # Create timestamp-based lookup for outflows
        outflow_dict = {dp.timestamp: dp.value for dp in outflows}
        
        for inflow in inflows:
            outflow_value = outflow_dict.get(inflow.timestamp)
            
            if outflow_value is not None:
                net_flow = outflow_value - inflow.value
                inflow_outflow_ratio = (
                    inflow.value / outflow_value if outflow_value != 0 else 0
                )
                
                flow_metrics.append(ExchangeFlowMetrics(
                    symbol=symbol.upper(),
                    timestamp=inflow.timestamp,
                    total_exchange_inflow_1h=inflow.value,
                    total_exchange_outflow_1h=outflow_value,
                    total_exchange_net_flow_1h=net_flow,
                    inflow_outflow_ratio=inflow_outflow_ratio
                ))
        
        return flow_metrics
    
    async def get_whale_activity(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        threshold: float = 1000000.0  # $1M USD
    ) -> List[WhaleActivity]:
        """
        Get whale activity metrics.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp
            until: End timestamp
            threshold: Minimum transaction size in USD
            
        Returns:
            List of whale activity metrics
        """
        # Fetch large transaction data
        large_txs = await self.get_metric(
            OnChainMetricType.LARGE_TRANSACTIONS,
            symbol=symbol,
            since=since,
            until=until,
            interval="1h"
        )
        
        whale_volume = await self.get_metric(
            OnChainMetricType.WHALE_TRANSACTIONS,
            symbol=symbol,
            since=since,
            until=until,
            interval="1h"
        )
        
        # Combine into whale activity
        whale_activities = []
        
        # Create timestamp-based lookup
        volume_dict = {dp.timestamp: dp.value for dp in whale_volume}
        
        for tx_data in large_txs:
            volume_value = volume_dict.get(tx_data.timestamp, 0)
            
            whale_activities.append(WhaleActivity(
                symbol=symbol.upper(),
                timestamp=tx_data.timestamp,
                large_tx_threshold=threshold,
                whale_tx_threshold=threshold * 10,  # 10x for whale threshold
                large_transactions_1h=int(tx_data.value) if tx_data.value else 0,
                large_transaction_volume_1h=volume_value
            ))
        
        return whale_activities
    
    async def get_network_health(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[NetworkHealthMetrics]:
        """
        Get network health metrics.
        
        Args:
            symbol: Asset symbol  
            since: Start timestamp
            until: End timestamp
            
        Returns:
            List of network health metrics
        """
        # Fetch multiple network metrics
        active_addresses = await self.get_metric(
            OnChainMetricType.ACTIVE_ADDRESSES,
            symbol=symbol,
            since=since,
            until=until,
            interval="1d"
        )
        
        transaction_count = await self.get_metric(
            OnChainMetricType.TRANSACTION_COUNT,
            symbol=symbol,
            since=since,
            until=until,
            interval="1d"
        )
        
        hash_rate = await self.get_metric(
            OnChainMetricType.HASH_RATE,
            symbol=symbol,
            since=since,
            until=until,
            interval="1d"
        ) if symbol.upper() == "BTC" else []
        
        network_fees = await self.get_metric(
            OnChainMetricType.NETWORK_FEE,
            symbol=symbol,
            since=since,
            until=until,
            interval="1d"
        )
        
        # Combine into network health metrics
        health_metrics = []
        
        # Create timestamp-based lookups
        tx_dict = {dp.timestamp: dp.value for dp in transaction_count}
        hash_dict = {dp.timestamp: dp.value for dp in hash_rate}
        fee_dict = {dp.timestamp: dp.value for dp in network_fees}
        
        for addr_data in active_addresses:
            tx_count = tx_dict.get(addr_data.timestamp, 0)
            hash_rate_val = hash_dict.get(addr_data.timestamp)
            fee_volume = fee_dict.get(addr_data.timestamp, 0)
            
            # Calculate basic health score
            health_score = min(1.0, addr_data.value / 1000000) * 0.5  # Active addresses component
            if tx_count > 0:
                health_score += min(1.0, tx_count / 300000) * 0.3  # Transaction count component
            if hash_rate_val:
                health_score += 0.2  # Hash rate stability component
            
            health_metrics.append(NetworkHealthMetrics(
                symbol=symbol.upper(),
                timestamp=addr_data.timestamp,
                active_addresses_24h=int(addr_data.value) if addr_data.value else 0,
                tx_count_24h=int(tx_count) if tx_count else 0,
                hash_rate=hash_rate_val,
                network_health_score=min(1.0, health_score)
            ))
        
        return health_metrics
    
    async def get_comprehensive_metrics(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[OnChainMetrics]:
        """
        Get comprehensive on-chain metrics for a symbol.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp  
            until: End timestamp
            
        Returns:
            List of comprehensive metrics
        """
        # Default time range if not provided
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)
        if until is None:
            until = datetime.utcnow()
        
        self.logger.info(f"Fetching comprehensive metrics for {symbol} from {since} to {until}")
        
        # Fetch all relevant metrics concurrently
        tasks = [
            self.get_exchange_flows(symbol, since, until),
            self.get_whale_activity(symbol, since, until),
            self.get_network_health(symbol, since, until)
        ]
        
        try:
            exchange_flows, whale_activities, network_health = await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error fetching comprehensive metrics: {e}")
            raise
        
        # Combine into comprehensive metrics
        comprehensive_metrics = []
        
        # Create timestamp-based lookups
        flows_dict = {fm.timestamp: fm for fm in exchange_flows}
        whale_dict = {wa.timestamp: wa for wa in whale_activities}
        health_dict = {nh.timestamp: nh for nh in network_health}
        
        # Get all unique timestamps
        all_timestamps = set()
        all_timestamps.update(flows_dict.keys())
        all_timestamps.update(whale_dict.keys())
        all_timestamps.update(health_dict.keys())
        
        for timestamp in sorted(all_timestamps):
            flow_data = flows_dict.get(timestamp)
            whale_data = whale_dict.get(timestamp)
            health_data = health_dict.get(timestamp)
            
            metrics = OnChainMetrics(
                symbol=symbol.upper(),
                timestamp=timestamp
            )
            
            # Exchange flow metrics
            if flow_data:
                metrics.exchange_inflow_btc = flow_data.total_exchange_inflow_1h
                metrics.exchange_outflow_btc = flow_data.total_exchange_outflow_1h
                metrics.exchange_net_flow_btc = flow_data.total_exchange_net_flow_1h
            
            # Whale activity metrics
            if whale_data:
                metrics.whale_transaction_count = whale_data.large_transactions_1h
                metrics.large_transaction_volume = whale_data.large_transaction_volume_1h
            
            # Network health metrics
            if health_data:
                metrics.active_addresses_24h = health_data.active_addresses_24h
                metrics.transaction_count_24h = health_data.tx_count_24h
                metrics.hash_rate_7d_ma = health_data.hash_rate
            
            comprehensive_metrics.append(metrics)
        
        self.logger.info(f"Generated {len(comprehensive_metrics)} comprehensive metric records")
        return comprehensive_metrics
