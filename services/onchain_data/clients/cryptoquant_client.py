"""
CryptoQuant API client for institutional flow and derivatives data.
https://docs.cryptoquant.com/
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
    OnChainMetrics, ExchangeFlowMetrics
)


class CryptoQuantAPIError(Exception):
    """CryptoQuant API specific errors."""
    pass


class CryptoQuantRateLimitError(CryptoQuantAPIError):
    """Rate limit exceeded error."""
    pass


class CryptoQuantClient:
    """
    Async client for CryptoQuant API.
    
    Specializes in:
    - Exchange-specific flows
    - Institutional trading data
    - Derivatives metrics
    - Miner behavior
    """
    
    BASE_URL = "https://api.cryptoquant.com"
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_minute: int = 300
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
        
        # CryptoQuant specific endpoint mappings
        self.flow_endpoints = {
            "binance": "exchange-flows/binance",
            "coinbase": "exchange-flows/coinbase", 
            "kraken": "exchange-flows/kraken",
            "bitfinex": "exchange-flows/bitfinex",
            "okex": "exchange-flows/okex",
            "huobi": "exchange-flows/huobi"
        }
        
        self.derivatives_endpoints = {
            "funding_rates": "derivatives/funding-rates",
            "open_interest": "derivatives/open-interest",
            "options_volume": "derivatives/options-volume",
            "futures_volume": "derivatives/futures-volume"
        }
        
        self.miner_endpoints = {
            "miner_flows": "miners/flows",
            "miner_reserves": "miners/reserves",
            "pool_flows": "miners/pool-flows"
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
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "HFT-CryptoBot/1.0",
                    "Content-Type": "application/json"
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
                raise CryptoQuantRateLimitError(f"Rate limit exceeded. Wait {sleep_time:.1f} seconds")
        
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
            CryptoQuantAPIError: On API errors
            CryptoQuantRateLimitError: On rate limit exceeded
        """
        await self._ensure_session()
        
        url = f"{self.BASE_URL}/v1/{endpoint}"
        params = params or {}
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check rate limits
                self._check_rate_limit()
                
                self.logger.debug(f"Making request to {url} with params: {params}")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Successful response: {len(data.get('result', []))} items")
                        return data
                    
                    elif response.status == 429:
                        # Rate limit exceeded
                        error_text = await response.text()
                        self.logger.warning(f"Rate limit exceeded: {error_text}")
                        raise CryptoQuantRateLimitError("Rate limit exceeded")
                    
                    elif response.status == 401:
                        error_text = await response.text()
                        self.logger.error(f"Authentication failed: {error_text}")
                        raise CryptoQuantAPIError(f"Authentication failed: {error_text}")
                    
                    elif response.status == 400:
                        error_text = await response.text()
                        self.logger.error(f"Bad request: {error_text}")
                        raise CryptoQuantAPIError(f"Bad request: {error_text}")
                    
                    else:
                        error_text = await response.text()
                        self.logger.warning(f"Request failed with status {response.status}: {error_text}")
                        
                        if attempt == self.max_retries:
                            raise CryptoQuantAPIError(f"Request failed: {response.status} {error_text}")
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    raise CryptoQuantAPIError(f"Request failed after {self.max_retries} retries: {e}")
            
            except CryptoQuantRateLimitError:
                # Don't retry rate limit errors immediately
                raise
            
            # Wait before retrying
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def get_exchange_flows_detailed(
        self,
        exchange: str,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        window: str = "1h"
    ) -> List[ExchangeFlowMetrics]:
        """
        Get detailed exchange flows for a specific exchange.
        
        Args:
            exchange: Exchange name (binance, coinbase, etc.)
            symbol: Asset symbol
            since: Start timestamp
            until: End timestamp
            window: Time window (1h, 4h, 1d)
            
        Returns:
            List of exchange flow metrics
        """
        if exchange.lower() not in self.flow_endpoints:
            raise CryptoQuantAPIError(f"Unsupported exchange: {exchange}")
        
        endpoint = self.flow_endpoints[exchange.lower()]
        
        params = {
            "symbol": symbol.upper(),
            "window": window
        }
        
        if since:
            params["from"] = int(since.timestamp())
        if until:
            params["to"] = int(until.timestamp())
        
        data = await self._make_request(endpoint, params)
        
        # Convert response to flow metrics
        flow_metrics = []
        
        if "result" in data and isinstance(data["result"], list):
            for item in data["result"]:
                timestamp = datetime.fromtimestamp(item.get("timestamp", 0))
                
                # CryptoQuant provides inflow/outflow in native units
                inflow = item.get("inflow", 0)
                outflow = item.get("outflow", 0)
                net_flow = outflow - inflow
                
                flow_ratio = inflow / outflow if outflow != 0 else 0
                
                # Create exchange-specific flow metrics
                flow_metric = ExchangeFlowMetrics(
                    symbol=symbol.upper(),
                    timestamp=timestamp
                )
                
                # Set exchange-specific fields
                if exchange.lower() == "binance":
                    flow_metric.binance_inflow_1h = inflow
                    flow_metric.binance_outflow_1h = outflow
                    flow_metric.binance_net_flow_1h = net_flow
                elif exchange.lower() == "coinbase":
                    flow_metric.coinbase_inflow_1h = inflow
                    flow_metric.coinbase_outflow_1h = outflow
                    flow_metric.coinbase_net_flow_1h = net_flow
                
                # Set aggregate fields
                flow_metric.total_exchange_inflow_1h = inflow
                flow_metric.total_exchange_outflow_1h = outflow
                flow_metric.total_exchange_net_flow_1h = net_flow
                flow_metric.inflow_outflow_ratio = flow_ratio
                
                flow_metrics.append(flow_metric)
        
        self.logger.info(f"Fetched {len(flow_metrics)} flow metrics for {exchange} ({symbol})")
        return flow_metrics
    
    async def get_all_exchange_flows(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        exchanges: Optional[List[str]] = None
    ) -> List[ExchangeFlowMetrics]:
        """
        Get aggregated flows from multiple exchanges.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp  
            until: End timestamp
            exchanges: List of exchanges to include
            
        Returns:
            List of aggregated flow metrics
        """
        if exchanges is None:
            exchanges = ["binance", "coinbase", "kraken"]
        
        # Fetch flows from all specified exchanges concurrently
        tasks = [
            self.get_exchange_flows_detailed(exchange, symbol, since, until)
            for exchange in exchanges
        ]
        
        try:
            all_flows = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error fetching exchange flows: {e}")
            raise
        
        # Aggregate flows by timestamp
        aggregated_flows = {}
        
        for i, flows in enumerate(all_flows):
            if isinstance(flows, Exception):
                self.logger.warning(f"Failed to fetch flows for {exchanges[i]}: {flows}")
                continue
                
            for flow in flows:
                timestamp = flow.timestamp
                
                if timestamp not in aggregated_flows:
                    aggregated_flows[timestamp] = ExchangeFlowMetrics(
                        symbol=symbol.upper(),
                        timestamp=timestamp,
                        total_exchange_inflow_1h=0,
                        total_exchange_outflow_1h=0,
                        total_exchange_net_flow_1h=0
                    )
                
                agg_flow = aggregated_flows[timestamp]
                
                # Aggregate the flows
                if flow.total_exchange_inflow_1h:
                    agg_flow.total_exchange_inflow_1h += flow.total_exchange_inflow_1h
                if flow.total_exchange_outflow_1h:
                    agg_flow.total_exchange_outflow_1h += flow.total_exchange_outflow_1h
                if flow.total_exchange_net_flow_1h:
                    agg_flow.total_exchange_net_flow_1h += flow.total_exchange_net_flow_1h
                
                # Update exchange-specific fields
                if flow.binance_inflow_1h is not None:
                    agg_flow.binance_inflow_1h = flow.binance_inflow_1h
                    agg_flow.binance_outflow_1h = flow.binance_outflow_1h
                    agg_flow.binance_net_flow_1h = flow.binance_net_flow_1h
                
                if flow.coinbase_inflow_1h is not None:
                    agg_flow.coinbase_inflow_1h = flow.coinbase_inflow_1h
                    agg_flow.coinbase_outflow_1h = flow.coinbase_outflow_1h
                    agg_flow.coinbase_net_flow_1h = flow.coinbase_net_flow_1h
        
        # Calculate final ratios and anomaly scores
        final_flows = []
        for flow in aggregated_flows.values():
            if flow.total_exchange_outflow_1h and flow.total_exchange_outflow_1h != 0:
                flow.inflow_outflow_ratio = (
                    flow.total_exchange_inflow_1h / flow.total_exchange_outflow_1h
                )
            
            final_flows.append(flow)
        
        # Sort by timestamp
        final_flows.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Aggregated {len(final_flows)} flow metrics from {len(exchanges)} exchanges")
        return final_flows
    
    async def get_funding_rates(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[OnChainDataPoint]:
        """
        Get funding rates data.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp
            until: End timestamp
            
        Returns:
            List of funding rate data points
        """
        endpoint = self.derivatives_endpoints["funding_rates"]
        
        params = {
            "symbol": f"{symbol.upper()}-PERP"
        }
        
        if since:
            params["from"] = int(since.timestamp())
        if until:
            params["to"] = int(until.timestamp())
        
        data = await self._make_request(endpoint, params)
        
        # Convert to data points
        data_points = []
        
        if "result" in data and isinstance(data["result"], list):
            for item in data["result"]:
                timestamp = datetime.fromtimestamp(item.get("timestamp", 0))
                funding_rate = item.get("funding_rate", 0)
                
                data_points.append(OnChainDataPoint(
                    metric_type=OnChainMetricType.FUNDING_RATES,
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    value=funding_rate,
                    source=OnChainDataSource.CRYPTOQUANT,
                    metadata={
                        "exchange": item.get("exchange", "unknown"),
                        "contract_type": "perpetual"
                    }
                ))
        
        return data_points
    
    async def get_open_interest(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[OnChainDataPoint]:
        """
        Get open interest data.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp
            until: End timestamp
            
        Returns:
            List of open interest data points
        """
        endpoint = self.derivatives_endpoints["open_interest"]
        
        params = {
            "symbol": f"{symbol.upper()}-PERP"
        }
        
        if since:
            params["from"] = int(since.timestamp())
        if until:
            params["to"] = int(until.timestamp())
        
        data = await self._make_request(endpoint, params)
        
        # Convert to data points
        data_points = []
        
        if "result" in data and isinstance(data["result"], list):
            for item in data["result"]:
                timestamp = datetime.fromtimestamp(item.get("timestamp", 0))
                open_interest = item.get("open_interest", 0)
                
                data_points.append(OnChainDataPoint(
                    metric_type=OnChainMetricType.OPEN_INTEREST,
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    value=open_interest,
                    source=OnChainDataSource.CRYPTOQUANT,
                    metadata={
                        "exchange": item.get("exchange", "unknown"),
                        "contract_type": "perpetual",
                        "currency": "USD"
                    }
                ))
        
        return data_points
    
    async def get_miner_flows(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[OnChainDataPoint]:
        """
        Get miner flow data (Bitcoin only).
        
        Args:
            symbol: Asset symbol (typically BTC)
            since: Start timestamp
            until: End timestamp
            
        Returns:
            List of miner flow data points
        """
        if symbol.upper() != "BTC":
            self.logger.warning(f"Miner flows only available for BTC, not {symbol}")
            return []
        
        endpoint = self.miner_endpoints["miner_flows"]
        
        params = {
            "symbol": symbol.upper()
        }
        
        if since:
            params["from"] = int(since.timestamp())
        if until:
            params["to"] = int(until.timestamp())
        
        data = await self._make_request(endpoint, params)
        
        # Convert to data points
        data_points = []
        
        if "result" in data and isinstance(data["result"], list):
            for item in data["result"]:
                timestamp = datetime.fromtimestamp(item.get("timestamp", 0))
                miner_outflow = item.get("miner_outflow", 0)
                
                data_points.append(OnChainDataPoint(
                    metric_type=OnChainMetricType.EXCHANGE_OUTFLOW,  # Miner selling
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    value=miner_outflow,
                    source=OnChainDataSource.CRYPTOQUANT,
                    metadata={
                        "flow_type": "miner_to_exchange",
                        "metric_category": "miner_behavior"
                    }
                ))
        
        return data_points
    
    async def get_comprehensive_institutional_data(
        self,
        symbol: str = "BTC",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> Dict[str, List[Any]]:
        """
        Get comprehensive institutional and derivatives data.
        
        Args:
            symbol: Asset symbol
            since: Start timestamp
            until: End timestamp
            
        Returns:
            Dictionary containing various data types
        """
        # Default time range if not provided
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)
        if until is None:
            until = datetime.utcnow()
        
        self.logger.info(f"Fetching comprehensive institutional data for {symbol}")
        
        # Fetch all data types concurrently
        tasks = [
            self.get_all_exchange_flows(symbol, since, until),
            self.get_funding_rates(symbol, since, until),
            self.get_open_interest(symbol, since, until)
        ]
        
        # Add miner data for Bitcoin
        if symbol.upper() == "BTC":
            tasks.append(self.get_miner_flows(symbol, since, until))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error fetching institutional data: {e}")
            raise
        
        # Package results
        data = {
            "exchange_flows": results[0] if not isinstance(results[0], Exception) else [],
            "funding_rates": results[1] if not isinstance(results[1], Exception) else [],
            "open_interest": results[2] if not isinstance(results[2], Exception) else [],
        }
        
        if symbol.upper() == "BTC" and len(results) > 3:
            data["miner_flows"] = results[3] if not isinstance(results[3], Exception) else []
        
        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                data_type = list(data.keys())[i]
                self.logger.warning(f"Failed to fetch {data_type}: {result}")
        
        total_points = sum(len(v) for v in data.values() if isinstance(v, list))
        self.logger.info(f"Fetched {total_points} total data points across all categories")
        
        return data
