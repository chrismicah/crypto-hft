"""
Data models for on-chain metrics and signals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import json


class OnChainMetricType(str, Enum):
    """Types of on-chain metrics."""
    # Exchange flows
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    EXCHANGE_NET_FLOW = "exchange_net_flow"
    
    # Network activity
    ACTIVE_ADDRESSES = "active_addresses"
    TRANSACTION_COUNT = "transaction_count"
    TRANSACTION_VOLUME = "transaction_volume"
    NETWORK_FEE = "network_fee"
    HASH_RATE = "hash_rate"
    
    # Wallet analysis
    WHALE_TRANSACTIONS = "whale_transactions"
    LARGE_TRANSACTIONS = "large_transactions"
    HODLER_BEHAVIOR = "hodler_behavior"
    
    # Market structure
    SUPPLY_DISTRIBUTION = "supply_distribution"
    LONG_TERM_HOLDERS = "long_term_holders"
    SHORT_TERM_HOLDERS = "short_term_holders"
    
    # DeFi metrics
    TVL_CHANGE = "tvl_change"
    YIELD_RATES = "yield_rates"
    LIQUIDITY_FLOWS = "liquidity_flows"
    
    # Sentiment
    FUNDING_RATES = "funding_rates"
    OPEN_INTEREST = "open_interest"
    DERIVATIVES_VOLUME = "derivatives_volume"


class OnChainDataSource(str, Enum):
    """Supported on-chain data sources."""
    GLASSNODE = "glassnode"
    CRYPTOQUANT = "cryptoquant"
    CHAINANALYSIS = "chainanalysis"
    DUNE = "dune"
    MESSARI = "messari"
    SANTIMENT = "santiment"
    COINMETRICS = "coinmetrics"


class SignalStrength(str, Enum):
    """Signal strength classifications."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


@dataclass
class OnChainDataPoint:
    """Individual on-chain data point."""
    metric_type: OnChainMetricType
    symbol: str
    timestamp: datetime
    value: Union[float, int, Decimal]
    source: OnChainDataSource
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "value": float(self.value) if isinstance(self.value, Decimal) else self.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnChainDataPoint":
        """Create from dictionary."""
        return cls(
            metric_type=OnChainMetricType(data["metric_type"]),
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=data["value"],
            source=OnChainDataSource(data["source"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )


class OnChainMetrics(BaseModel):
    """Collection of on-chain metrics for a symbol."""
    symbol: str
    timestamp: datetime
    metrics: Dict[OnChainMetricType, float] = Field(default_factory=dict)
    
    # Exchange flows (BTC equivalent)
    exchange_inflow_btc: Optional[float] = None
    exchange_outflow_btc: Optional[float] = None
    exchange_net_flow_btc: Optional[float] = None
    
    # Network activity
    active_addresses_24h: Optional[int] = None
    transaction_count_24h: Optional[int] = None
    transaction_volume_usd: Optional[float] = None
    avg_fee_usd: Optional[float] = None
    hash_rate_7d_ma: Optional[float] = None
    
    # Whale activity
    whale_transaction_count: Optional[int] = None
    large_transaction_volume: Optional[float] = None
    
    # HODLer metrics
    supply_1y_plus: Optional[float] = None  # Supply held for 1+ years
    supply_short_term: Optional[float] = None  # Supply held <155 days
    
    # Market structure
    realized_cap: Optional[float] = None
    mvrv_ratio: Optional[float] = None  # Market Value to Realized Value
    nvt_ratio: Optional[float] = None   # Network Value to Transactions
    
    # DeFi metrics (for ETH primarily)
    total_value_locked: Optional[float] = None
    defi_dominance: Optional[float] = None
    
    class Config:
        use_enum_values = True
        
    def get_metric(self, metric_type: OnChainMetricType) -> Optional[float]:
        """Get specific metric value."""
        return self.metrics.get(metric_type)
    
    def set_metric(self, metric_type: OnChainMetricType, value: float, confidence: float = 1.0):
        """Set metric value."""
        self.metrics[metric_type] = value


class OnChainSignal(BaseModel):
    """On-chain derived trading signal."""
    signal_id: str
    symbol: str
    timestamp: datetime
    signal_type: str
    strength: SignalStrength
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)  # Normalized signal score
    
    # Contributing factors
    primary_metrics: List[OnChainMetricType] = Field(default_factory=list)
    supporting_metrics: List[OnChainMetricType] = Field(default_factory=list)
    
    # Signal details
    time_horizon: str = "1d"  # Signal validity period
    decay_rate: float = 0.1   # How quickly signal decays
    
    # Metadata
    model_version: str = "1.0"
    features_used: List[str] = Field(default_factory=list)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.strength in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH]
    
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.strength in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH]
    
    def get_signal_weight(self) -> float:
        """Get weighted signal score based on confidence."""
        return self.score * self.confidence


class ExchangeFlowMetrics(BaseModel):
    """Exchange flow specific metrics."""
    symbol: str
    timestamp: datetime
    
    # Binance flows
    binance_inflow_1h: Optional[float] = None
    binance_outflow_1h: Optional[float] = None
    binance_net_flow_1h: Optional[float] = None
    
    # Coinbase flows
    coinbase_inflow_1h: Optional[float] = None
    coinbase_outflow_1h: Optional[float] = None
    coinbase_net_flow_1h: Optional[float] = None
    
    # Aggregated flows
    total_exchange_inflow_1h: Optional[float] = None
    total_exchange_outflow_1h: Optional[float] = None
    total_exchange_net_flow_1h: Optional[float] = None
    
    # Flow ratios and anomalies
    inflow_outflow_ratio: Optional[float] = None
    flow_anomaly_score: Optional[float] = None  # 0-1, higher = more anomalous
    
    def calculate_net_flow(self) -> Optional[float]:
        """Calculate total net flow."""
        if self.total_exchange_inflow_1h is not None and self.total_exchange_outflow_1h is not None:
            return self.total_exchange_outflow_1h - self.total_exchange_inflow_1h
        return None
    
    def is_significant_outflow(self, threshold: float = 1000.0) -> bool:
        """Check if there's significant outflow (typically bullish)."""
        net_flow = self.calculate_net_flow()
        return net_flow is not None and net_flow > threshold
    
    def is_significant_inflow(self, threshold: float = -1000.0) -> bool:
        """Check if there's significant inflow (typically bearish)."""
        net_flow = self.calculate_net_flow()
        return net_flow is not None and net_flow < threshold


class WhaleActivity(BaseModel):
    """Whale wallet activity metrics."""
    symbol: str
    timestamp: datetime
    
    # Transaction thresholds (in USD)
    large_tx_threshold: float = 1000000.0  # $1M+
    whale_tx_threshold: float = 10000000.0  # $10M+
    
    # Activity counts
    large_transactions_1h: Optional[int] = None
    whale_transactions_1h: Optional[int] = None
    
    # Volume metrics
    large_transaction_volume_1h: Optional[float] = None
    whale_transaction_volume_1h: Optional[float] = None
    
    # Behavioral analysis
    whale_accumulation_score: Optional[float] = None  # -1 to 1
    whale_distribution_score: Optional[float] = None  # -1 to 1
    
    # Top whale addresses activity
    top10_whale_activity: Optional[Dict[str, float]] = None
    top100_whale_activity: Optional[Dict[str, float]] = None
    
    def get_whale_sentiment(self) -> SignalStrength:
        """Derive whale sentiment from activity."""
        if self.whale_accumulation_score is None:
            return SignalStrength.NEUTRAL
            
        if self.whale_accumulation_score > 0.5:
            return SignalStrength.VERY_BULLISH
        elif self.whale_accumulation_score > 0.2:
            return SignalStrength.BULLISH
        elif self.whale_accumulation_score < -0.5:
            return SignalStrength.VERY_BEARISH
        elif self.whale_accumulation_score < -0.2:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.NEUTRAL


class NetworkHealthMetrics(BaseModel):
    """Network health and activity metrics."""
    symbol: str
    timestamp: datetime
    
    # Core network metrics
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    active_addresses_24h: Optional[int] = None
    new_addresses_24h: Optional[int] = None
    
    # Transaction metrics
    tx_count_24h: Optional[int] = None
    tx_volume_24h: Optional[float] = None
    avg_tx_size: Optional[float] = None
    median_tx_fee: Optional[float] = None
    
    # Network utilization
    mempool_size: Optional[int] = None
    fee_rate_percentiles: Optional[Dict[str, float]] = None  # 25th, 50th, 75th, 95th
    
    # Health scores
    network_health_score: Optional[float] = None  # 0-1
    congestion_score: Optional[float] = None      # 0-1, higher = more congested
    
    def is_network_congested(self, threshold: float = 0.7) -> bool:
        """Check if network is congested."""
        return self.congestion_score is not None and self.congestion_score > threshold
    
    def get_network_sentiment(self) -> SignalStrength:
        """Derive sentiment from network health."""
        if self.network_health_score is None:
            return SignalStrength.NEUTRAL
            
        if self.network_health_score > 0.8:
            return SignalStrength.BULLISH
        elif self.network_health_score > 0.6:
            return SignalStrength.NEUTRAL
        else:
            return SignalStrength.BEARISH


class OnChainFeatureSet(BaseModel):
    """Engineered features for ML models."""
    symbol: str
    timestamp: datetime
    
    # Raw features
    raw_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Engineered features
    exchange_flow_ma_7d: Optional[float] = None
    exchange_flow_ma_30d: Optional[float] = None
    exchange_flow_momentum: Optional[float] = None
    
    whale_activity_ma_7d: Optional[float] = None
    whale_activity_volatility: Optional[float] = None
    
    network_health_trend: Optional[float] = None
    fee_pressure_indicator: Optional[float] = None
    
    # Composite scores
    onchain_bullish_score: Optional[float] = None
    onchain_bearish_score: Optional[float] = None
    onchain_momentum_score: Optional[float] = None
    
    # Cross-asset features
    btc_dominance_factor: Optional[float] = None
    eth_defi_factor: Optional[float] = None
    
    def get_feature_vector(self) -> List[float]:
        """Get feature vector for ML models."""
        features = []
        
        # Add all numeric features
        for field_name, field_info in self.__fields__.items():
            if field_name in ['symbol', 'timestamp', 'raw_metrics']:
                continue
                
            value = getattr(self, field_name)
            if isinstance(value, (int, float)):
                features.append(float(value) if value is not None else 0.0)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for ML models."""
        feature_names = []
        
        for field_name, field_info in self.__fields__.items():
            if field_name in ['symbol', 'timestamp', 'raw_metrics']:
                continue
                
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if isinstance(value, (int, float, type(None))):
                    feature_names.append(field_name)
        
        return feature_names


class OnChainAlert(BaseModel):
    """On-chain based alerts and notifications."""
    alert_id: str
    symbol: str
    timestamp: datetime
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    
    title: str
    description: str
    
    # Alert conditions
    metric_type: OnChainMetricType
    threshold_value: float
    actual_value: float
    
    # Context
    historical_context: Optional[str] = None
    market_impact: Optional[str] = None
    recommended_action: Optional[str] = None
    
    # Metadata
    confidence: float = Field(ge=0.0, le=1.0)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if alert has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def get_severity_score(self) -> int:
        """Get numeric severity score."""
        severity_map = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4
        }
        return severity_map.get(self.severity, 1)
