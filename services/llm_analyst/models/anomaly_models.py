"""
Data models for LLM-based anomaly diagnosis system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DROP = "performance_drop"
    SUDDEN_LOSS = "sudden_loss"
    VOLUME_SPIKE = "volume_spike"
    CORRELATION_BREAK = "correlation_break"
    EXECUTION_DELAY = "execution_delay"
    FUNDING_RATE_ANOMALY = "funding_rate_anomaly"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEM_ERROR = "system_error"
    MARKET_REGIME_CHANGE = "market_regime_change"
    STRATEGY_FAILURE = "strategy_failure"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataSource(str, Enum):
    """Sources of data for analysis."""
    PNL_DATA = "pnl_data"
    MARKET_DATA = "market_data"
    SYSTEM_LOGS = "system_logs"
    EXECUTION_LOGS = "execution_logs"
    RISK_METRICS = "risk_metrics"
    EXTERNAL_NEWS = "external_news"
    FUNDING_RATES = "funding_rates"
    ORDER_BOOK = "order_book"
    NETWORK_METRICS = "network_metrics"


@dataclass
class TimeSeriesData:
    """Time series data container."""
    timestamps: List[datetime]
    values: List[float]
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        })
        if self.labels:
            df['label'] = self.labels
        return df
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics."""
        values = np.array(self.values)
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'skewness': float(pd.Series(values).skew()),
            'kurtosis': float(pd.Series(values).kurtosis())
        }


@dataclass
class MarketContext:
    """Market context information."""
    timestamp: datetime
    btc_price: float
    eth_price: float
    market_volatility: float
    funding_rates: Dict[str, float]
    volume_24h: Dict[str, float]
    fear_greed_index: Optional[float] = None
    major_news: List[str] = field(default_factory=list)
    macro_events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'btc_price': self.btc_price,
            'eth_price': self.eth_price,
            'market_volatility': self.market_volatility,
            'funding_rates': self.funding_rates,
            'volume_24h': self.volume_24h,
            'fear_greed_index': self.fear_greed_index,
            'major_news': self.major_news,
            'macro_events': self.macro_events
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_latency: float
    kafka_lag: Dict[str, int]
    active_connections: int
    error_rate: float
    response_time_p95: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency,
            'kafka_lag': self.kafka_lag,
            'active_connections': self.active_connections,
            'error_rate': self.error_rate,
            'response_time_p95': self.response_time_p95
        }


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    timestamp: datetime
    pnl: float
    cumulative_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int
    active_positions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pnl': self.pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_trade_duration': self.avg_trade_duration,
            'total_trades': self.total_trades,
            'active_positions': self.active_positions
        }


class AnomalyDetection(BaseModel):
    """Detected anomaly information."""
    id: str = Field(..., description="Unique anomaly identifier")
    timestamp: datetime = Field(..., description="When the anomaly was detected")
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: AnomalySeverity = Field(..., description="Severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    
    # Statistical information
    z_score: float = Field(..., description="Statistical z-score of the anomaly")
    p_value: float = Field(..., description="Statistical p-value")
    threshold_exceeded: float = Field(..., description="How much the threshold was exceeded")
    
    # Context
    affected_metrics: List[str] = Field(default_factory=list, description="Metrics affected by the anomaly")
    data_sources: List[DataSource] = Field(default_factory=list, description="Data sources involved")
    
    # Raw data
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw data for analysis")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DiagnosisRequest(BaseModel):
    """Request for LLM diagnosis."""
    anomaly: AnomalyDetection = Field(..., description="Detected anomaly to diagnose")
    
    # Time series data
    pnl_data: Optional[TimeSeriesData] = Field(None, description="PnL time series")
    market_data: Optional[Dict[str, TimeSeriesData]] = Field(None, description="Market data time series")
    
    # Context information
    market_context: Optional[MarketContext] = Field(None, description="Market context")
    system_metrics: Optional[List[SystemMetrics]] = Field(None, description="System metrics")
    trading_metrics: Optional[List[TradingMetrics]] = Field(None, description="Trading metrics")
    
    # Additional context
    recent_changes: List[str] = Field(default_factory=list, description="Recent system changes")
    active_strategies: List[str] = Field(default_factory=list, description="Currently active strategies")
    
    # Analysis parameters
    lookback_hours: int = Field(24, description="Hours to look back for analysis")
    include_external_factors: bool = Field(True, description="Include external market factors")
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class Hypothesis:
    """A hypothesis about the anomaly cause."""
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    contradicting_evidence: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'supporting_evidence': self.supporting_evidence,
            'contradicting_evidence': self.contradicting_evidence,
            'recommended_actions': self.recommended_actions
        }


@dataclass
class DiagnosisResult:
    """Result of LLM diagnosis."""
    anomaly_id: str
    timestamp: datetime
    
    # Primary diagnosis
    primary_hypothesis: Hypothesis
    alternative_hypotheses: List[Hypothesis] = field(default_factory=list)
    
    # Analysis summary
    executive_summary: str = ""
    detailed_analysis: str = ""
    
    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    preventive_measures: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_duration: float = 0.0  # seconds
    llm_model_used: str = ""
    confidence_score: float = 0.0
    
    # Supporting data
    key_correlations: Dict[str, float] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anomaly_id': self.anomaly_id,
            'timestamp': self.timestamp.isoformat(),
            'primary_hypothesis': self.primary_hypothesis.to_dict(),
            'alternative_hypotheses': [h.to_dict() for h in self.alternative_hypotheses],
            'executive_summary': self.executive_summary,
            'detailed_analysis': self.detailed_analysis,
            'immediate_actions': self.immediate_actions,
            'preventive_measures': self.preventive_measures,
            'monitoring_recommendations': self.monitoring_recommendations,
            'analysis_duration': self.analysis_duration,
            'llm_model_used': self.llm_model_used,
            'confidence_score': self.confidence_score,
            'key_correlations': self.key_correlations,
            'statistical_tests': self.statistical_tests
        }


class AnalysisTemplate(BaseModel):
    """Template for structured analysis."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    
    # Analysis steps
    data_requirements: List[DataSource] = Field(..., description="Required data sources")
    analysis_steps: List[str] = Field(..., description="Analysis steps to perform")
    
    # Prompts
    system_prompt: str = Field(..., description="System prompt for LLM")
    analysis_prompt: str = Field(..., description="Analysis prompt template")
    
    # Parameters
    min_data_points: int = Field(10, description="Minimum data points required")
    lookback_period: int = Field(24, description="Default lookback period in hours")
    
    class Config:
        use_enum_values = True


@dataclass
class AnalysisContext:
    """Context for analysis execution."""
    template: AnalysisTemplate
    request: DiagnosisRequest
    
    # Processed data
    processed_data: Dict[str, Any] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # LLM interaction
    llm_messages: List[Dict[str, str]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add message to LLM conversation."""
        self.llm_messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })


class AlertConfiguration(BaseModel):
    """Configuration for anomaly alerts."""
    anomaly_types: List[AnomalyType] = Field(..., description="Anomaly types to alert on")
    severity_threshold: AnomalySeverity = Field(AnomalySeverity.MEDIUM, description="Minimum severity")
    
    # Notification settings
    email_recipients: List[str] = Field(default_factory=list, description="Email recipients")
    slack_webhook: Optional[str] = Field(None, description="Slack webhook URL")
    
    # Throttling
    max_alerts_per_hour: int = Field(10, description="Maximum alerts per hour")
    cooldown_minutes: int = Field(30, description="Cooldown between similar alerts")
    
    class Config:
        use_enum_values = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for the diagnosis system."""
    total_anomalies_detected: int = 0
    total_diagnoses_completed: int = 0
    avg_diagnosis_time: float = 0.0
    accuracy_score: float = 0.0  # Based on feedback
    
    # LLM usage
    total_tokens_used: int = 0
    avg_tokens_per_diagnosis: float = 0.0
    llm_cost_usd: float = 0.0
    
    # Response times
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_anomalies_detected': self.total_anomalies_detected,
            'total_diagnoses_completed': self.total_diagnoses_completed,
            'avg_diagnosis_time': self.avg_diagnosis_time,
            'accuracy_score': self.accuracy_score,
            'total_tokens_used': self.total_tokens_used,
            'avg_tokens_per_diagnosis': self.avg_tokens_per_diagnosis,
            'llm_cost_usd': self.llm_cost_usd,
            'avg_response_time': self.avg_response_time,
            'p95_response_time': self.p95_response_time
        }


# Utility functions for data processing
def calculate_rolling_statistics(data: TimeSeriesData, window: int = 24) -> Dict[str, List[float]]:
    """Calculate rolling statistics for time series data."""
    df = data.to_dataframe()
    df.set_index('timestamp', inplace=True)
    
    rolling = df['value'].rolling(window=window)
    
    return {
        'rolling_mean': rolling.mean().tolist(),
        'rolling_std': rolling.std().tolist(),
        'rolling_min': rolling.min().tolist(),
        'rolling_max': rolling.max().tolist()
    }


def detect_changepoints(data: TimeSeriesData, method: str = 'pelt') -> List[int]:
    """Detect changepoints in time series data."""
    try:
        import ruptures as rpt
        
        signal = np.array(data.values)
        
        if method == 'pelt':
            algo = rpt.Pelt(model="rbf").fit(signal)
            changepoints = algo.predict(pen=10)
        elif method == 'binseg':
            algo = rpt.Binseg(model="l2").fit(signal)
            changepoints = algo.predict(n_bkps=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return changepoints[:-1]  # Remove last point (end of series)
        
    except ImportError:
        # Fallback to simple threshold-based detection
        values = np.array(data.values)
        diff = np.abs(np.diff(values))
        threshold = np.mean(diff) + 2 * np.std(diff)
        
        changepoints = []
        for i, d in enumerate(diff):
            if d > threshold:
                changepoints.append(i + 1)
        
        return changepoints


def calculate_correlation_matrix(data_dict: Dict[str, TimeSeriesData]) -> pd.DataFrame:
    """Calculate correlation matrix between multiple time series."""
    # Align all time series to common timestamps
    all_dfs = []
    
    for name, ts_data in data_dict.items():
        df = ts_data.to_dataframe()
        df.set_index('timestamp', inplace=True)
        df = df.rename(columns={'value': name})
        all_dfs.append(df)
    
    # Merge all DataFrames
    combined_df = all_dfs[0]
    for df in all_dfs[1:]:
        combined_df = combined_df.join(df, how='outer')
    
    # Calculate correlation matrix
    return combined_df.corr()


def format_data_for_llm(data: Any, max_length: int = 1000) -> str:
    """Format data for LLM consumption with length limits."""
    if isinstance(data, (dict, list)):
        import json
        formatted = json.dumps(data, indent=2, default=str)
    elif isinstance(data, pd.DataFrame):
        formatted = data.to_string()
    elif isinstance(data, TimeSeriesData):
        stats = data.get_statistics()
        formatted = f"Time series with {len(data.values)} points:\n"
        formatted += f"Statistics: {json.dumps(stats, indent=2)}\n"
        formatted += f"Recent values: {data.values[-10:]}"
    else:
        formatted = str(data)
    
    # Truncate if too long
    if len(formatted) > max_length:
        formatted = formatted[:max_length-3] + "..."
    
    return formatted
