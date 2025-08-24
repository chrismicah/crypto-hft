"""
Statistical anomaly detection system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..models.anomaly_models import (
    AnomalyDetection, AnomalyType, AnomalySeverity, TimeSeriesData,
    DataSource, calculate_rolling_statistics, detect_changepoints
)
from ...common.logger import get_logger


@dataclass
class DetectionRule:
    """Configuration for anomaly detection rule."""
    name: str
    anomaly_type: AnomalyType
    metric_name: str
    method: str  # 'zscore', 'iqr', 'isolation_forest', 'changepoint'
    threshold: float
    window_size: int = 24
    min_samples: int = 10
    severity_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'low': self.threshold,
                'medium': self.threshold * 1.5,
                'high': self.threshold * 2.0,
                'critical': self.threshold * 3.0
            }


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using multiple methods."""
    
    def __init__(self):
        self.logger = get_logger("anomaly_detector")
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Default detection rules
        self.detection_rules = self._create_default_rules()
        
        # Detection history for adaptive thresholds
        self.detection_history = []
        self.baseline_stats = {}
    
    def _create_default_rules(self) -> List[DetectionRule]:
        """Create default detection rules."""
        return [
            # PnL anomalies
            DetectionRule(
                name="pnl_sudden_drop",
                anomaly_type=AnomalyType.PERFORMANCE_DROP,
                metric_name="pnl",
                method="zscore",
                threshold=2.0,
                window_size=24
            ),
            DetectionRule(
                name="pnl_large_loss",
                anomaly_type=AnomalyType.SUDDEN_LOSS,
                metric_name="pnl",
                method="threshold",
                threshold=-1000.0,  # $1000 loss threshold
                window_size=1
            ),
            
            # Volume anomalies
            DetectionRule(
                name="volume_spike",
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                metric_name="volume",
                method="zscore",
                threshold=3.0,
                window_size=24
            ),
            
            # System performance anomalies
            DetectionRule(
                name="execution_delay",
                anomaly_type=AnomalyType.EXECUTION_DELAY,
                metric_name="response_time",
                method="zscore",
                threshold=2.5,
                window_size=12
            ),
            DetectionRule(
                name="error_rate_spike",
                anomaly_type=AnomalyType.SYSTEM_ERROR,
                metric_name="error_rate",
                method="zscore",
                threshold=2.0,
                window_size=6
            ),
            
            # Market structure anomalies
            DetectionRule(
                name="correlation_breakdown",
                anomaly_type=AnomalyType.CORRELATION_BREAK,
                metric_name="correlation",
                method="changepoint",
                threshold=0.3,  # 30% correlation change
                window_size=48
            ),
            DetectionRule(
                name="funding_rate_anomaly",
                anomaly_type=AnomalyType.FUNDING_RATE_ANOMALY,
                metric_name="funding_rate",
                method="zscore",
                threshold=2.5,
                window_size=24
            )
        ]
    
    def detect_anomalies(
        self,
        data: Dict[str, TimeSeriesData],
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect anomalies in the provided data."""
        anomalies = []
        
        try:
            self.logger.info(f"Running anomaly detection at {timestamp}")
            
            # Update baseline statistics
            self._update_baseline_stats(data)
            
            # Apply each detection rule
            for rule in self.detection_rules:
                if rule.metric_name in data:
                    rule_anomalies = self._apply_detection_rule(
                        rule, data[rule.metric_name], timestamp
                    )
                    anomalies.extend(rule_anomalies)
            
            # Multi-variate anomaly detection
            multivariate_anomalies = self._detect_multivariate_anomalies(
                data, timestamp
            )
            anomalies.extend(multivariate_anomalies)
            
            # Filter and rank anomalies
            anomalies = self._filter_and_rank_anomalies(anomalies)
            
            self.logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def _apply_detection_rule(
        self,
        rule: DetectionRule,
        data: TimeSeriesData,
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Apply a specific detection rule."""
        anomalies = []
        
        try:
            if len(data.values) < rule.min_samples:
                return anomalies
            
            if rule.method == "zscore":
                anomalies = self._detect_zscore_anomalies(rule, data, timestamp)
            elif rule.method == "iqr":
                anomalies = self._detect_iqr_anomalies(rule, data, timestamp)
            elif rule.method == "threshold":
                anomalies = self._detect_threshold_anomalies(rule, data, timestamp)
            elif rule.method == "changepoint":
                anomalies = self._detect_changepoint_anomalies(rule, data, timestamp)
            elif rule.method == "isolation_forest":
                anomalies = self._detect_isolation_forest_anomalies(rule, data, timestamp)
            
        except Exception as e:
            self.logger.warning(f"Error applying rule {rule.name}: {e}")
        
        return anomalies
    
    def _detect_zscore_anomalies(
        self,
        rule: DetectionRule,
        data: TimeSeriesData,
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect anomalies using Z-score method."""
        anomalies = []
        values = np.array(data.values)
        
        if len(values) < rule.window_size:
            return anomalies
        
        # Calculate rolling statistics
        window_values = values[-rule.window_size:]
        mean = np.mean(window_values[:-1])  # Exclude current value
        std = np.std(window_values[:-1])
        
        if std == 0:
            return anomalies
        
        # Calculate Z-score for current value
        current_value = values[-1]
        z_score = (current_value - mean) / std
        
        # Check if anomaly
        if abs(z_score) >= rule.threshold:
            # Determine severity
            severity = self._determine_severity(abs(z_score), rule.severity_thresholds)
            
            # Calculate p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            anomaly = AnomalyDetection(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                anomaly_type=rule.anomaly_type,
                severity=severity,
                confidence=min(0.99, abs(z_score) / 5.0),  # Scale confidence
                z_score=float(z_score),
                p_value=float(p_value),
                threshold_exceeded=float(abs(z_score) - rule.threshold),
                affected_metrics=[rule.metric_name],
                data_sources=[DataSource.PNL_DATA],  # Would be dynamic
                raw_data={
                    'current_value': float(current_value),
                    'mean': float(mean),
                    'std': float(std),
                    'window_size': rule.window_size,
                    'method': 'zscore'
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_iqr_anomalies(
        self,
        rule: DetectionRule,
        data: TimeSeriesData,
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect anomalies using Interquartile Range method."""
        anomalies = []
        values = np.array(data.values)
        
        if len(values) < rule.window_size:
            return anomalies
        
        window_values = values[-rule.window_size:]
        q1 = np.percentile(window_values, 25)
        q3 = np.percentile(window_values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return anomalies
        
        # Calculate bounds
        lower_bound = q1 - rule.threshold * iqr
        upper_bound = q3 + rule.threshold * iqr
        
        current_value = values[-1]
        
        if current_value < lower_bound or current_value > upper_bound:
            # Calculate how far outside the bounds
            if current_value < lower_bound:
                distance = (lower_bound - current_value) / iqr
            else:
                distance = (current_value - upper_bound) / iqr
            
            severity = self._determine_severity(distance, rule.severity_thresholds)
            
            anomaly = AnomalyDetection(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                anomaly_type=rule.anomaly_type,
                severity=severity,
                confidence=min(0.99, distance / 3.0),
                z_score=float(distance),  # Use distance as proxy
                p_value=0.05,  # Approximate
                threshold_exceeded=float(distance - rule.threshold),
                affected_metrics=[rule.metric_name],
                data_sources=[DataSource.PNL_DATA],
                raw_data={
                    'current_value': float(current_value),
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': 'iqr'
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_threshold_anomalies(
        self,
        rule: DetectionRule,
        data: TimeSeriesData,
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect anomalies using simple threshold method."""
        anomalies = []
        
        if not data.values:
            return anomalies
        
        current_value = data.values[-1]
        
        # Check threshold (assuming negative threshold for losses)
        if current_value <= rule.threshold:
            magnitude = abs(current_value - rule.threshold)
            severity = self._determine_severity(magnitude, rule.severity_thresholds)
            
            anomaly = AnomalyDetection(
                id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                anomaly_type=rule.anomaly_type,
                severity=severity,
                confidence=0.95,  # High confidence for threshold breaches
                z_score=0.0,  # Not applicable
                p_value=0.01,  # Assume significant
                threshold_exceeded=float(magnitude),
                affected_metrics=[rule.metric_name],
                data_sources=[DataSource.PNL_DATA],
                raw_data={
                    'current_value': float(current_value),
                    'threshold': float(rule.threshold),
                    'method': 'threshold'
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_changepoint_anomalies(
        self,
        rule: DetectionRule,
        data: TimeSeriesData,
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect anomalies using changepoint detection."""
        anomalies = []
        
        if len(data.values) < rule.window_size:
            return anomalies
        
        try:
            # Detect changepoints
            changepoints = detect_changepoints(data)
            
            # Check if there's a recent changepoint
            if changepoints:
                recent_changepoint = max(changepoints)
                data_length = len(data.values)
                
                # If changepoint is in the last 10% of data, consider it recent
                if recent_changepoint > data_length * 0.9:
                    # Calculate magnitude of change
                    before_cp = np.mean(data.values[:recent_changepoint])
                    after_cp = np.mean(data.values[recent_changepoint:])
                    
                    if before_cp != 0:
                        change_magnitude = abs(after_cp - before_cp) / abs(before_cp)
                    else:
                        change_magnitude = abs(after_cp - before_cp)
                    
                    if change_magnitude >= rule.threshold:
                        severity = self._determine_severity(
                            change_magnitude, rule.severity_thresholds
                        )
                        
                        anomaly = AnomalyDetection(
                            id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                            timestamp=timestamp,
                            anomaly_type=rule.anomaly_type,
                            severity=severity,
                            confidence=0.8,
                            z_score=float(change_magnitude),
                            p_value=0.05,
                            threshold_exceeded=float(change_magnitude - rule.threshold),
                            affected_metrics=[rule.metric_name],
                            data_sources=[DataSource.MARKET_DATA],
                            raw_data={
                                'changepoint_index': int(recent_changepoint),
                                'before_mean': float(before_cp),
                                'after_mean': float(after_cp),
                                'change_magnitude': float(change_magnitude),
                                'method': 'changepoint'
                            }
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.warning(f"Error in changepoint detection: {e}")
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(
        self,
        rule: DetectionRule,
        data: TimeSeriesData,
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect anomalies using Isolation Forest."""
        anomalies = []
        
        if len(data.values) < rule.min_samples:
            return anomalies
        
        try:
            # Prepare features (value + rolling statistics)
            values = np.array(data.values)
            features = []
            
            for i in range(len(values)):
                if i >= 5:  # Need minimum window for rolling stats
                    window = values[max(0, i-5):i+1]
                    feature_vector = [
                        values[i],  # Current value
                        np.mean(window),  # Rolling mean
                        np.std(window),   # Rolling std
                        np.min(window),   # Rolling min
                        np.max(window)    # Rolling max
                    ]
                    features.append(feature_vector)
            
            if len(features) < rule.min_samples:
                return anomalies
            
            features = np.array(features)
            
            # Fit isolation forest
            self.isolation_forest.fit(features)
            
            # Predict anomalies
            predictions = self.isolation_forest.predict(features)
            scores = self.isolation_forest.decision_function(features)
            
            # Check if current point is anomaly
            if predictions[-1] == -1:  # Anomaly
                anomaly_score = abs(scores[-1])
                severity = self._determine_severity(anomaly_score, rule.severity_thresholds)
                
                anomaly = AnomalyDetection(
                    id=f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    anomaly_type=rule.anomaly_type,
                    severity=severity,
                    confidence=min(0.95, anomaly_score),
                    z_score=float(anomaly_score),
                    p_value=0.1,  # Approximate
                    threshold_exceeded=float(anomaly_score),
                    affected_metrics=[rule.metric_name],
                    data_sources=[DataSource.PNL_DATA],
                    raw_data={
                        'anomaly_score': float(anomaly_score),
                        'method': 'isolation_forest',
                        'n_features': len(features[0])
                    }
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.warning(f"Error in isolation forest detection: {e}")
        
        return anomalies
    
    def _detect_multivariate_anomalies(
        self,
        data: Dict[str, TimeSeriesData],
        timestamp: datetime
    ) -> List[AnomalyDetection]:
        """Detect multivariate anomalies across multiple metrics."""
        anomalies = []
        
        try:
            # Collect aligned data
            aligned_data = self._align_time_series(data)
            
            if aligned_data.shape[0] < 10 or aligned_data.shape[1] < 2:
                return anomalies
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=min(3, aligned_data.shape[1]))
            pca_data = pca.fit_transform(aligned_data)
            
            # Apply isolation forest on PCA components
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(pca_data)
            
            # Check current point
            current_point = pca_data[-1:].reshape(1, -1)
            prediction = iso_forest.predict(current_point)
            score = iso_forest.decision_function(current_point)[0]
            
            if prediction[0] == -1:  # Anomaly detected
                anomaly = AnomalyDetection(
                    id=f"multivariate_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.MARKET_REGIME_CHANGE,
                    severity=AnomalySeverity.MEDIUM,
                    confidence=min(0.9, abs(score)),
                    z_score=float(score),
                    p_value=0.1,
                    threshold_exceeded=float(abs(score)),
                    affected_metrics=list(data.keys()),
                    data_sources=[DataSource.PNL_DATA, DataSource.MARKET_DATA],
                    raw_data={
                        'pca_components': pca.n_components_,
                        'explained_variance': pca.explained_variance_ratio_.tolist(),
                        'anomaly_score': float(score),
                        'method': 'multivariate_isolation_forest'
                    }
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.warning(f"Error in multivariate anomaly detection: {e}")
        
        return anomalies
    
    def _align_time_series(self, data: Dict[str, TimeSeriesData]) -> np.ndarray:
        """Align multiple time series for multivariate analysis."""
        # Find common length (minimum)
        min_length = min(len(ts.values) for ts in data.values())
        
        # Create aligned matrix
        aligned = []
        for name, ts in data.items():
            if ts.values:
                # Take last min_length values
                values = ts.values[-min_length:]
                aligned.append(values)
        
        if not aligned:
            return np.array([])
        
        # Transpose to get (time, features) shape
        aligned_array = np.array(aligned).T
        
        # Handle NaN values
        aligned_array = np.nan_to_num(aligned_array, nan=0.0)
        
        # Standardize
        if aligned_array.shape[0] > 1:
            scaler = StandardScaler()
            aligned_array = scaler.fit_transform(aligned_array)
        
        return aligned_array
    
    def _determine_severity(
        self,
        magnitude: float,
        severity_thresholds: Dict[str, float]
    ) -> AnomalySeverity:
        """Determine severity based on magnitude and thresholds."""
        if magnitude >= severity_thresholds.get('critical', float('inf')):
            return AnomalySeverity.CRITICAL
        elif magnitude >= severity_thresholds.get('high', float('inf')):
            return AnomalySeverity.HIGH
        elif magnitude >= severity_thresholds.get('medium', float('inf')):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _filter_and_rank_anomalies(
        self,
        anomalies: List[AnomalyDetection]
    ) -> List[AnomalyDetection]:
        """Filter and rank anomalies by importance."""
        if not anomalies:
            return anomalies
        
        # Remove duplicates (same type and timestamp)
        unique_anomalies = {}
        for anomaly in anomalies:
            key = f"{anomaly.anomaly_type}_{anomaly.timestamp.strftime('%Y%m%d_%H')}"
            if key not in unique_anomalies or anomaly.confidence > unique_anomalies[key].confidence:
                unique_anomalies[key] = anomaly
        
        filtered_anomalies = list(unique_anomalies.values())
        
        # Rank by severity and confidence
        severity_order = {
            AnomalySeverity.CRITICAL: 4,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 1
        }
        
        filtered_anomalies.sort(
            key=lambda x: (severity_order[x.severity], x.confidence),
            reverse=True
        )
        
        return filtered_anomalies
    
    def _update_baseline_stats(self, data: Dict[str, TimeSeriesData]):
        """Update baseline statistics for adaptive thresholds."""
        for name, ts_data in data.items():
            if ts_data.values:
                stats = ts_data.get_statistics()
                self.baseline_stats[name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'updated_at': datetime.now()
                }
    
    def add_detection_rule(self, rule: DetectionRule):
        """Add a custom detection rule."""
        self.detection_rules.append(rule)
        self.logger.info(f"Added detection rule: {rule.name}")
    
    def remove_detection_rule(self, rule_name: str):
        """Remove a detection rule by name."""
        self.detection_rules = [r for r in self.detection_rules if r.name != rule_name]
        self.logger.info(f"Removed detection rule: {rule_name}")
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detection configuration."""
        return {
            'total_rules': len(self.detection_rules),
            'rules_by_type': {
                anomaly_type.value: len([r for r in self.detection_rules if r.anomaly_type == anomaly_type])
                for anomaly_type in AnomalyType
            },
            'baseline_metrics': list(self.baseline_stats.keys()),
            'detection_history_length': len(self.detection_history)
        }


class AnomalyDetectionOrchestrator:
    """Orchestrates anomaly detection across multiple data sources."""
    
    def __init__(self):
        self.logger = get_logger("anomaly_detection_orchestrator")
        self.detector = StatisticalAnomalyDetector()
        self.detection_history = []
        self.false_positive_feedback = {}
    
    async def run_detection_cycle(
        self,
        data: Dict[str, TimeSeriesData],
        timestamp: Optional[datetime] = None
    ) -> List[AnomalyDetection]:
        """Run a complete detection cycle."""
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            self.logger.info(f"Starting detection cycle at {timestamp}")
            
            # Run anomaly detection
            anomalies = self.detector.detect_anomalies(data, timestamp)
            
            # Apply feedback filtering
            filtered_anomalies = self._apply_feedback_filtering(anomalies)
            
            # Store in history
            self.detection_history.append({
                'timestamp': timestamp,
                'anomalies': filtered_anomalies,
                'data_sources': list(data.keys())
            })
            
            # Limit history size
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            self.logger.info(f"Detection cycle completed. Found {len(filtered_anomalies)} anomalies")
            return filtered_anomalies
            
        except Exception as e:
            self.logger.error(f"Error in detection cycle: {e}")
            return []
    
    def _apply_feedback_filtering(
        self,
        anomalies: List[AnomalyDetection]
    ) -> List[AnomalyDetection]:
        """Filter anomalies based on historical feedback."""
        filtered = []
        
        for anomaly in anomalies:
            # Check if this type of anomaly has been marked as false positive
            fp_key = f"{anomaly.anomaly_type}_{anomaly.affected_metrics[0] if anomaly.affected_metrics else 'unknown'}"
            
            if fp_key in self.false_positive_feedback:
                fp_rate = self.false_positive_feedback[fp_key]['rate']
                if fp_rate > 0.8:  # High false positive rate
                    # Increase threshold for this type
                    anomaly.confidence *= (1 - fp_rate * 0.5)
                    if anomaly.confidence < 0.3:
                        continue  # Skip low confidence anomalies
            
            filtered.append(anomaly)
        
        return filtered
    
    def add_feedback(
        self,
        anomaly_id: str,
        is_true_positive: bool,
        feedback_notes: str = ""
    ):
        """Add feedback about anomaly detection accuracy."""
        # Find the anomaly in history
        for history_entry in reversed(self.detection_history):
            for anomaly in history_entry['anomalies']:
                if anomaly.id == anomaly_id:
                    # Update feedback statistics
                    fp_key = f"{anomaly.anomaly_type}_{anomaly.affected_metrics[0] if anomaly.affected_metrics else 'unknown'}"
                    
                    if fp_key not in self.false_positive_feedback:
                        self.false_positive_feedback[fp_key] = {
                            'total': 0,
                            'false_positives': 0,
                            'rate': 0.0
                        }
                    
                    self.false_positive_feedback[fp_key]['total'] += 1
                    if not is_true_positive:
                        self.false_positive_feedback[fp_key]['false_positives'] += 1
                    
                    # Update rate
                    fp_stats = self.false_positive_feedback[fp_key]
                    fp_stats['rate'] = fp_stats['false_positives'] / fp_stats['total']
                    
                    self.logger.info(f"Added feedback for {anomaly_id}: {'TP' if is_true_positive else 'FP'}")
                    return
        
        self.logger.warning(f"Anomaly {anomaly_id} not found in history")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {}
        
        total_anomalies = sum(len(entry['anomalies']) for entry in self.detection_history)
        
        # Count by type
        type_counts = {}
        severity_counts = {}
        
        for entry in self.detection_history:
            for anomaly in entry['anomalies']:
                type_counts[anomaly.anomaly_type.value] = type_counts.get(anomaly.anomaly_type.value, 0) + 1
                severity_counts[anomaly.severity.value] = severity_counts.get(anomaly.severity.value, 0) + 1
        
        return {
            'total_detection_cycles': len(self.detection_history),
            'total_anomalies_detected': total_anomalies,
            'avg_anomalies_per_cycle': total_anomalies / len(self.detection_history),
            'anomalies_by_type': type_counts,
            'anomalies_by_severity': severity_counts,
            'false_positive_rates': {
                k: v['rate'] for k, v in self.false_positive_feedback.items()
            },
            'detection_rules_count': len(self.detector.detection_rules)
        }
