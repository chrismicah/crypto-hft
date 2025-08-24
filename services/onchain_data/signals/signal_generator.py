"""
On-chain signal generation using machine learning models.
Converts engineered features into trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
import pickle
import json
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb

from ..models import (
    OnChainFeatureSet, OnChainSignal, SignalStrength,
    OnChainMetricType, OnChainAlert
)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_type: str = "xgboost"  # xgboost, lightgbm, random_forest, gradient_boosting
    classification: bool = True  # True for classification, False for regression
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    min_training_samples: int = 100
    
    # Feature selection
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01
    
    # Signal generation
    signal_threshold_strong: float = 0.7  # Threshold for strong signals
    signal_threshold_weak: float = 0.3    # Threshold for weak signals
    confidence_threshold: float = 0.5     # Minimum confidence for signals


class OnChainSignalGenerator:
    """
    ML-based signal generator for on-chain data.
    
    Uses multiple models to generate trading signals:
    - Classification models for direction (bullish/bearish/neutral)
    - Regression models for signal strength
    - Ensemble methods for robust predictions
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.classification_model = None
        self.regression_model = None
        self.ensemble_models = {}
        
        # Training data and labels
        self.feature_columns = []
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Model performance tracking
        self.model_metrics = {}
        self.feature_importance = {}
        
        # Signal history for validation
        self.signal_history = []
    
    def _create_model(self, model_type: str, classification: bool = True):
        """Create a model instance based on configuration."""
        if model_type == "xgboost":
            if classification:
                return xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    objective='multi:softprob'
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state
                )
        
        elif model_type == "lightgbm":
            if classification:
                return lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    objective='multiclass'
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state
                )
        
        elif model_type == "random_forest":
            if classification:
                return RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    max_features=self.config.max_features
                )
            else:
                return RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    max_features=self.config.max_features
                )
        
        elif model_type == "gradient_boosting":
            if classification:
                return GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state
                )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_training_data(
        self,
        feature_sets: List[OnChainFeatureSet],
        price_data: Optional[pd.DataFrame] = None,
        forward_returns_days: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from feature sets and price data.
        
        Args:
            feature_sets: List of OnChainFeatureSet objects
            price_data: DataFrame with price data (timestamp, close)
            forward_returns_days: Days ahead to calculate returns for labels
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        if not feature_sets:
            raise ValueError("No feature sets provided")
        
        # Convert feature sets to DataFrame
        features_data = []
        timestamps = []
        
        for fs in feature_sets:
            feature_vector = fs.get_feature_vector()
            if feature_vector:  # Only add if we have features
                features_data.append(feature_vector)
                timestamps.append(fs.timestamp)
        
        if not features_data:
            raise ValueError("No valid feature vectors found")
        
        features = np.array(features_data)
        feature_names = feature_sets[0].get_feature_names()
        
        # Create labels based on forward returns
        labels = []
        
        if price_data is not None:
            # Ensure price data is sorted by timestamp
            price_data = price_data.sort_index()
            
            for timestamp in timestamps:
                # Find current and future price
                current_price = self._get_price_at_timestamp(price_data, timestamp)
                future_timestamp = timestamp + timedelta(days=forward_returns_days)
                future_price = self._get_price_at_timestamp(price_data, future_timestamp)
                
                if current_price and future_price:
                    return_pct = (future_price - current_price) / current_price
                    
                    # Convert returns to signal categories
                    if return_pct > 0.02:  # 2% gain
                        labels.append(SignalStrength.VERY_BULLISH.value)
                    elif return_pct > 0.005:  # 0.5% gain
                        labels.append(SignalStrength.BULLISH.value)
                    elif return_pct < -0.02:  # 2% loss
                        labels.append(SignalStrength.VERY_BEARISH.value)
                    elif return_pct < -0.005:  # 0.5% loss
                        labels.append(SignalStrength.BEARISH.value)
                    else:
                        labels.append(SignalStrength.NEUTRAL.value)
                else:
                    labels.append(SignalStrength.NEUTRAL.value)
        else:
            # Use composite scores as labels if no price data
            for fs in feature_sets:
                bullish = fs.onchain_bullish_score or 0
                bearish = fs.onchain_bearish_score or 0
                
                net_score = bullish - bearish
                
                if net_score > 0.5:
                    labels.append(SignalStrength.VERY_BULLISH.value)
                elif net_score > 0.1:
                    labels.append(SignalStrength.BULLISH.value)
                elif net_score < -0.5:
                    labels.append(SignalStrength.VERY_BEARISH.value)
                elif net_score < -0.1:
                    labels.append(SignalStrength.BEARISH.value)
                else:
                    labels.append(SignalStrength.NEUTRAL.value)
        
        # Ensure we have the same number of features and labels
        min_length = min(len(features), len(labels))
        features = features[:min_length]
        labels = np.array(labels[:min_length])
        
        self.logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels, feature_names
    
    def _get_price_at_timestamp(
        self,
        price_data: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[float]:
        """Get price at or closest to given timestamp."""
        try:
            # Find the closest timestamp
            closest_idx = price_data.index.get_indexer([timestamp], method='nearest')[0]
            if closest_idx >= 0:
                return float(price_data.iloc[closest_idx]['close'])
        except (KeyError, IndexError, ValueError):
            pass
        return None
    
    def train_models(
        self,
        feature_sets: List[OnChainFeatureSet],
        price_data: Optional[pd.DataFrame] = None,
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Train classification and regression models.
        
        Args:
            feature_sets: Training feature sets
            price_data: Price data for label generation
            validate: Whether to perform validation
            
        Returns:
            Dictionary of model performance metrics
        """
        self.logger.info("Training on-chain signal models")
        
        # Prepare training data
        features, labels, feature_names = self.prepare_training_data(
            feature_sets, price_data
        )
        
        if len(features) < self.config.min_training_samples:
            raise ValueError(f"Insufficient training samples: {len(features)} < {self.config.min_training_samples}")
        
        self.feature_columns = feature_names
        
        # Encode labels for classification
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data for validation
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels_encoded,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labels_encoded
            )
        else:
            X_train, y_train = features, labels_encoded
            X_val, y_val = None, None
        
        # Train classification model
        self.classification_model = self._create_model(
            self.config.model_type, classification=True
        )
        
        self.logger.info("Training classification model")
        self.classification_model.fit(X_train, y_train)
        
        # Train regression model for signal strength
        # Convert categorical labels to numeric scores
        label_scores = []
        for label in labels:
            if label == SignalStrength.VERY_BULLISH.value:
                label_scores.append(1.0)
            elif label == SignalStrength.BULLISH.value:
                label_scores.append(0.5)
            elif label == SignalStrength.NEUTRAL.value:
                label_scores.append(0.0)
            elif label == SignalStrength.BEARISH.value:
                label_scores.append(-0.5)
            elif label == SignalStrength.VERY_BEARISH.value:
                label_scores.append(-1.0)
            else:
                label_scores.append(0.0)
        
        label_scores = np.array(label_scores)
        
        if validate:
            y_train_scores = label_scores[:len(X_train)]
            y_val_scores = label_scores[len(X_train):len(X_train)+len(X_val)] if X_val is not None else None
        else:
            y_train_scores = label_scores
            y_val_scores = None
        
        self.regression_model = self._create_model(
            self.config.model_type, classification=False
        )
        
        self.logger.info("Training regression model")
        self.regression_model.fit(X_train, y_train_scores)
        
        # Store feature importance
        self._calculate_feature_importance()
        
        # Validate models
        metrics = {}
        if validate and X_val is not None:
            metrics = self._validate_models(X_val, y_val, y_val_scores)
        
        self.is_fitted = True
        self.logger.info("Model training completed")
        
        return metrics
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        if hasattr(self.classification_model, 'feature_importances_'):
            importance = self.classification_model.feature_importances_
            self.feature_importance = dict(zip(self.feature_columns, importance))
            
            # Log top features
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            self.logger.info("Top 10 important features:")
            for feature, importance in sorted_features:
                self.logger.info(f"  {feature}: {importance:.4f}")
    
    def _validate_models(
        self,
        X_val: np.ndarray,
        y_val_class: np.ndarray,
        y_val_reg: np.ndarray
    ) -> Dict[str, float]:
        """Validate trained models."""
        metrics = {}
        
        # Classification metrics
        y_pred_class = self.classification_model.predict(X_val)
        
        metrics['classification_accuracy'] = accuracy_score(y_val_class, y_pred_class)
        metrics['classification_precision'] = precision_score(
            y_val_class, y_pred_class, average='weighted', zero_division=0
        )
        metrics['classification_recall'] = recall_score(
            y_val_class, y_pred_class, average='weighted', zero_division=0
        )
        metrics['classification_f1'] = f1_score(
            y_val_class, y_pred_class, average='weighted', zero_division=0
        )
        
        # Regression metrics
        y_pred_reg = self.regression_model.predict(X_val)
        
        mse = np.mean((y_val_reg - y_pred_reg) ** 2)
        mae = np.mean(np.abs(y_val_reg - y_pred_reg))
        r2 = 1 - (np.sum((y_val_reg - y_pred_reg) ** 2) / 
                  np.sum((y_val_reg - np.mean(y_val_reg)) ** 2))
        
        metrics['regression_mse'] = mse
        metrics['regression_mae'] = mae
        metrics['regression_r2'] = r2
        
        self.model_metrics = metrics
        
        self.logger.info("Model validation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def generate_signal(
        self,
        feature_set: OnChainFeatureSet,
        symbol: str
    ) -> Optional[OnChainSignal]:
        """
        Generate a trading signal from a feature set.
        
        Args:
            feature_set: OnChainFeatureSet object
            symbol: Asset symbol
            
        Returns:
            OnChainSignal object or None
        """
        if not self.is_fitted:
            self.logger.warning("Models not trained. Cannot generate signal.")
            return None
        
        # Get feature vector
        feature_vector = feature_set.get_feature_vector()
        if not feature_vector:
            self.logger.warning("No features available for signal generation")
            return None
        
        features = np.array(feature_vector).reshape(1, -1)
        
        # Get classification prediction and probabilities
        class_pred = self.classification_model.predict(features)[0]
        class_proba = self.classification_model.predict_proba(features)[0]
        
        # Get regression prediction for signal strength
        reg_pred = self.regression_model.predict(features)[0]
        
        # Convert predictions to signal
        predicted_label = self.label_encoder.inverse_transform([class_pred])[0]
        signal_strength = SignalStrength(predicted_label)
        
        # Calculate confidence from prediction probabilities
        confidence = float(np.max(class_proba))
        
        # Normalize regression score to [-1, 1] range
        signal_score = np.clip(reg_pred, -1.0, 1.0)
        
        # Only generate signal if confidence is above threshold
        if confidence < self.config.confidence_threshold:
            return None
        
        # Determine primary metrics that contributed to signal
        primary_metrics = self._get_contributing_metrics(feature_set)
        
        # Create signal
        signal = OnChainSignal(
            signal_id=f"onchain_{symbol}_{int(feature_set.timestamp.timestamp())}",
            symbol=symbol,
            timestamp=feature_set.timestamp,
            signal_type="onchain_ml",
            strength=signal_strength,
            confidence=confidence,
            score=float(signal_score),
            primary_metrics=primary_metrics,
            model_version=f"{self.config.model_type}_v1.0",
            features_used=self.feature_columns,
            feature_importance=self.feature_importance
        )
        
        # Store signal for validation
        self.signal_history.append(signal)
        
        return signal
    
    def _get_contributing_metrics(
        self,
        feature_set: OnChainFeatureSet
    ) -> List[OnChainMetricType]:
        """Identify which metrics contributed most to the signal."""
        contributing_metrics = []
        
        # Check key composite scores
        if feature_set.onchain_bullish_score and feature_set.onchain_bullish_score > 0.5:
            contributing_metrics.extend([
                OnChainMetricType.EXCHANGE_OUTFLOW,
                OnChainMetricType.WHALE_TRANSACTIONS
            ])
        
        if feature_set.onchain_bearish_score and feature_set.onchain_bearish_score > 0.5:
            contributing_metrics.extend([
                OnChainMetricType.EXCHANGE_INFLOW
            ])
        
        if feature_set.exchange_flow_momentum and abs(feature_set.exchange_flow_momentum) > 0.1:
            contributing_metrics.append(OnChainMetricType.EXCHANGE_NET_FLOW)
        
        if feature_set.whale_activity_ma_7d and feature_set.whale_activity_ma_7d > 0:
            contributing_metrics.append(OnChainMetricType.WHALE_TRANSACTIONS)
        
        return contributing_metrics[:5]  # Limit to top 5
    
    def generate_batch_signals(
        self,
        feature_sets: List[OnChainFeatureSet],
        symbol: str
    ) -> List[OnChainSignal]:
        """
        Generate signals for a batch of feature sets.
        
        Args:
            feature_sets: List of feature sets
            symbol: Asset symbol
            
        Returns:
            List of generated signals
        """
        signals = []
        
        for feature_set in feature_sets:
            signal = self.generate_signal(feature_set, symbol)
            if signal:
                signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} signals from {len(feature_sets)} feature sets")
        return signals
    
    def generate_alerts(
        self,
        signals: List[OnChainSignal],
        alert_thresholds: Optional[Dict[str, float]] = None
    ) -> List[OnChainAlert]:
        """
        Generate alerts based on signals.
        
        Args:
            signals: List of signals to analyze
            alert_thresholds: Custom alert thresholds
            
        Returns:
            List of alerts
        """
        if alert_thresholds is None:
            alert_thresholds = {
                "strong_signal": 0.8,
                "anomaly_score": 0.9,
                "confidence": 0.95
            }
        
        alerts = []
        
        for signal in signals:
            # Strong signal alert
            if abs(signal.score) > alert_thresholds["strong_signal"]:
                severity = "HIGH" if abs(signal.score) > 0.9 else "MEDIUM"
                
                alert = OnChainAlert(
                    alert_id=f"strong_signal_{signal.signal_id}",
                    symbol=signal.symbol,
                    timestamp=signal.timestamp,
                    alert_type="strong_onchain_signal",
                    severity=severity,
                    title=f"Strong On-Chain Signal: {signal.strength.value.title()}",
                    description=f"High confidence {signal.strength.value} signal detected with score {signal.score:.3f}",
                    metric_type=OnChainMetricType.EXCHANGE_NET_FLOW,  # Primary metric
                    threshold_value=alert_thresholds["strong_signal"],
                    actual_value=abs(signal.score),
                    confidence=signal.confidence,
                    recommended_action=self._get_recommended_action(signal)
                )
                alerts.append(alert)
        
        return alerts
    
    def _get_recommended_action(self, signal: OnChainSignal) -> str:
        """Get recommended action based on signal."""
        if signal.is_bullish():
            if abs(signal.score) > 0.8:
                return "STRONG_BUY: Consider increasing long positions"
            else:
                return "BUY: Consider taking long positions"
        elif signal.is_bearish():
            if abs(signal.score) > 0.8:
                return "STRONG_SELL: Consider reducing long positions or shorting"
            else:
                return "SELL: Consider reducing positions"
        else:
            return "HOLD: No clear directional signal"
    
    def save_models(self, filepath: str):
        """Save trained models to file."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")
        
        model_data = {
            'classification_model': self.classification_model,
            'regression_model': self.regression_model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classification_model = model_data['classification_model']
        self.regression_model = model_data['regression_model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data.get('config', self.config)
        self.feature_importance = model_data.get('feature_importance', {})
        self.model_metrics = model_data.get('model_metrics', {})
        
        self.is_fitted = True
        self.logger.info(f"Models loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models."""
        if not self.is_fitted:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "model_type": self.config.model_type,
            "num_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "feature_importance": self.feature_importance,
            "model_metrics": self.model_metrics,
            "signals_generated": len(self.signal_history)
        }
