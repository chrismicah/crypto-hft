"""
Hidden Markov Model classifier for market regime identification.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")
    hmm = None
    GaussianHMM = None

from ..models.regime_models import (
    MarketRegime, RegimeClassification, RegimeConfidence, RegimeTransition,
    RegimeModelConfig, RegimePerformanceMetrics, get_confidence_level
)
from ..features.feature_extractor import FeatureExtractor
from ...common.logger import get_logger


class HMMRegimeClassifier:
    """Hidden Markov Model for market regime classification."""
    
    def __init__(self, config: RegimeModelConfig):
        self.config = config
        self.logger = get_logger("hmm_classifier")
        
        # HMM model
        self.model: Optional[GaussianHMM] = None
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
        # State mappings
        self.state_to_regime: Dict[int, MarketRegime] = {}
        self.regime_to_state: Dict[MarketRegime, int] = {}
        
        # Model metadata
        self.is_trained = False
        self.training_date = None
        self.feature_names = []
        self.performance_metrics = RegimePerformanceMetrics()
        
        # Classification history
        self.classification_history: List[RegimeClassification] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Smoothing parameters
        self.regime_persistence_counter = {}
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_start_time = datetime.now()
    
    def train(self, 
              feature_data: List[np.ndarray], 
              timestamps: List[datetime],
              validation_split: float = 0.2) -> RegimePerformanceMetrics:
        """
        Train the HMM model on historical data.
        
        Args:
            feature_data: List of feature arrays
            timestamps: Corresponding timestamps
            validation_split: Fraction of data for validation
            
        Returns:
            Performance metrics
        """
        try:
            self.logger.info(f"Training HMM model with {len(feature_data)} samples")
            
            if len(feature_data) < self.config.n_components * 10:
                raise ValueError(f"Insufficient data for training. Need at least {self.config.n_components * 10} samples")
            
            # Prepare data
            X = np.vstack(feature_data)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            if validation_split > 0:
                X_train, X_val, ts_train, ts_val = train_test_split(
                    X_scaled, timestamps, 
                    test_size=validation_split, 
                    shuffle=False  # Preserve temporal order
                )
            else:
                X_train, X_val = X_scaled, None
                ts_train, ts_val = timestamps, None
            
            # Initialize and train HMM
            self.model = GaussianHMM(
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=42
            )
            
            # Fit the model
            self.model.fit(X_train)
            
            # Create state-to-regime mapping
            self._create_regime_mapping(X_train, ts_train)
            
            # Evaluate model
            if X_val is not None:
                self.performance_metrics = self._evaluate_model(X_val, ts_val)
            
            # Update model metadata
            self.is_trained = True
            self.training_date = datetime.now()
            self.feature_names = self.feature_extractor.get_feature_names()
            
            self.logger.info(f"HMM training completed. Log-likelihood: {self.model.score(X_train):.2f}")
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error training HMM model: {e}")
            raise
    
    def predict(self, features: np.ndarray, timestamp: datetime) -> RegimeClassification:
        """
        Predict market regime for given features.
        
        Args:
            features: Feature array
            timestamp: Current timestamp
            
        Returns:
            Regime classification
        """
        if not self.is_trained:
            return RegimeClassification(
                timestamp=timestamp,
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                confidence_level=RegimeConfidence.VERY_LOW
            )
        
        try:
            start_time = datetime.now()
            
            # Prepare features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get state probabilities
            log_probs = self.model.score_samples(features_scaled)
            state_probs = self.model.predict_proba(features_scaled)[0]
            
            # Map to regime probabilities
            regime_probs = {}
            for state, prob in enumerate(state_probs):
                if state in self.state_to_regime:
                    regime = self.state_to_regime[state]
                    regime_probs[regime] = float(prob)
            
            # Get most likely regime
            if regime_probs:
                predicted_regime = max(regime_probs, key=regime_probs.get)
                confidence = regime_probs[predicted_regime]
            else:
                predicted_regime = MarketRegime.UNKNOWN
                confidence = 0.0
            
            # Apply smoothing to reduce noise
            smoothed_regime, smoothed_confidence = self._apply_regime_smoothing(
                predicted_regime, confidence, timestamp
            )
            
            # Check for regime transition
            transition = None
            if smoothed_regime != self.current_regime:
                transition = RegimeTransition(
                    timestamp=timestamp,
                    from_regime=self.current_regime,
                    to_regime=smoothed_regime,
                    transition_probability=smoothed_confidence,
                    duration_in_previous=timestamp - self.regime_start_time
                )
                self.transition_history.append(transition)
                self.current_regime = smoothed_regime
                self.regime_start_time = timestamp
            
            # Create classification result
            classification = RegimeClassification(
                timestamp=timestamp,
                regime=smoothed_regime,
                confidence=smoothed_confidence,
                confidence_level=get_confidence_level(smoothed_confidence),
                regime_probabilities=regime_probs,
                classification_time=(datetime.now() - start_time).total_seconds(),
                previous_regime=transition.from_regime if transition else None,
                regime_duration=timestamp - self.regime_start_time,
                transition_probability=transition.transition_probability if transition else 0.0
            )
            
            # Store in history
            self.classification_history.append(classification)
            
            # Limit history size
            if len(self.classification_history) > 10000:
                self.classification_history = self.classification_history[-5000:]
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Error predicting regime: {e}")
            return RegimeClassification(
                timestamp=timestamp,
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                confidence_level=RegimeConfidence.VERY_LOW
            )
    
    def predict_sequence(self, 
                        feature_sequence: np.ndarray, 
                        timestamps: List[datetime]) -> List[RegimeClassification]:
        """
        Predict regime sequence for multiple time points.
        
        Args:
            feature_sequence: Array of features [n_samples, n_features]
            timestamps: List of timestamps
            
        Returns:
            List of regime classifications
        """
        if not self.is_trained:
            return []
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(feature_sequence)
            
            # Get state sequence
            state_sequence = self.model.predict(features_scaled)
            state_probs = self.model.predict_proba(features_scaled)
            
            classifications = []
            
            for i, (state, probs, timestamp) in enumerate(zip(state_sequence, state_probs, timestamps)):
                # Map state to regime
                regime = self.state_to_regime.get(state, MarketRegime.UNKNOWN)
                confidence = float(probs[state])
                
                # Create regime probabilities
                regime_probs = {}
                for s, prob in enumerate(probs):
                    if s in self.state_to_regime:
                        r = self.state_to_regime[s]
                        regime_probs[r] = float(prob)
                
                classification = RegimeClassification(
                    timestamp=timestamp,
                    regime=regime,
                    confidence=confidence,
                    confidence_level=get_confidence_level(confidence),
                    regime_probabilities=regime_probs
                )
                
                classifications.append(classification)
            
            return classifications
            
        except Exception as e:
            self.logger.error(f"Error predicting sequence: {e}")
            return []
    
    def _create_regime_mapping(self, X: np.ndarray, timestamps: List[datetime]):
        """Create mapping between HMM states and market regimes."""
        try:
            # Get state sequence
            states = self.model.predict(X)
            
            # Analyze each state's characteristics
            state_characteristics = {}
            
            for state in range(self.config.n_components):
                state_mask = states == state
                if not np.any(state_mask):
                    continue
                
                state_features = X[state_mask]
                
                # Calculate average characteristics for this state
                # Assuming feature order: returns, volatility, volume, technical, etc.
                avg_return_24h = np.mean(state_features[:, 2]) if state_features.shape[1] > 2 else 0
                avg_volatility_24h = np.mean(state_features[:, 6]) if state_features.shape[1] > 6 else 0
                avg_volume_ratio = np.mean(state_features[:, 8]) if state_features.shape[1] > 8 else 1
                
                state_characteristics[state] = {
                    'return_24h': avg_return_24h,
                    'volatility_24h': avg_volatility_24h,
                    'volume_ratio': avg_volume_ratio,
                    'sample_count': np.sum(state_mask)
                }
            
            # Map states to regimes based on characteristics
            self._assign_regimes_to_states(state_characteristics)
            
            self.logger.info(f"Created regime mapping: {self.state_to_regime}")
            
        except Exception as e:
            self.logger.error(f"Error creating regime mapping: {e}")
            # Fallback mapping
            for i in range(self.config.n_components):
                self.state_to_regime[i] = MarketRegime.UNKNOWN
    
    def _assign_regimes_to_states(self, state_characteristics: Dict[int, Dict[str, float]]):
        """Assign market regimes to HMM states based on characteristics."""
        
        # Define thresholds
        high_vol_threshold = 0.5  # Normalized volatility threshold
        high_return_threshold = 0.02  # 2% return threshold
        low_return_threshold = -0.02
        
        for state, chars in state_characteristics.items():
            return_24h = chars['return_24h']
            volatility_24h = chars['volatility_24h']
            
            # Classify based on volatility and returns
            if volatility_24h > high_vol_threshold:
                # High volatility regimes
                if return_24h > high_return_threshold:
                    regime = MarketRegime.HIGH_VOL_BULL
                elif return_24h < low_return_threshold:
                    regime = MarketRegime.HIGH_VOL_BEAR
                else:
                    regime = MarketRegime.HIGH_VOL_RANGE
            else:
                # Low volatility regimes
                if return_24h > high_return_threshold:
                    regime = MarketRegime.LOW_VOL_BULL
                elif return_24h < low_return_threshold:
                    regime = MarketRegime.LOW_VOL_BEAR
                else:
                    if abs(return_24h) < 0.005:  # Very stable
                        regime = MarketRegime.STABLE_RANGE
                    else:
                        regime = MarketRegime.LOW_VOL_RANGE
            
            self.state_to_regime[state] = regime
            self.regime_to_state[regime] = state
    
    def _apply_regime_smoothing(self, 
                               predicted_regime: MarketRegime, 
                               confidence: float, 
                               timestamp: datetime) -> Tuple[MarketRegime, float]:
        """Apply smoothing to reduce regime switching noise."""
        
        # Initialize persistence counter if needed
        if predicted_regime not in self.regime_persistence_counter:
            self.regime_persistence_counter[predicted_regime] = 0
        
        # Check if confidence is above threshold
        if confidence < self.config.min_confidence_threshold:
            # Low confidence - stick with current regime
            return self.current_regime, confidence * 0.8
        
        # Check if this is a new regime
        if predicted_regime != self.current_regime:
            # Increment counter for new regime
            self.regime_persistence_counter[predicted_regime] += 1
            
            # Check if we have enough persistence
            if self.regime_persistence_counter[predicted_regime] >= self.config.regime_persistence:
                # Confirm regime change
                self.regime_persistence_counter = {predicted_regime: 0}  # Reset counters
                return predicted_regime, confidence
            else:
                # Not enough persistence - stick with current regime
                return self.current_regime, confidence * 0.9
        else:
            # Same regime - reset other counters
            for regime in self.regime_persistence_counter:
                if regime != predicted_regime:
                    self.regime_persistence_counter[regime] = 0
            
            return predicted_regime, confidence
    
    def _evaluate_model(self, X_val: np.ndarray, ts_val: List[datetime]) -> RegimePerformanceMetrics:
        """Evaluate model performance on validation data."""
        try:
            # Get predictions
            predicted_states = self.model.predict(X_val)
            state_probs = self.model.predict_proba(X_val)
            
            # Calculate log-likelihood
            log_likelihood = self.model.score(X_val)
            
            # Calculate AIC and BIC
            n_params = (self.config.n_components * (self.config.n_components - 1) +  # Transition matrix
                       self.config.n_components * X_val.shape[1] +  # Means
                       self.config.n_components * X_val.shape[1])   # Covariances (simplified)
            
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(len(X_val)) * n_params - 2 * log_likelihood
            
            # Create performance metrics
            metrics = RegimePerformanceMetrics(
                log_likelihood=float(log_likelihood),
                aic_score=float(aic),
                bic_score=float(bic)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return RegimePerformanceMetrics()
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime classifications."""
        if not self.classification_history:
            return {}
        
        try:
            # Count regime occurrences
            regime_counts = {}
            for classification in self.classification_history:
                regime = classification.regime
                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
            
            # Calculate average confidence by regime
            regime_confidences = {}
            for regime in MarketRegime:
                regime_classifications = [c for c in self.classification_history if c.regime == regime]
                if regime_classifications:
                    avg_confidence = np.mean([c.confidence for c in regime_classifications])
                    regime_confidences[regime.value] = float(avg_confidence)
            
            # Calculate transition statistics
            transition_counts = {}
            for transition in self.transition_history:
                key = f"{transition.from_regime.value} -> {transition.to_regime.value}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            # Calculate regime durations
            regime_durations = {}
            for transition in self.transition_history:
                regime = transition.from_regime.value
                duration_hours = transition.duration_in_previous.total_seconds() / 3600
                if regime not in regime_durations:
                    regime_durations[regime] = []
                regime_durations[regime].append(duration_hours)
            
            # Average durations
            avg_durations = {}
            for regime, durations in regime_durations.items():
                avg_durations[regime] = float(np.mean(durations))
            
            return {
                'total_classifications': len(self.classification_history),
                'regime_counts': regime_counts,
                'regime_confidences': regime_confidences,
                'total_transitions': len(self.transition_history),
                'transition_counts': transition_counts,
                'average_regime_durations': avg_durations,
                'current_regime': self.current_regime.value,
                'regime_start_time': self.regime_start_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regime statistics: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config.dict(),
                'state_to_regime': {k: v.value for k, v in self.state_to_regime.items()},
                'regime_to_state': {k.value: v for k, v in self.regime_to_state.items()},
                'training_date': self.training_date.isoformat(),
                'feature_names': self.feature_names,
                'performance_metrics': self.performance_metrics.to_dict()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.config = RegimeModelConfig(**model_data['config'])
            
            # Restore mappings
            self.state_to_regime = {k: MarketRegime(v) for k, v in model_data['state_to_regime'].items()}
            self.regime_to_state = {MarketRegime(k): v for k, v in model_data['regime_to_state'].items()}
            
            self.training_date = datetime.fromisoformat(model_data['training_date'])
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'is_trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'n_components': self.config.n_components,
            'covariance_type': self.config.covariance_type,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'regime_mapping': {k: v.value for k, v in self.state_to_regime.items()},
            'classification_count': len(self.classification_history),
            'transition_count': len(self.transition_history),
            'current_regime': self.current_regime.value,
            'performance_metrics': self.performance_metrics.to_dict()
        }
