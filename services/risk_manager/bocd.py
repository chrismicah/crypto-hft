"""Bayesian Online Changepoint Detection implementation for risk management."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import scipy.stats as stats
from scipy.special import logsumexp
import warnings

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChangePointEvent:
    """Represents a detected changepoint event."""
    timestamp: datetime
    probability: float
    run_length: int
    data_point: float
    confidence_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Determine confidence level based on probability."""
        if self.probability >= 0.9:
            self.confidence_level = "CRITICAL"
        elif self.probability >= 0.7:
            self.confidence_level = "HIGH"
        elif self.probability >= 0.5:
            self.confidence_level = "MEDIUM"
        else:
            self.confidence_level = "LOW"


@dataclass
class BOCDState:
    """Internal state of the BOCD algorithm."""
    run_length_probs: np.ndarray
    message_length: float
    growth_probs: np.ndarray
    changepoint_probs: List[float] = field(default_factory=list)
    data_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def __post_init__(self):
        """Initialize arrays if not provided."""
        if self.run_length_probs is None:
            self.run_length_probs = np.array([1.0])
        if self.growth_probs is None:
            self.growth_probs = np.array([1.0])


class BOCDWrapper:
    """
    Wrapper for Bayesian Online Changepoint Detection algorithm.
    
    This implementation uses a simplified BOCD approach suitable for
    real-time financial time series monitoring.
    """
    
    def __init__(
        self,
        hazard_rate: float = 1/250,  # Expected changepoint every 250 observations
        alpha0: float = 1.0,         # Prior shape parameter
        beta0: float = 1.0,          # Prior rate parameter
        kappa0: float = 1.0,         # Prior precision parameter
        mu0: float = 0.0,            # Prior mean
        max_run_length: int = 500,   # Maximum run length to track
        min_observations: int = 10    # Minimum observations before detection
    ):
        """
        Initialize BOCD wrapper.
        
        Args:
            hazard_rate: Prior probability of changepoint at each step
            alpha0: Prior shape parameter for precision
            beta0: Prior rate parameter for precision
            kappa0: Prior precision parameter for mean
            mu0: Prior mean
            max_run_length: Maximum run length to track
            min_observations: Minimum observations before changepoint detection
        """
        self.hazard_rate = hazard_rate
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.mu0 = mu0
        self.max_run_length = max_run_length
        self.min_observations = min_observations
        
        # Initialize state
        self.reset()
        
        logger.info("BOCD wrapper initialized",
                   hazard_rate=hazard_rate,
                   max_run_length=max_run_length,
                   min_observations=min_observations)
    
    def reset(self) -> None:
        """Reset the BOCD algorithm state."""
        self.state = BOCDState(
            run_length_probs=np.array([1.0]),
            message_length=0.0,
            growth_probs=np.array([1.0])
        )
        self.observation_count = 0
        
        logger.debug("BOCD state reset")
    
    def update(self, observation: float, timestamp: datetime) -> Tuple[float, ChangePointEvent]:
        """
        Update BOCD with new observation.
        
        Args:
            observation: New data point (e.g., spread value)
            timestamp: Timestamp of observation
            
        Returns:
            Tuple of (changepoint_probability, changepoint_event_or_none)
        """
        self.observation_count += 1
        
        # Store observation
        self.state.data_history.append(observation)
        self.state.timestamp_history.append(timestamp)
        
        # Skip detection for initial observations
        if self.observation_count < self.min_observations:
            return 0.0, None
        
        try:
            # Compute changepoint probability
            changepoint_prob = self._compute_changepoint_probability(observation)
            
            # Update internal state
            self._update_run_length_distribution(observation, changepoint_prob)
            
            # Store probability
            self.state.changepoint_probs.append(changepoint_prob)
            
            # Create changepoint event if significant
            changepoint_event = None
            if changepoint_prob > 0.1:  # Only create events for non-trivial probabilities
                changepoint_event = ChangePointEvent(
                    timestamp=timestamp,
                    probability=changepoint_prob,
                    run_length=self._get_most_likely_run_length(),
                    data_point=observation,
                    confidence_level="",  # Will be set in __post_init__
                    metadata={
                        'observation_count': self.observation_count,
                        'recent_mean': np.mean(list(self.state.data_history)[-20:]),
                        'recent_std': np.std(list(self.state.data_history)[-20:])
                    }
                )
            
            logger.debug("BOCD updated",
                        observation=observation,
                        changepoint_prob=changepoint_prob,
                        observation_count=self.observation_count)
            
            return changepoint_prob, changepoint_event
            
        except Exception as e:
            logger.error("Error in BOCD update", error=str(e), observation=observation)
            return 0.0, None
    
    def _compute_changepoint_probability(self, observation: float) -> float:
        """Compute the probability of a changepoint at current step."""
        try:
            # Get current run length distribution
            R = len(self.state.run_length_probs)
            
            # Compute predictive probabilities for each run length
            pred_probs = np.zeros(R)
            
            for r in range(R):
                if r == 0:
                    # New run length (changepoint occurred)
                    pred_probs[r] = self._predictive_probability(observation, 0)
                else:
                    # Continuing run length
                    pred_probs[r] = self._predictive_probability(observation, r)
            
            # Compute growth probabilities (probability of continuing each run length)
            growth_probs = pred_probs * self.state.run_length_probs * (1 - self.hazard_rate)
            
            # Compute changepoint probability (probability of starting new run length)
            changepoint_prob = (pred_probs[0] * 
                              np.sum(self.state.run_length_probs * self.hazard_rate))
            
            # Normalize
            total_prob = np.sum(growth_probs) + changepoint_prob
            
            if total_prob > 0:
                normalized_changepoint_prob = changepoint_prob / total_prob
            else:
                normalized_changepoint_prob = 0.0
            
            return float(np.clip(normalized_changepoint_prob, 0.0, 1.0))
            
        except Exception as e:
            logger.error("Error computing changepoint probability", error=str(e))
            return 0.0
    
    def _predictive_probability(self, observation: float, run_length: int) -> float:
        """
        Compute predictive probability of observation given run length.
        
        Uses Student's t-distribution as the predictive distribution.
        """
        try:
            if run_length == 0:
                # Use prior parameters
                alpha = self.alpha0
                beta = self.beta0
                kappa = self.kappa0
                mu = self.mu0
            else:
                # Update parameters based on recent observations
                recent_data = list(self.state.data_history)[-run_length:]
                
                if len(recent_data) == 0:
                    return self._predictive_probability(observation, 0)
                
                n = len(recent_data)
                sample_mean = np.mean(recent_data)
                sample_var = np.var(recent_data, ddof=1) if n > 1 else 1.0
                
                # Bayesian updates
                kappa = self.kappa0 + n
                mu = (self.kappa0 * self.mu0 + n * sample_mean) / kappa
                alpha = self.alpha0 + n / 2
                beta = (self.beta0 + 
                       0.5 * n * sample_var + 
                       0.5 * self.kappa0 * n * (sample_mean - self.mu0)**2 / kappa)
            
            # Student's t-distribution parameters
            df = 2 * alpha
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            
            # Compute probability density
            prob = stats.t.pdf(observation, df=df, loc=mu, scale=scale)
            
            return float(np.clip(prob, 1e-10, np.inf))
            
        except Exception as e:
            logger.error("Error computing predictive probability", 
                        error=str(e), 
                        observation=observation, 
                        run_length=run_length)
            return 1e-10
    
    def _update_run_length_distribution(self, observation: float, changepoint_prob: float) -> None:
        """Update the run length distribution."""
        try:
            R = len(self.state.run_length_probs)
            
            # Compute new run length probabilities
            new_probs = np.zeros(min(R + 1, self.max_run_length))
            
            # Probability of changepoint (new run length = 0)
            new_probs[0] = changepoint_prob
            
            # Probabilities of continuing run lengths
            for r in range(1, len(new_probs)):
                if r - 1 < len(self.state.run_length_probs):
                    pred_prob = self._predictive_probability(observation, r - 1)
                    new_probs[r] = (self.state.run_length_probs[r - 1] * 
                                   pred_prob * (1 - self.hazard_rate))
            
            # Normalize
            total_prob = np.sum(new_probs)
            if total_prob > 0:
                new_probs /= total_prob
            else:
                new_probs = np.array([1.0] + [0.0] * (len(new_probs) - 1))
            
            # Update state
            self.state.run_length_probs = new_probs
            
        except Exception as e:
            logger.error("Error updating run length distribution", error=str(e))
    
    def _get_most_likely_run_length(self) -> int:
        """Get the most likely current run length."""
        return int(np.argmax(self.state.run_length_probs))
    
    def get_recent_probabilities(self, n: int = 50) -> List[float]:
        """Get recent changepoint probabilities."""
        return list(self.state.changepoint_probs)[-n:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        recent_probs = self.get_recent_probabilities(100)
        
        stats_dict = {
            'observation_count': self.observation_count,
            'most_likely_run_length': self._get_most_likely_run_length(),
            'recent_mean_probability': np.mean(recent_probs) if recent_probs else 0.0,
            'recent_max_probability': np.max(recent_probs) if recent_probs else 0.0,
            'recent_data_mean': np.mean(list(self.state.data_history)) if self.state.data_history else 0.0,
            'recent_data_std': np.std(list(self.state.data_history)) if len(self.state.data_history) > 1 else 0.0,
            'hazard_rate': self.hazard_rate,
            'max_run_length': self.max_run_length
        }
        
        return stats_dict


class AdaptiveBOCDWrapper(BOCDWrapper):
    """
    Adaptive BOCD wrapper that adjusts parameters based on market conditions.
    """
    
    def __init__(
        self,
        base_hazard_rate: float = 1/250,
        volatility_adjustment: bool = True,
        trend_adjustment: bool = True,
        **kwargs
    ):
        """
        Initialize adaptive BOCD wrapper.
        
        Args:
            base_hazard_rate: Base hazard rate
            volatility_adjustment: Whether to adjust hazard rate based on volatility
            trend_adjustment: Whether to adjust hazard rate based on trend
            **kwargs: Additional arguments for base class
        """
        super().__init__(hazard_rate=base_hazard_rate, **kwargs)
        
        self.base_hazard_rate = base_hazard_rate
        self.volatility_adjustment = volatility_adjustment
        self.trend_adjustment = trend_adjustment
        
        # Adaptive parameters
        self.volatility_window = 50
        self.trend_window = 20
        
        logger.info("Adaptive BOCD wrapper initialized",
                   base_hazard_rate=base_hazard_rate,
                   volatility_adjustment=volatility_adjustment,
                   trend_adjustment=trend_adjustment)
    
    def update(self, observation: float, timestamp: datetime) -> Tuple[float, ChangePointEvent]:
        """Update with adaptive hazard rate."""
        # Adjust hazard rate based on market conditions
        self._adjust_hazard_rate()
        
        # Call parent update
        return super().update(observation, timestamp)
    
    def _adjust_hazard_rate(self) -> None:
        """Adjust hazard rate based on recent market conditions."""
        if len(self.state.data_history) < self.volatility_window:
            return
        
        try:
            recent_data = np.array(list(self.state.data_history)[-self.volatility_window:])
            
            adjustment_factor = 1.0
            
            # Volatility adjustment
            if self.volatility_adjustment:
                recent_volatility = np.std(recent_data)
                long_term_volatility = np.std(list(self.state.data_history))
                
                if long_term_volatility > 0:
                    vol_ratio = recent_volatility / long_term_volatility
                    # Higher volatility -> higher hazard rate
                    adjustment_factor *= (1 + 0.5 * (vol_ratio - 1))
            
            # Trend adjustment
            if self.trend_adjustment and len(recent_data) >= self.trend_window:
                trend_data = recent_data[-self.trend_window:]
                
                # Simple trend detection using linear regression slope
                x = np.arange(len(trend_data))
                slope = np.polyfit(x, trend_data, 1)[0]
                
                # Strong trends might indicate regime changes
                normalized_slope = abs(slope) / (np.std(trend_data) + 1e-8)
                trend_factor = 1 + 0.3 * normalized_slope
                adjustment_factor *= trend_factor
            
            # Apply bounds
            adjustment_factor = np.clip(adjustment_factor, 0.1, 5.0)
            
            # Update hazard rate
            self.hazard_rate = self.base_hazard_rate * adjustment_factor
            
            logger.debug("Hazard rate adjusted",
                        base_rate=self.base_hazard_rate,
                        adjustment_factor=adjustment_factor,
                        new_rate=self.hazard_rate)
            
        except Exception as e:
            logger.error("Error adjusting hazard rate", error=str(e))
            self.hazard_rate = self.base_hazard_rate


def create_synthetic_changepoint_data(
    n_points: int = 1000,
    changepoint_locations: List[int] = None,
    noise_std: float = 1.0,
    regime_means: List[float] = None,
    regime_stds: List[float] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Create synthetic data with known changepoints for testing.
    
    Args:
        n_points: Number of data points
        changepoint_locations: List of changepoint locations
        noise_std: Standard deviation of noise
        regime_means: Mean for each regime
        regime_stds: Standard deviation for each regime
        
    Returns:
        Tuple of (data_array, actual_changepoints)
    """
    if changepoint_locations is None:
        changepoint_locations = [n_points // 3, 2 * n_points // 3]
    
    if regime_means is None:
        regime_means = [0.0, 2.0, -1.0]
    
    if regime_stds is None:
        regime_stds = [1.0, 1.5, 0.8]
    
    # Ensure we have enough regimes
    n_regimes = len(changepoint_locations) + 1
    while len(regime_means) < n_regimes:
        regime_means.append(np.random.normal(0, 2))
    while len(regime_stds) < n_regimes:
        regime_stds.append(np.random.uniform(0.5, 2.0))
    
    data = np.zeros(n_points)
    current_regime = 0
    
    for i in range(n_points):
        # Check if we've hit a changepoint
        if i in changepoint_locations:
            current_regime += 1
        
        # Generate data point for current regime
        mean = regime_means[current_regime]
        std = regime_stds[current_regime]
        
        data[i] = np.random.normal(mean, std)
    
    logger.info("Created synthetic changepoint data",
               n_points=n_points,
               changepoints=changepoint_locations,
               n_regimes=n_regimes)
    
    return data, changepoint_locations
