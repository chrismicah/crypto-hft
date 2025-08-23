"""Kalman Filter implementation for dynamic hedge ratio calculation."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import structlog
from dataclasses import dataclass
from datetime import datetime

logger = structlog.get_logger(__name__)


@dataclass
class KalmanState:
    """Represents the state of the Kalman filter."""
    state_mean: np.ndarray
    state_covariance: np.ndarray
    timestamp: datetime
    n_observations: int


class DynamicHedgeRatioKalman:
    """
    Kalman Filter for calculating dynamic hedge ratios between two cointegrated assets.
    
    This implementation uses a state-space model where:
    - State: [hedge_ratio, hedge_ratio_velocity]
    - Observation: price_ratio = asset1_price / asset2_price
    
    The model assumes the hedge ratio follows a random walk with drift.
    """
    
    def __init__(
        self,
        initial_hedge_ratio: float = 1.0,
        process_variance: float = 1e-5,
        observation_variance: float = 1e-3,
        initial_state_variance: float = 1.0
    ):
        """
        Initialize the Kalman filter.
        
        Args:
            initial_hedge_ratio: Initial estimate of the hedge ratio
            process_variance: Variance of the process noise (how much hedge ratio can change)
            observation_variance: Variance of the observation noise (measurement error)
            initial_state_variance: Initial uncertainty in the hedge ratio estimate
        """
        self.process_variance = process_variance
        self.observation_variance = observation_variance
        
        # State vector: [hedge_ratio, hedge_ratio_velocity]
        self.state_dim = 2
        self.obs_dim = 1
        
        # Initialize state
        self.state_mean = np.array([initial_hedge_ratio, 0.0])
        self.state_covariance = np.eye(self.state_dim) * initial_state_variance
        
        # State transition matrix (random walk with drift)
        self.transition_matrix = np.array([
            [1.0, 1.0],  # hedge_ratio(t) = hedge_ratio(t-1) + velocity(t-1)
            [0.0, 1.0]   # velocity(t) = velocity(t-1)
        ])
        
        # Observation matrix (we observe the hedge ratio directly)
        self.observation_matrix = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.process_covariance = np.array([
            [process_variance, 0.0],
            [0.0, process_variance * 0.1]  # Velocity has lower variance
        ])
        
        # Observation noise covariance
        self.observation_covariance = np.array([[observation_variance]])
        
        # Statistics
        self.n_observations = 0
        self.last_update = None
        
        logger.info(
            "Kalman filter initialized",
            initial_hedge_ratio=initial_hedge_ratio,
            process_variance=process_variance,
            observation_variance=observation_variance
        )
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state (prior update).
        
        Returns:
            Tuple of (predicted_state_mean, predicted_state_covariance)
        """
        # Predict state: x_k|k-1 = F * x_k-1|k-1
        predicted_state_mean = self.transition_matrix @ self.state_mean
        
        # Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        predicted_state_covariance = (
            self.transition_matrix @ self.state_covariance @ self.transition_matrix.T
            + self.process_covariance
        )
        
        return predicted_state_mean, predicted_state_covariance
    
    def update(self, observation: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update the filter with a new observation.
        
        Args:
            observation: The observed price ratio (asset1_price / asset2_price)
            timestamp: Timestamp of the observation
            
        Returns:
            Dictionary containing updated hedge ratio and filter statistics
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Convert observation to numpy array
        obs_vector = np.array([observation])
        
        # Prediction step
        predicted_state_mean, predicted_state_covariance = self.predict()
        
        # Innovation (prediction error)
        predicted_observation = self.observation_matrix @ predicted_state_mean
        innovation = obs_vector - predicted_observation
        
        # Innovation covariance
        innovation_covariance = (
            self.observation_matrix @ predicted_state_covariance @ self.observation_matrix.T
            + self.observation_covariance
        )
        
        # Kalman gain
        kalman_gain = (
            predicted_state_covariance @ self.observation_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        
        # Update state estimate
        self.state_mean = predicted_state_mean + kalman_gain @ innovation
        
        # Update covariance estimate (Joseph form for numerical stability)
        identity = np.eye(self.state_dim)
        joseph_factor = identity - kalman_gain @ self.observation_matrix
        self.state_covariance = (
            joseph_factor @ predicted_state_covariance @ joseph_factor.T
            + kalman_gain @ self.observation_covariance @ kalman_gain.T
        )
        
        # Update statistics
        self.n_observations += 1
        self.last_update = timestamp
        
        # Calculate confidence metrics
        hedge_ratio = self.state_mean[0]
        hedge_ratio_variance = self.state_covariance[0, 0]
        confidence_interval_95 = 1.96 * np.sqrt(hedge_ratio_variance)
        
        result = {
            'hedge_ratio': float(hedge_ratio),
            'hedge_ratio_velocity': float(self.state_mean[1]),
            'hedge_ratio_variance': float(hedge_ratio_variance),
            'confidence_interval_95': float(confidence_interval_95),
            'innovation': float(innovation[0]),
            'innovation_variance': float(innovation_covariance[0, 0]),
            'log_likelihood': self._calculate_log_likelihood(innovation, innovation_covariance),
            'n_observations': self.n_observations,
            'timestamp': timestamp.isoformat()
        }
        
        logger.debug(
            "Kalman filter updated",
            hedge_ratio=hedge_ratio,
            observation=observation,
            innovation=float(innovation[0]),
            n_observations=self.n_observations
        )
        
        return result
    
    def _calculate_log_likelihood(self, innovation: np.ndarray, innovation_covariance: np.ndarray) -> float:
        """Calculate the log-likelihood of the current observation."""
        try:
            # Multivariate normal log-likelihood
            k = len(innovation)
            sign, logdet = np.linalg.slogdet(innovation_covariance)
            if sign <= 0:
                return -np.inf
            
            log_likelihood = -0.5 * (
                k * np.log(2 * np.pi)
                + logdet
                + innovation.T @ np.linalg.inv(innovation_covariance) @ innovation
            )
            return float(log_likelihood)
        except np.linalg.LinAlgError:
            return -np.inf
    
    def get_current_state(self) -> KalmanState:
        """Get the current state of the filter."""
        return KalmanState(
            state_mean=self.state_mean.copy(),
            state_covariance=self.state_covariance.copy(),
            timestamp=self.last_update or datetime.utcnow(),
            n_observations=self.n_observations
        )
    
    def set_state(self, state: KalmanState) -> None:
        """Set the filter state (used for loading from persistence)."""
        self.state_mean = state.state_mean.copy()
        self.state_covariance = state.state_covariance.copy()
        self.last_update = state.timestamp
        self.n_observations = state.n_observations
        
        logger.info(
            "Kalman filter state loaded",
            hedge_ratio=float(self.state_mean[0]),
            n_observations=self.n_observations,
            timestamp=state.timestamp.isoformat()
        )
    
    def reset(self, initial_hedge_ratio: float = 1.0) -> None:
        """Reset the filter to initial state."""
        self.state_mean = np.array([initial_hedge_ratio, 0.0])
        self.state_covariance = np.eye(self.state_dim)
        self.n_observations = 0
        self.last_update = None
        
        logger.info("Kalman filter reset", initial_hedge_ratio=initial_hedge_ratio)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the filter."""
        return {
            'state_mean': self.state_mean.tolist(),
            'state_covariance': self.state_covariance.tolist(),
            'state_covariance_determinant': float(np.linalg.det(self.state_covariance)),
            'state_covariance_condition_number': float(np.linalg.cond(self.state_covariance)),
            'n_observations': self.n_observations,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'process_variance': self.process_variance,
            'observation_variance': self.observation_variance
        }


class PairTradingKalmanFilter:
    """
    Higher-level interface for pair trading with Kalman filter.
    
    Manages multiple asset pairs and their respective Kalman filters.
    """
    
    def __init__(self):
        self.filters: Dict[str, DynamicHedgeRatioKalman] = {}
        self.pair_configs: Dict[str, Dict[str, Any]] = {}
        
    def add_pair(
        self,
        pair_id: str,
        asset1: str,
        asset2: str,
        initial_hedge_ratio: float = 1.0,
        process_variance: float = 1e-5,
        observation_variance: float = 1e-3
    ) -> None:
        """Add a new trading pair with its own Kalman filter."""
        self.filters[pair_id] = DynamicHedgeRatioKalman(
            initial_hedge_ratio=initial_hedge_ratio,
            process_variance=process_variance,
            observation_variance=observation_variance
        )
        
        self.pair_configs[pair_id] = {
            'asset1': asset1,
            'asset2': asset2,
            'initial_hedge_ratio': initial_hedge_ratio,
            'process_variance': process_variance,
            'observation_variance': observation_variance
        }
        
        logger.info(
            "Trading pair added",
            pair_id=pair_id,
            asset1=asset1,
            asset2=asset2,
            initial_hedge_ratio=initial_hedge_ratio
        )
    
    def update_pair(
        self,
        pair_id: str,
        asset1_price: float,
        asset2_price: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a trading pair with new prices."""
        if pair_id not in self.filters:
            logger.warning("Unknown trading pair", pair_id=pair_id)
            return None
        
        if asset2_price == 0:
            logger.warning("Zero price for asset2", pair_id=pair_id, asset2_price=asset2_price)
            return None
        
        # Calculate price ratio
        price_ratio = asset1_price / asset2_price
        
        # Update filter
        result = self.filters[pair_id].update(price_ratio, timestamp)
        
        # Add pair metadata
        result.update({
            'pair_id': pair_id,
            'asset1': self.pair_configs[pair_id]['asset1'],
            'asset2': self.pair_configs[pair_id]['asset2'],
            'asset1_price': asset1_price,
            'asset2_price': asset2_price,
            'price_ratio': price_ratio
        })
        
        return result
    
    def get_pair_state(self, pair_id: str) -> Optional[KalmanState]:
        """Get the current state of a trading pair."""
        if pair_id not in self.filters:
            return None
        return self.filters[pair_id].get_current_state()
    
    def get_all_pairs(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all trading pairs."""
        result = {}
        for pair_id in self.pair_configs.keys():
            config = self.pair_configs[pair_id]
            filter_obj = self.filters[pair_id]
            
            result[pair_id] = {
                **config,
                'current_hedge_ratio': float(filter_obj.state_mean[0]),
                'n_observations': filter_obj.n_observations,
                'last_update': filter_obj.last_update.isoformat() if filter_obj.last_update else None
            }
        
        return result
