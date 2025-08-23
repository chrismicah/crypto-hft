"""GARCH model implementation for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
import warnings
from collections import deque

from arch import arch_model
from arch.univariate import GARCH, ConstantMean
import statsmodels.api as sm

logger = structlog.get_logger(__name__)

# Suppress ARCH warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='arch')


@dataclass
class GARCHForecast:
    """Container for GARCH model forecast results."""
    volatility_forecast: float
    variance_forecast: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    forecast_horizon: int
    model_aic: float
    model_bic: float
    timestamp: datetime


@dataclass
class GARCHModelState:
    """Container for GARCH model state and parameters."""
    omega: float  # Constant term
    alpha: float  # ARCH coefficient
    beta: float   # GARCH coefficient
    log_likelihood: float
    aic: float
    bic: float
    fitted_at: datetime
    n_observations: int


class RollingGARCHModel:
    """
    Rolling GARCH(1,1) model for volatility forecasting.
    
    This implementation maintains a rolling window of spread data and
    continuously updates the GARCH model to provide real-time volatility forecasts.
    """
    
    def __init__(
        self,
        window_size: int = 252,  # ~1 year of daily data
        min_observations: int = 50,
        refit_frequency: int = 10,  # Refit every N new observations
        confidence_level: float = 0.95
    ):
        """
        Initialize the rolling GARCH model.
        
        Args:
            window_size: Size of the rolling window for model fitting
            min_observations: Minimum observations required to fit model
            refit_frequency: How often to refit the model (in observations)
            confidence_level: Confidence level for forecast intervals
        """
        self.window_size = window_size
        self.min_observations = min_observations
        self.refit_frequency = refit_frequency
        self.confidence_level = confidence_level
        
        # Rolling data storage
        self.spread_data = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Model state
        self.model = None
        self.fitted_model = None
        self.last_fit_time = None
        self.observations_since_fit = 0
        self.model_state: Optional[GARCHModelState] = None
        
        # Statistics
        self.total_observations = 0
        self.successful_fits = 0
        self.failed_fits = 0
        
        logger.info(
            "GARCH model initialized",
            window_size=window_size,
            min_observations=min_observations,
            refit_frequency=refit_frequency,
            confidence_level=confidence_level
        )
    
    def add_observation(
        self, 
        spread_value: float, 
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Add a new spread observation to the rolling window.
        
        Args:
            spread_value: The spread value to add
            timestamp: Timestamp of the observation
            
        Returns:
            True if observation was added successfully
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Validate input
        if not np.isfinite(spread_value):
            logger.warning("Invalid spread value", value=spread_value)
            return False
        
        # Add to rolling window
        self.spread_data.append(spread_value)
        self.timestamps.append(timestamp)
        
        self.total_observations += 1
        self.observations_since_fit += 1
        
        logger.debug(
            "Added spread observation",
            value=spread_value,
            timestamp=timestamp.isoformat(),
            window_size=len(self.spread_data)
        )
        
        return True
    
    def should_refit(self) -> bool:
        """Check if the model should be refitted."""
        if len(self.spread_data) < self.min_observations:
            return False
        
        if self.fitted_model is None:
            return True
        
        if self.observations_since_fit >= self.refit_frequency:
            return True
        
        return False
    
    def fit_model(self, force: bool = False) -> bool:
        """
        Fit the GARCH(1,1) model to current data.
        
        Args:
            force: Force refitting even if not needed
            
        Returns:
            True if model was fitted successfully
        """
        if not force and not self.should_refit():
            return True
        
        if len(self.spread_data) < self.min_observations:
            logger.warning(
                "Insufficient data for GARCH fitting",
                current_size=len(self.spread_data),
                required=self.min_observations
            )
            return False
        
        try:
            # Convert to pandas Series for ARCH
            spread_series = pd.Series(
                list(self.spread_data),
                index=pd.DatetimeIndex(list(self.timestamps))
            )
            
            # Calculate returns (percentage changes)
            returns = spread_series.pct_change().dropna() * 100  # Convert to percentage
            
            if len(returns) < self.min_observations - 1:
                logger.warning("Insufficient returns data after transformation")
                return False
            
            # Check for constant returns (would cause GARCH to fail)
            if returns.std() == 0:
                logger.warning("Returns have zero variance, cannot fit GARCH")
                return False
            
            # Create and fit GARCH(1,1) model
            self.model = arch_model(
                returns,
                vol='GARCH',
                p=1,  # GARCH order
                q=1,  # ARCH order
                mean='Constant',
                dist='Normal'
            )
            
            # Fit with error handling
            self.fitted_model = self.model.fit(
                disp='off',  # Suppress output
                show_warning=False,
                options={'maxiter': 1000}
            )
            
            # Store model state
            params = self.fitted_model.params
            self.model_state = GARCHModelState(
                omega=params['omega'],
                alpha=params['alpha[1]'],
                beta=params['beta[1]'],
                log_likelihood=self.fitted_model.loglikelihood,
                aic=self.fitted_model.aic,
                bic=self.fitted_model.bic,
                fitted_at=datetime.utcnow(),
                n_observations=len(returns)
            )
            
            self.last_fit_time = datetime.utcnow()
            self.observations_since_fit = 0
            self.successful_fits += 1
            
            logger.info(
                "GARCH model fitted successfully",
                omega=self.model_state.omega,
                alpha=self.model_state.alpha,
                beta=self.model_state.beta,
                aic=self.model_state.aic,
                bic=self.model_state.bic,
                n_observations=self.model_state.n_observations
            )
            
            return True
            
        except Exception as e:
            self.failed_fits += 1
            logger.error(
                "Failed to fit GARCH model",
                error=str(e),
                data_size=len(self.spread_data),
                exc_info=True
            )
            return False
    
    def forecast_volatility(
        self, 
        horizon: int = 1,
        refit_if_needed: bool = True
    ) -> Optional[GARCHForecast]:
        """
        Generate volatility forecast using the fitted GARCH model.
        
        Args:
            horizon: Forecast horizon (number of periods ahead)
            refit_if_needed: Whether to refit model if needed
            
        Returns:
            GARCHForecast object or None if forecast failed
        """
        # Refit if needed
        if refit_if_needed and self.should_refit():
            if not self.fit_model():
                logger.warning("Could not refit model for forecasting")
                return None
        
        if self.fitted_model is None:
            logger.warning("No fitted model available for forecasting")
            return None
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
            
            # Extract forecast values
            variance_forecast = forecast.variance.iloc[-1, 0]  # Last forecast
            volatility_forecast = np.sqrt(variance_forecast)
            
            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            z_score = sm.stats.stattools.jarque_bera(
                self.fitted_model.resid
            )[0] if len(self.fitted_model.resid) > 0 else 1.96
            
            # Use normal approximation for confidence intervals
            z_critical = 1.96  # 95% confidence
            margin_of_error = z_critical * np.sqrt(variance_forecast / len(self.spread_data))
            
            confidence_lower = volatility_forecast - margin_of_error
            confidence_upper = volatility_forecast + margin_of_error
            
            result = GARCHForecast(
                volatility_forecast=float(volatility_forecast),
                variance_forecast=float(variance_forecast),
                confidence_interval_lower=float(max(0, confidence_lower)),  # Volatility can't be negative
                confidence_interval_upper=float(confidence_upper),
                forecast_horizon=horizon,
                model_aic=self.model_state.aic if self.model_state else np.nan,
                model_bic=self.model_state.bic if self.model_state else np.nan,
                timestamp=datetime.utcnow()
            )
            
            logger.debug(
                "Volatility forecast generated",
                volatility=result.volatility_forecast,
                variance=result.variance_forecast,
                confidence_lower=result.confidence_interval_lower,
                confidence_upper=result.confidence_interval_upper,
                horizon=horizon
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to generate volatility forecast",
                error=str(e),
                horizon=horizon,
                exc_info=True
            )
            return None
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the model."""
        diagnostics = {
            'total_observations': self.total_observations,
            'window_size': len(self.spread_data),
            'successful_fits': self.successful_fits,
            'failed_fits': self.failed_fits,
            'observations_since_fit': self.observations_since_fit,
            'last_fit_time': self.last_fit_time.isoformat() if self.last_fit_time else None,
            'model_fitted': self.fitted_model is not None
        }
        
        if self.model_state:
            diagnostics.update({
                'model_omega': self.model_state.omega,
                'model_alpha': self.model_state.alpha,
                'model_beta': self.model_state.beta,
                'model_aic': self.model_state.aic,
                'model_bic': self.model_state.bic,
                'model_persistence': self.model_state.alpha + self.model_state.beta,
                'model_fitted_at': self.model_state.fitted_at.isoformat()
            })
        
        return diagnostics
    
    def get_current_data(self) -> pd.DataFrame:
        """Get current data in the rolling window as DataFrame."""
        if not self.spread_data:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'timestamp': list(self.timestamps),
            'spread': list(self.spread_data)
        })
    
    def reset(self) -> None:
        """Reset the model state and clear all data."""
        self.spread_data.clear()
        self.timestamps.clear()
        self.model = None
        self.fitted_model = None
        self.last_fit_time = None
        self.observations_since_fit = 0
        self.model_state = None
        self.total_observations = 0
        self.successful_fits = 0
        self.failed_fits = 0
        
        logger.info("GARCH model reset")


class MultiPairGARCHManager:
    """
    Manages GARCH models for multiple trading pairs.
    """
    
    def __init__(
        self,
        default_window_size: int = 252,
        default_min_observations: int = 50,
        default_refit_frequency: int = 10
    ):
        """Initialize the multi-pair GARCH manager."""
        self.default_window_size = default_window_size
        self.default_min_observations = default_min_observations
        self.default_refit_frequency = default_refit_frequency
        
        self.models: Dict[str, RollingGARCHModel] = {}
        self.pair_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Multi-pair GARCH manager initialized")
    
    def add_pair(
        self,
        pair_id: str,
        window_size: Optional[int] = None,
        min_observations: Optional[int] = None,
        refit_frequency: Optional[int] = None
    ) -> None:
        """Add a new trading pair for GARCH modeling."""
        config = {
            'window_size': window_size or self.default_window_size,
            'min_observations': min_observations or self.default_min_observations,
            'refit_frequency': refit_frequency or self.default_refit_frequency
        }
        
        self.models[pair_id] = RollingGARCHModel(**config)
        self.pair_configs[pair_id] = config
        
        logger.info(
            "Added GARCH model for pair",
            pair_id=pair_id,
            **config
        )
    
    def add_observation(
        self,
        pair_id: str,
        spread_value: float,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Add observation to a specific pair's GARCH model."""
        if pair_id not in self.models:
            logger.warning("Unknown pair for GARCH observation", pair_id=pair_id)
            return False
        
        return self.models[pair_id].add_observation(spread_value, timestamp)
    
    def forecast_volatility(
        self,
        pair_id: str,
        horizon: int = 1
    ) -> Optional[GARCHForecast]:
        """Generate volatility forecast for a specific pair."""
        if pair_id not in self.models:
            logger.warning("Unknown pair for GARCH forecast", pair_id=pair_id)
            return None
        
        return self.models[pair_id].forecast_volatility(horizon)
    
    def get_all_forecasts(self, horizon: int = 1) -> Dict[str, Optional[GARCHForecast]]:
        """Get volatility forecasts for all pairs."""
        forecasts = {}
        for pair_id, model in self.models.items():
            forecasts[pair_id] = model.forecast_volatility(horizon)
        return forecasts
    
    def get_pair_diagnostics(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Get diagnostics for a specific pair."""
        if pair_id not in self.models:
            return None
        return self.models[pair_id].get_model_diagnostics()
    
    def get_all_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Get diagnostics for all pairs."""
        diagnostics = {}
        for pair_id, model in self.models.items():
            diagnostics[pair_id] = model.get_model_diagnostics()
        return diagnostics
