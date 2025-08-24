"""
Main service for market regime classification and adaptive strategy management.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .models.regime_models import (
    MarketRegime, RegimeClassification, RegimeAlert, RegimeModelConfig,
    MarketFeatures, StrategyParameters
)
from .features.feature_extractor import FeatureExtractor
from .classifiers.hmm_classifier import HMMRegimeClassifier
from .adapters.strategy_adapter import StrategyParameterAdapter
from ..common.db.client import DatabaseClient
from ..common.logger import get_logger, configure_logging
from ..common.metrics import ServiceMetrics


class RegimeClassificationService:
    """
    Main service for market regime classification and adaptive strategy management.
    """
    
    def __init__(self,
                 db_client: DatabaseClient,
                 config: Optional[RegimeModelConfig] = None,
                 classification_interval: int = 300,  # 5 minutes
                 feature_lookback_hours: int = 168):  # 1 week
        
        self.db_client = db_client
        self.config = config or RegimeModelConfig()
        self.classification_interval = classification_interval
        self.feature_lookback_hours = feature_lookback_hours
        
        self.logger = get_logger("regime_classifier_service")
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.hmm_classifier = HMMRegimeClassifier(self.config)
        self.strategy_adapter = StrategyParameterAdapter()
        
        # Service state
        self.is_running = False
        self.current_regime = MarketRegime.UNKNOWN
        self.current_parameters: Optional[StrategyParameters] = None
        
        # Data storage
        self.classification_history: List[RegimeClassification] = []
        self.alert_history: List[RegimeAlert] = []
        
        # Metrics
        self.metrics = ServiceMetrics("regime_classifier")
        
        # Performance tracking
        self.performance_stats = {
            'total_classifications': 0,
            'regime_changes': 0,
            'alerts_generated': 0,
            'avg_classification_time': 0.0,
            'model_accuracy': 0.0
        }
        
        # Model management
        self.model_path = "models/regime_classifier.pkl"
        self.last_training_time: Optional[datetime] = None
        self.training_data_cache: List[Tuple[np.ndarray, datetime]] = []
    
    async def start(self):
        """Start the regime classification service."""
        self.logger.info("Starting Regime Classification Service")
        self.is_running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Load existing model if available
            await self._load_model()
            
            # Start main classification loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the regime classification service."""
        self.logger.info("Stopping Regime Classification Service")
        self.is_running = False
        
        # Save current state
        await self._save_state()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def _main_loop(self):
        """Main classification and adaptation loop."""
        self.logger.info("Starting main classification loop")
        
        while self.is_running:
            try:
                # Run classification cycle
                await self._run_classification_cycle()
                
                # Check if model needs retraining
                await self._check_model_retraining()
                
                # Wait for next cycle
                await asyncio.sleep(self.classification_interval)
                
            except Exception as e:
                self.logger.error(f"Error in classification cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _run_classification_cycle(self):
        """Run a complete classification and adaptation cycle."""
        cycle_start = datetime.now()
        
        try:
            self.logger.info(f"Starting classification cycle at {cycle_start}")
            
            # Collect market data
            market_data = await self._collect_market_data()
            
            if not market_data:
                self.logger.warning("No market data available, skipping cycle")
                return
            
            # Extract features
            features = self._extract_features(market_data)
            
            if features is None:
                self.logger.warning("Feature extraction failed, skipping cycle")
                return
            
            # Classify regime
            classification = await self._classify_regime(features, cycle_start)
            
            # Adapt strategy parameters
            performance_data = await self._get_recent_performance()
            adapted_parameters = self._adapt_strategy_parameters(classification, performance_data)
            
            # Check for alerts
            alerts = self._check_for_alerts(classification)
            
            # Store results
            await self._store_results(classification, adapted_parameters, alerts)
            
            # Update metrics
            self._update_metrics(classification, cycle_start)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"Classification cycle completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in classification cycle: {e}")
            self.metrics.record_error("classification_cycle_error")
    
    async def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """Collect market data for feature extraction."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.feature_lookback_hours)
            
            # Query price data from database
            price_query = """
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM market_data
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """
            
            with self.db_client.get_session() as session:
                price_data = session.execute(price_query, (start_time, end_time)).fetchall()
            
            if not price_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data, columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Get primary trading pair data
            primary_symbol = 'BTCUSDT'  # Default to BTC
            symbol_data = df[df['symbol'] == primary_symbol].copy()
            
            if symbol_data.empty:
                # Fallback to any available symbol
                available_symbols = df['symbol'].unique()
                if len(available_symbols) > 0:
                    primary_symbol = available_symbols[0]
                    symbol_data = df[df['symbol'] == primary_symbol].copy()
                else:
                    return None
            
            # Collect additional data
            market_data = {
                'price_data': symbol_data,
                'symbol': primary_symbol,
                'external_data': await self._collect_external_data()
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            return None
    
    async def _collect_external_data(self) -> Dict[str, Any]:
        """Collect external data (funding rates, sentiment, etc.)."""
        external_data = {}
        
        try:
            # Get funding rates (placeholder - would integrate with exchange APIs)
            external_data['funding_rate'] = 0.0001  # 0.01%
            external_data['open_interest_change'] = 0.0
            
            # Get sentiment data (placeholder - would integrate with sentiment APIs)
            external_data['fear_greed_index'] = 50.0
            external_data['social_sentiment'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error collecting external data: {e}")
        
        return external_data
    
    def _extract_features(self, market_data: Dict[str, Any]) -> Optional[MarketFeatures]:
        """Extract features from market data."""
        try:
            price_data = market_data['price_data']
            external_data = market_data.get('external_data', {})
            
            # Extract features using feature extractor
            features = self.feature_extractor.extract_features(
                price_data=price_data,
                external_data=external_data
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    async def _classify_regime(self, features: MarketFeatures, timestamp: datetime) -> RegimeClassification:
        """Classify market regime using HMM model."""
        try:
            # Convert features to array
            feature_array = features.to_array()
            
            # Classify using HMM
            classification = self.hmm_classifier.predict(feature_array, timestamp)
            
            # Add features to classification
            classification.features = features
            
            # Store training data for future retraining
            self.training_data_cache.append((feature_array, timestamp))
            
            # Limit cache size
            if len(self.training_data_cache) > 10000:
                self.training_data_cache = self.training_data_cache[-5000:]
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            return RegimeClassification(
                timestamp=timestamp,
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                confidence_level=get_confidence_level(0.0)
            )
    
    def _adapt_strategy_parameters(self, 
                                 classification: RegimeClassification,
                                 performance_data: Optional[Dict[str, float]]) -> StrategyParameters:
        """Adapt strategy parameters based on regime classification."""
        try:
            adapted_params = self.strategy_adapter.adapt_parameters(
                classification, performance_data
            )
            
            self.current_parameters = adapted_params
            self.current_regime = classification.regime
            
            return adapted_params
            
        except Exception as e:
            self.logger.error(f"Error adapting strategy parameters: {e}")
            # Return safe default parameters
            return StrategyParameters(regime=MarketRegime.UNKNOWN)
    
    async def _get_recent_performance(self) -> Optional[Dict[str, float]]:
        """Get recent trading performance metrics."""
        try:
            # Query recent trading performance
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)  # Last 24 hours
            
            performance_query = """
            SELECT 
                AVG(pnl) as avg_return,
                STDDEV(pnl) as return_std,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) * 1.0 / COUNT(*) as win_rate,
                MIN(cumulative_pnl) as max_drawdown
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
            """
            
            with self.db_client.get_session() as session:
                result = session.execute(performance_query, (start_time, end_time)).fetchone()
            
            if result and result[0] is not None:
                avg_return = float(result[0])
                return_std = float(result[1]) if result[1] else 0.0
                win_rate = float(result[2]) if result[2] else 0.5
                max_drawdown = abs(float(result[3])) if result[3] else 0.0
                
                # Calculate Sharpe ratio
                sharpe_ratio = avg_return / return_std if return_std > 0 else 0.0
                
                return {
                    'avg_return': avg_return,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting recent performance: {e}")
            return None
    
    def _check_for_alerts(self, classification: RegimeClassification) -> List[RegimeAlert]:
        """Check for regime-based alerts."""
        alerts = []
        
        try:
            # Regime change alert
            if (classification.regime != self.current_regime and 
                self.current_regime != MarketRegime.UNKNOWN):
                
                alert = RegimeAlert(
                    timestamp=classification.timestamp,
                    alert_type="regime_change",
                    regime=classification.regime,
                    previous_regime=self.current_regime,
                    confidence=classification.confidence,
                    message=f"Market regime changed from {self.current_regime.value} to {classification.regime.value}",
                    severity="medium" if classification.confidence > 0.7 else "low",
                    suggested_actions=[
                        f"Review strategy parameters for {classification.regime.value} regime",
                        "Monitor position sizes and risk exposure",
                        "Consider rebalancing portfolio"
                    ],
                    expected_impact="medium",
                    risk_level=get_regime_risk_level(classification.regime)
                )
                alerts.append(alert)
            
            # High volatility alert
            if classification.regime in [MarketRegime.HIGH_VOL_BULL, MarketRegime.HIGH_VOL_BEAR, 
                                       MarketRegime.HIGH_VOL_RANGE, MarketRegime.CRISIS]:
                
                alert = RegimeAlert(
                    timestamp=classification.timestamp,
                    alert_type="high_volatility",
                    regime=classification.regime,
                    previous_regime=None,
                    confidence=classification.confidence,
                    message=f"High volatility regime detected: {classification.regime.value}",
                    severity="high" if classification.regime == MarketRegime.CRISIS else "medium",
                    suggested_actions=[
                        "Reduce position sizes",
                        "Widen stop-loss levels",
                        "Increase monitoring frequency",
                        "Consider hedging strategies"
                    ],
                    expected_impact="high",
                    risk_level="high"
                )
                alerts.append(alert)
            
            # Crisis detection alert
            if classification.regime == MarketRegime.CRISIS:
                alert = RegimeAlert(
                    timestamp=classification.timestamp,
                    alert_type="crisis_detected",
                    regime=classification.regime,
                    previous_regime=self.current_regime,
                    confidence=classification.confidence,
                    message="CRISIS REGIME DETECTED - Extreme market conditions",
                    severity="critical",
                    suggested_actions=[
                        "IMMEDIATELY reduce all position sizes to minimum",
                        "Halt new position entries",
                        "Activate emergency risk protocols",
                        "Monitor liquidity conditions closely",
                        "Prepare for potential system shutdown"
                    ],
                    expected_impact="critical",
                    risk_level="critical"
                )
                alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking for alerts: {e}")
        
        return alerts
    
    async def _store_results(self, 
                           classification: RegimeClassification,
                           parameters: StrategyParameters,
                           alerts: List[RegimeAlert]):
        """Store classification results and alerts."""
        try:
            # Store classification in history
            self.classification_history.append(classification)
            
            # Store alerts
            for alert in alerts:
                self.alert_history.append(alert)
                self.logger.warning(f"REGIME ALERT: {alert.message}")
            
            # Limit history sizes
            if len(self.classification_history) > 10000:
                self.classification_history = self.classification_history[-5000:]
            
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
            
            # Store in database (optional - could implement regime_classifications table)
            # await self._store_in_database(classification, parameters, alerts)
            
        except Exception as e:
            self.logger.error(f"Error storing results: {e}")
    
    def _update_metrics(self, classification: RegimeClassification, cycle_start: datetime):
        """Update service metrics."""
        try:
            # Update performance stats
            self.performance_stats['total_classifications'] += 1
            
            if classification.regime != self.current_regime:
                self.performance_stats['regime_changes'] += 1
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            total_classifications = self.performance_stats['total_classifications']
            
            # Update average classification time
            current_avg = self.performance_stats['avg_classification_time']
            new_avg = ((current_avg * (total_classifications - 1)) + cycle_duration) / total_classifications
            self.performance_stats['avg_classification_time'] = new_avg
            
            # Record metrics
            self.metrics.record_request("classification_cycle")
            self.metrics.record_latency("classification_time", cycle_duration)
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    async def _check_model_retraining(self):
        """Check if model needs retraining."""
        try:
            if not self.last_training_time:
                # No previous training - check if we have enough data
                if len(self.training_data_cache) >= 1000:
                    await self._retrain_model()
                return
            
            # Check if it's time for scheduled retraining
            time_since_training = datetime.now() - self.last_training_time
            retrain_interval = timedelta(hours=self.config.retrain_frequency)
            
            if time_since_training >= retrain_interval:
                await self._retrain_model()
            
        except Exception as e:
            self.logger.error(f"Error checking model retraining: {e}")
    
    async def _retrain_model(self):
        """Retrain the HMM model with recent data."""
        try:
            if len(self.training_data_cache) < 100:
                self.logger.warning("Insufficient data for retraining")
                return
            
            self.logger.info(f"Retraining HMM model with {len(self.training_data_cache)} samples")
            
            # Prepare training data
            feature_arrays = [data[0] for data in self.training_data_cache]
            timestamps = [data[1] for data in self.training_data_cache]
            
            # Retrain model
            performance_metrics = self.hmm_classifier.train(feature_arrays, timestamps)
            
            # Update training time
            self.last_training_time = datetime.now()
            
            # Save updated model
            await self._save_model()
            
            self.logger.info(f"Model retraining completed. Performance: {performance_metrics.to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
    
    async def _load_model(self):
        """Load existing model if available."""
        try:
            if os.path.exists(self.model_path):
                self.hmm_classifier.load_model(self.model_path)
                self.logger.info(f"Loaded existing model from {self.model_path}")
            else:
                self.logger.info("No existing model found, will train when sufficient data is available")
                
        except Exception as e:
            self.logger.warning(f"Error loading model: {e}")
    
    async def _save_model(self):
        """Save current model."""
        try:
            if self.hmm_classifier.is_trained:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.hmm_classifier.save_model(self.model_path)
                self.logger.info(f"Model saved to {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    async def _save_state(self):
        """Save current service state."""
        try:
            state = {
                'performance_stats': self.performance_stats,
                'current_regime': self.current_regime.value,
                'classification_history_count': len(self.classification_history),
                'alert_history_count': len(self.alert_history),
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_data_cache_size': len(self.training_data_cache),
                'model_info': self.hmm_classifier.get_model_info(),
                'strategy_summary': self.strategy_adapter.get_parameter_summary()
            }
            
            # Save to file
            state_file = "regime_classifier_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Service state saved to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    # API methods for external interaction
    
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current market regime information."""
        if self.classification_history:
            latest = self.classification_history[-1]
            return {
                'regime': latest.regime.value,
                'confidence': latest.confidence,
                'confidence_level': latest.confidence_level.value,
                'timestamp': latest.timestamp.isoformat(),
                'regime_probabilities': {k.value: v for k, v in latest.regime_probabilities.items()},
                'regime_duration': latest.regime_duration.total_seconds() if latest.regime_duration else 0
            }
        else:
            return {
                'regime': MarketRegime.UNKNOWN.value,
                'confidence': 0.0,
                'confidence_level': 'very_low',
                'timestamp': datetime.now().isoformat(),
                'regime_probabilities': {},
                'regime_duration': 0
            }
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        if self.current_parameters:
            return self.current_parameters.to_dict()
        else:
            return StrategyParameters(regime=MarketRegime.UNKNOWN).to_dict()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent regime alerts."""
        recent_alerts = self.alert_history[-limit:] if self.alert_history else []
        return [alert.to_dict() for alert in recent_alerts]
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            'is_running': self.is_running,
            'classification_interval': self.classification_interval,
            'current_regime': self.current_regime.value,
            'model_trained': self.hmm_classifier.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'performance_stats': self.performance_stats,
            'classification_history_count': len(self.classification_history),
            'alert_history_count': len(self.alert_history),
            'training_data_cache_size': len(self.training_data_cache)
        }
    
    async def force_classification(self) -> Dict[str, Any]:
        """Force immediate regime classification."""
        try:
            self.logger.info("Forcing immediate regime classification")
            
            # Collect data and classify
            market_data = await self._collect_market_data()
            if not market_data:
                return {'error': 'No market data available'}
            
            features = self._extract_features(market_data)
            if not features:
                return {'error': 'Feature extraction failed'}
            
            classification = await self._classify_regime(features, datetime.now())
            
            # Adapt parameters
            performance_data = await self._get_recent_performance()
            parameters = self._adapt_strategy_parameters(classification, performance_data)
            
            return {
                'classification': classification.to_dict(),
                'parameters': parameters.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in forced classification: {e}")
            return {'error': str(e)}


async def main():
    """Main entry point for the regime classification service."""
    # Configure logging
    configure_logging()
    logger = get_logger("main")
    
    try:
        # Initialize database client
        db_client = DatabaseClient()
        await db_client.initialize_db()
        
        # Create and start service
        config = RegimeModelConfig(
            n_components=6,
            training_window=2160,  # 90 days
            retrain_frequency=168  # Weekly
        )
        
        service = RegimeClassificationService(
            db_client=db_client,
            config=config,
            classification_interval=300  # 5 minutes
        )
        
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
