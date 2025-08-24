"""
Integration module for regime-aware execution.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import asdict
import numpy as np

from ..regime_classifier.models.regime_models import (
    MarketRegime, RegimeClassification, StrategyParameters,
    RegimeAlert, get_regime_risk_level
)
from ..common.logger import get_logger


class RegimeAwareExecutionManager:
    """
    Manages execution strategy based on market regime classifications.
    """
    
    def __init__(self, 
                 execution_service,  # Reference to main execution service
                 regime_update_callback: Optional[Callable] = None,
                 alert_callback: Optional[Callable] = None):
        
        self.execution_service = execution_service
        self.regime_update_callback = regime_update_callback
        self.alert_callback = alert_callback
        
        self.logger = get_logger("regime_execution_manager")
        
        # Current regime state
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self.current_parameters: Optional[StrategyParameters] = None
        self.regime_confidence: float = 0.0
        self.last_regime_update: Optional[datetime] = None
        
        # Regime history for analysis
        self.regime_history: List[RegimeClassification] = []
        self.alert_history: List[RegimeAlert] = []
        
        # Risk management state
        self.regime_risk_level: str = "medium"
        self.emergency_halt: bool = False
        self.position_scaling_factor: float = 1.0
        
        # Performance tracking by regime
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}
        
        # Configuration
        self.min_confidence_threshold: float = 0.5
        self.emergency_regimes = {MarketRegime.CRISIS}
        self.high_risk_regimes = {
            MarketRegime.HIGH_VOL_BEAR, 
            MarketRegime.HIGH_VOL_RANGE,
            MarketRegime.CRISIS,
            MarketRegime.RECOVERY
        }
    
    async def process_regime_classification(self, classification_data: Dict[str, Any]):
        """
        Process incoming regime classification and update execution parameters.
        
        Args:
            classification_data: Regime classification data from Kafka
        """
        try:
            # Parse classification data
            regime = MarketRegime(classification_data['regime'])
            confidence = float(classification_data['confidence'])
            timestamp = datetime.fromisoformat(classification_data['timestamp'])
            
            # Create classification object
            classification = RegimeClassification(
                timestamp=timestamp,
                regime=regime,
                confidence=confidence,
                confidence_level=classification_data.get('confidence_level', 'medium'),
                regime_probabilities={
                    MarketRegime(k): v for k, v in classification_data.get('regime_probabilities', {}).items()
                },
                previous_regime=MarketRegime(classification_data['previous_regime']) if classification_data.get('previous_regime') else None,
                transition_probability=classification_data.get('transition_probability', 0.0)
            )
            
            # Update regime state
            await self._update_regime_state(classification)
            
            # Process strategy parameters if provided
            if 'parameters' in classification_data:
                await self._update_strategy_parameters(classification_data['parameters'])
            
            # Check for regime-based alerts
            await self._check_regime_alerts(classification)
            
            # Update execution service
            if self.regime_update_callback:
                await self.regime_update_callback(classification, self.current_parameters)
            
            self.logger.info(f"Processed regime classification: {regime.value} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error processing regime classification: {e}")
    
    async def process_regime_alert(self, alert_data: Dict[str, Any]):
        """
        Process incoming regime alert and take appropriate action.
        
        Args:
            alert_data: Regime alert data from Kafka
        """
        try:
            # Parse alert data
            alert = RegimeAlert(
                timestamp=datetime.fromisoformat(alert_data['timestamp']),
                alert_type=alert_data['alert_type'],
                regime=MarketRegime(alert_data['regime']),
                previous_regime=MarketRegime(alert_data['previous_regime']) if alert_data.get('previous_regime') else None,
                confidence=float(alert_data['confidence']),
                message=alert_data['message'],
                severity=alert_data['severity'],
                suggested_actions=alert_data.get('suggested_actions', []),
                expected_impact=alert_data.get('expected_impact', 'unknown'),
                risk_level=alert_data.get('risk_level', 'medium')
            )
            
            # Store alert
            self.alert_history.append(alert)
            
            # Take action based on alert severity
            await self._handle_regime_alert(alert)
            
            # Notify callback
            if self.alert_callback:
                await self.alert_callback(alert)
            
            self.logger.warning(f"Processed regime alert: {alert.alert_type} - {alert.severity}")
            
        except Exception as e:
            self.logger.error(f"Error processing regime alert: {e}")
    
    async def _update_regime_state(self, classification: RegimeClassification):
        """Update internal regime state."""
        
        # Check if regime changed
        regime_changed = classification.regime != self.current_regime
        
        # Update state
        previous_regime = self.current_regime
        self.current_regime = classification.regime
        self.regime_confidence = classification.confidence
        self.last_regime_update = classification.timestamp
        self.regime_risk_level = get_regime_risk_level(classification.regime)
        
        # Store in history
        self.regime_history.append(classification)
        
        # Limit history size
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]
        
        # Log regime change
        if regime_changed and previous_regime != MarketRegime.UNKNOWN:
            self.logger.info(f"Regime changed: {previous_regime.value} -> {classification.regime.value}")
            
            # Update performance tracking
            await self._update_regime_performance(previous_regime)
    
    async def _update_strategy_parameters(self, parameters_data: Dict[str, Any]):
        """Update strategy parameters based on regime."""
        
        try:
            # Create StrategyParameters object
            self.current_parameters = StrategyParameters(
                regime=self.current_regime,
                entry_z_score=parameters_data.get('entry_z_score', 2.0),
                exit_z_score=parameters_data.get('exit_z_score', 0.5),
                stop_loss_z_score=parameters_data.get('stop_loss_z_score', 4.0),
                max_position_size=parameters_data.get('max_position_size', 1.0),
                position_sizing_factor=parameters_data.get('position_sizing_factor', 1.0),
                kelly_fraction=parameters_data.get('kelly_fraction', 0.25),
                max_drawdown_threshold=parameters_data.get('max_drawdown_threshold', 0.05),
                volatility_scaling=parameters_data.get('volatility_scaling', True),
                correlation_threshold=parameters_data.get('correlation_threshold', 0.7),
                execution_urgency=parameters_data.get('execution_urgency', 0.5),
                slippage_tolerance=parameters_data.get('slippage_tolerance', 0.001),
                holding_period_target=parameters_data.get('holding_period_target', 240),
                rebalance_frequency=parameters_data.get('rebalance_frequency', 60)
            )
            
            # Calculate position scaling based on regime risk
            self.position_scaling_factor = self._calculate_position_scaling()
            
            self.logger.info(f"Updated strategy parameters for regime: {self.current_regime.value}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy parameters: {e}")
    
    async def _check_regime_alerts(self, classification: RegimeClassification):
        """Check for regime-based alerts and conditions."""
        
        # Low confidence alert
        if classification.confidence < self.min_confidence_threshold:
            alert = RegimeAlert(
                timestamp=classification.timestamp,
                alert_type="low_confidence",
                regime=classification.regime,
                confidence=classification.confidence,
                message=f"Low confidence regime classification: {classification.confidence:.3f}",
                severity="medium",
                suggested_actions=["Reduce position sizes", "Increase monitoring"],
                risk_level="medium"
            )
            await self._handle_regime_alert(alert)
        
        # Emergency regime alert
        if classification.regime in self.emergency_regimes:
            alert = RegimeAlert(
                timestamp=classification.timestamp,
                alert_type="emergency_regime",
                regime=classification.regime,
                confidence=classification.confidence,
                message=f"EMERGENCY: {classification.regime.value} regime detected",
                severity="critical",
                suggested_actions=[
                    "HALT all new positions",
                    "Reduce existing positions",
                    "Activate emergency protocols"
                ],
                risk_level="critical"
            )
            await self._handle_regime_alert(alert)
    
    async def _handle_regime_alert(self, alert: RegimeAlert):
        """Handle regime alert by taking appropriate action."""
        
        try:
            if alert.severity == "critical":
                # Critical alerts - immediate action required
                if alert.alert_type == "emergency_regime" or alert.regime == MarketRegime.CRISIS:
                    self.emergency_halt = True
                    self.position_scaling_factor = 0.1  # Reduce to 10% of normal size
                    
                    # Notify execution service to halt trading
                    if hasattr(self.execution_service, 'emergency_halt'):
                        await self.execution_service.emergency_halt("Regime-based emergency halt")
                    
                    self.logger.critical(f"EMERGENCY HALT activated due to: {alert.message}")
            
            elif alert.severity == "high":
                # High severity - significant risk reduction
                self.position_scaling_factor = min(self.position_scaling_factor, 0.5)
                self.logger.warning(f"High severity alert: {alert.message}")
            
            elif alert.severity == "medium":
                # Medium severity - moderate adjustments
                if alert.alert_type == "low_confidence":
                    self.position_scaling_factor = min(self.position_scaling_factor, 0.8)
                self.logger.info(f"Medium severity alert: {alert.message}")
            
            # Store alert
            self.alert_history.append(alert)
            
            # Limit alert history
            if len(self.alert_history) > 500:
                self.alert_history = self.alert_history[-250:]
                
        except Exception as e:
            self.logger.error(f"Error handling regime alert: {e}")
    
    def _calculate_position_scaling(self) -> float:
        """Calculate position scaling factor based on current regime."""
        
        base_scaling = 1.0
        
        # Regime-based scaling
        if self.current_regime == MarketRegime.CRISIS:
            base_scaling *= 0.1  # 10% of normal size
        elif self.current_regime in self.high_risk_regimes:
            base_scaling *= 0.5  # 50% of normal size
        elif self.current_regime == MarketRegime.STABLE_RANGE:
            base_scaling *= 1.2  # 120% of normal size
        
        # Confidence-based scaling
        if self.regime_confidence < 0.5:
            base_scaling *= 0.7  # Reduce when uncertain
        elif self.regime_confidence > 0.8:
            base_scaling *= 1.1  # Slightly increase when confident
        
        # Emergency halt override
        if self.emergency_halt:
            base_scaling = min(base_scaling, 0.1)
        
        return max(0.05, min(2.0, base_scaling))  # Clamp between 5% and 200%
    
    async def _update_regime_performance(self, previous_regime: MarketRegime):
        """Update performance tracking for the previous regime."""
        
        try:
            # This would typically query recent trading performance
            # For now, we'll use placeholder logic
            
            if previous_regime not in self.regime_performance:
                self.regime_performance[previous_regime] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'avg_holding_time': 0.0
                }
            
            # In a real implementation, this would query the database
            # for trades during the previous regime period
            
        except Exception as e:
            self.logger.error(f"Error updating regime performance: {e}")
    
    def get_current_regime_info(self) -> Dict[str, Any]:
        """Get current regime information."""
        
        return {
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'risk_level': self.regime_risk_level,
            'last_update': self.last_regime_update.isoformat() if self.last_regime_update else None,
            'position_scaling_factor': self.position_scaling_factor,
            'emergency_halt': self.emergency_halt,
            'parameters': self.current_parameters.to_dict() if self.current_parameters else None
        }
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime classification statistics."""
        
        if not self.regime_history:
            return {'message': 'No regime history available'}
        
        # Count regime occurrences
        regime_counts = {}
        total_classifications = len(self.regime_history)
        
        for classification in self.regime_history:
            regime = classification.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate percentages
        regime_percentages = {
            regime: (count / total_classifications) * 100
            for regime, count in regime_counts.items()
        }
        
        # Recent regime changes
        recent_changes = []
        for i in range(1, min(10, len(self.regime_history))):
            current = self.regime_history[-i]
            previous = self.regime_history[-i-1]
            
            if current.regime != previous.regime:
                recent_changes.append({
                    'timestamp': current.timestamp.isoformat(),
                    'from': previous.regime.value,
                    'to': current.regime.value,
                    'confidence': current.confidence
                })
        
        return {
            'total_classifications': total_classifications,
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'recent_changes': recent_changes,
            'current_regime': self.current_regime.value,
            'average_confidence': np.mean([c.confidence for c in self.regime_history]),
            'alert_count': len(self.alert_history)
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent regime alerts."""
        
        recent_alerts = self.alert_history[-limit:] if self.alert_history else []
        return [alert.to_dict() for alert in recent_alerts]
    
    def should_allow_new_position(self, proposed_size: float) -> Tuple[bool, float, str]:
        """
        Check if a new position should be allowed based on regime.
        
        Args:
            proposed_size: Proposed position size
            
        Returns:
            Tuple of (allowed, adjusted_size, reason)
        """
        
        # Emergency halt check
        if self.emergency_halt:
            return False, 0.0, "Emergency halt active due to regime conditions"
        
        # Apply position scaling
        adjusted_size = proposed_size * self.position_scaling_factor
        
        # Check minimum size threshold
        if adjusted_size < 0.01:  # Minimum 1% of proposed size
            return False, 0.0, f"Position size too small after regime scaling ({self.position_scaling_factor:.2f})"
        
        # Regime-specific checks
        if self.current_regime == MarketRegime.CRISIS:
            return False, 0.0, "No new positions allowed in crisis regime"
        
        if self.current_regime in self.high_risk_regimes and self.regime_confidence < 0.6:
            return False, 0.0, f"High risk regime with low confidence: {self.regime_confidence:.3f}"
        
        reason = f"Position scaled by {self.position_scaling_factor:.2f} due to {self.current_regime.value} regime"
        return True, adjusted_size, reason
    
    def get_execution_parameters(self) -> Dict[str, Any]:
        """Get current execution parameters based on regime."""
        
        if not self.current_parameters:
            return {}
        
        return {
            'entry_z_score': self.current_parameters.entry_z_score,
            'exit_z_score': self.current_parameters.exit_z_score,
            'stop_loss_z_score': self.current_parameters.stop_loss_z_score,
            'execution_urgency': self.current_parameters.execution_urgency,
            'slippage_tolerance': self.current_parameters.slippage_tolerance,
            'position_scaling_factor': self.position_scaling_factor,
            'max_position_size': self.current_parameters.max_position_size * self.position_scaling_factor,
            'rebalance_frequency': self.current_parameters.rebalance_frequency
        }
    
    async def reset_emergency_halt(self, reason: str = "Manual reset"):
        """Reset emergency halt status."""
        
        if self.emergency_halt:
            self.emergency_halt = False
            self.position_scaling_factor = self._calculate_position_scaling()
            
            self.logger.info(f"Emergency halt reset: {reason}")
            
            # Notify execution service
            if hasattr(self.execution_service, 'resume_trading'):
                await self.execution_service.resume_trading(reason)
    
    def export_regime_data(self) -> Dict[str, Any]:
        """Export regime data for analysis."""
        
        return {
            'current_state': self.get_current_regime_info(),
            'statistics': self.get_regime_statistics(),
            'recent_alerts': self.get_recent_alerts(20),
            'regime_history': [
                {
                    'timestamp': c.timestamp.isoformat(),
                    'regime': c.regime.value,
                    'confidence': c.confidence
                }
                for c in self.regime_history[-100:]  # Last 100 classifications
            ],
            'performance_by_regime': {
                regime.value: performance 
                for regime, performance in self.regime_performance.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
