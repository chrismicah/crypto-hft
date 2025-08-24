"""
Adaptive strategy parameter adjustment based on market regime classification.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from dataclasses import asdict

from ..models.regime_models import (
    MarketRegime, RegimeClassification, StrategyParameters, 
    DEFAULT_STRATEGY_PARAMETERS, get_regime_risk_level,
    is_high_volatility_regime, is_trending_regime
)
from ...common.logger import get_logger


class StrategyParameterAdapter:
    """
    Adapts trading strategy parameters based on market regime classification.
    """
    
    def __init__(self, 
                 base_parameters: Optional[Dict[MarketRegime, StrategyParameters]] = None,
                 adaptation_speed: float = 0.1,
                 min_confidence_threshold: float = 0.5):
        
        self.logger = get_logger("strategy_adapter")
        
        # Use default parameters if none provided
        self.base_parameters = base_parameters or DEFAULT_STRATEGY_PARAMETERS.copy()
        
        # Adaptation settings
        self.adaptation_speed = adaptation_speed  # How quickly to adapt (0-1)
        self.min_confidence_threshold = min_confidence_threshold
        
        # Current parameters
        self.current_parameters: Optional[StrategyParameters] = None
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        
        # Parameter history
        self.parameter_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}
        
        # Adaptive learning
        self.parameter_adjustments: Dict[MarketRegime, Dict[str, float]] = {}
        self._initialize_adjustments()
    
    def _initialize_adjustments(self):
        """Initialize parameter adjustment factors."""
        for regime in MarketRegime:
            self.parameter_adjustments[regime] = {
                'entry_z_score_factor': 1.0,
                'exit_z_score_factor': 1.0,
                'position_size_factor': 1.0,
                'kelly_fraction_factor': 1.0,
                'execution_urgency_factor': 1.0
            }
    
    def adapt_parameters(self, 
                        regime_classification: RegimeClassification,
                        recent_performance: Optional[Dict[str, float]] = None) -> StrategyParameters:
        """
        Adapt strategy parameters based on current market regime.
        
        Args:
            regime_classification: Current regime classification
            recent_performance: Recent performance metrics
            
        Returns:
            Adapted strategy parameters
        """
        try:
            regime = regime_classification.regime
            confidence = regime_classification.confidence
            
            self.logger.info(f"Adapting parameters for regime: {regime.value} (confidence: {confidence:.3f})")
            
            # Get base parameters for this regime
            if regime in self.base_parameters:
                base_params = self.base_parameters[regime]
            else:
                # Fallback to unknown regime parameters
                base_params = self.base_parameters[MarketRegime.UNKNOWN]
            
            # Create adapted parameters
            adapted_params = StrategyParameters(
                regime=regime,
                entry_z_score=base_params.entry_z_score,
                exit_z_score=base_params.exit_z_score,
                stop_loss_z_score=base_params.stop_loss_z_score,
                max_position_size=base_params.max_position_size,
                position_sizing_factor=base_params.position_sizing_factor,
                kelly_fraction=base_params.kelly_fraction,
                max_drawdown_threshold=base_params.max_drawdown_threshold,
                volatility_scaling=base_params.volatility_scaling,
                correlation_threshold=base_params.correlation_threshold,
                execution_urgency=base_params.execution_urgency,
                slippage_tolerance=base_params.slippage_tolerance,
                holding_period_target=base_params.holding_period_target,
                rebalance_frequency=base_params.rebalance_frequency
            )
            
            # Apply confidence-based adjustments
            adapted_params = self._apply_confidence_adjustments(adapted_params, confidence)
            
            # Apply regime-specific adaptations
            adapted_params = self._apply_regime_adaptations(adapted_params, regime_classification)
            
            # Apply performance-based learning
            if recent_performance:
                adapted_params = self._apply_performance_learning(adapted_params, regime, recent_performance)
            
            # Apply transition-based adjustments
            if regime_classification.previous_regime and regime_classification.previous_regime != regime:
                adapted_params = self._apply_transition_adjustments(adapted_params, regime_classification)
            
            # Store current parameters
            self.current_parameters = adapted_params
            self.current_regime = regime
            self.regime_confidence = confidence
            
            # Record in history
            self._record_parameter_change(adapted_params, regime_classification)
            
            return adapted_params
            
        except Exception as e:
            self.logger.error(f"Error adapting parameters: {e}")
            # Return safe default parameters
            return self.base_parameters.get(MarketRegime.UNKNOWN, StrategyParameters(MarketRegime.UNKNOWN))
    
    def _apply_confidence_adjustments(self, 
                                    params: StrategyParameters, 
                                    confidence: float) -> StrategyParameters:
        """Apply adjustments based on classification confidence."""
        
        if confidence < self.min_confidence_threshold:
            # Low confidence - be more conservative
            confidence_factor = confidence / self.min_confidence_threshold
            
            # Reduce position sizes
            params.max_position_size *= (0.5 + 0.5 * confidence_factor)
            params.position_sizing_factor *= (0.7 + 0.3 * confidence_factor)
            params.kelly_fraction *= (0.6 + 0.4 * confidence_factor)
            
            # Widen entry thresholds (be more selective)
            params.entry_z_score *= (1.0 + 0.5 * (1 - confidence_factor))
            
            # Increase execution urgency (exit faster when uncertain)
            params.execution_urgency = min(1.0, params.execution_urgency * (1.0 + 0.3 * (1 - confidence_factor)))
            
            self.logger.info(f"Applied conservative adjustments for low confidence: {confidence:.3f}")
        
        return params
    
    def _apply_regime_adaptations(self, 
                                params: StrategyParameters, 
                                classification: RegimeClassification) -> StrategyParameters:
        """Apply regime-specific adaptations."""
        
        regime = classification.regime
        
        # High volatility regime adaptations
        if is_high_volatility_regime(regime):
            # Wider stops and more conservative sizing
            params.stop_loss_z_score *= 1.2
            params.max_position_size *= 0.8
            params.slippage_tolerance *= 1.5
            
            # Faster execution in volatile markets
            params.execution_urgency = min(1.0, params.execution_urgency * 1.3)
            
            # Shorter holding periods
            params.holding_period_target = int(params.holding_period_target * 0.7)
            params.rebalance_frequency = int(params.rebalance_frequency * 0.8)
        
        # Trending regime adaptations
        if is_trending_regime(regime):
            # Allow larger positions in trending markets
            params.max_position_size *= 1.1
            params.kelly_fraction *= 1.1
            
            # Wider exit thresholds to ride trends
            params.exit_z_score *= 0.8
            
            # Longer holding periods for trends
            params.holding_period_target = int(params.holding_period_target * 1.2)
        
        # Crisis regime special handling
        if regime == MarketRegime.CRISIS:
            # Extremely conservative
            params.max_position_size *= 0.3
            params.kelly_fraction *= 0.2
            params.entry_z_score *= 2.0
            params.execution_urgency = 0.95
            params.max_drawdown_threshold *= 0.5
        
        # Stable range regime optimizations
        elif regime == MarketRegime.STABLE_RANGE:
            # Can be more aggressive in stable conditions
            params.max_position_size *= 1.3
            params.kelly_fraction *= 1.2
            params.entry_z_score *= 0.8
            params.holding_period_target = int(params.holding_period_target * 1.5)
        
        return params
    
    def _apply_performance_learning(self, 
                                  params: StrategyParameters, 
                                  regime: MarketRegime,
                                  performance: Dict[str, float]) -> StrategyParameters:
        """Apply performance-based learning adjustments."""
        
        try:
            # Update regime performance tracking
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {}
            
            # Store recent performance
            for metric, value in performance.items():
                self.regime_performance[regime][metric] = value
            
            # Get performance metrics
            sharpe_ratio = performance.get('sharpe_ratio', 0.0)
            win_rate = performance.get('win_rate', 0.5)
            max_drawdown = performance.get('max_drawdown', 0.0)
            avg_return = performance.get('avg_return', 0.0)
            
            # Adjust parameters based on performance
            adjustments = self.parameter_adjustments[regime]
            
            # Sharpe ratio adjustments
            if sharpe_ratio > 1.5:
                # Good performance - can be slightly more aggressive
                adjustments['position_size_factor'] = min(1.2, adjustments['position_size_factor'] * 1.02)
                adjustments['kelly_fraction_factor'] = min(1.2, adjustments['kelly_fraction_factor'] * 1.01)
            elif sharpe_ratio < 0.5:
                # Poor performance - be more conservative
                adjustments['position_size_factor'] = max(0.5, adjustments['position_size_factor'] * 0.98)
                adjustments['entry_z_score_factor'] = min(1.5, adjustments['entry_z_score_factor'] * 1.01)
            
            # Win rate adjustments
            if win_rate > 0.6:
                # High win rate - can tighten exits
                adjustments['exit_z_score_factor'] = max(0.7, adjustments['exit_z_score_factor'] * 0.99)
            elif win_rate < 0.4:
                # Low win rate - widen exits
                adjustments['exit_z_score_factor'] = min(1.3, adjustments['exit_z_score_factor'] * 1.01)
            
            # Drawdown adjustments
            if max_drawdown > 0.1:  # 10% drawdown
                # High drawdown - reduce risk
                adjustments['position_size_factor'] = max(0.3, adjustments['position_size_factor'] * 0.95)
                adjustments['kelly_fraction_factor'] = max(0.3, adjustments['kelly_fraction_factor'] * 0.95)
            
            # Apply learned adjustments
            params.max_position_size *= adjustments['position_size_factor']
            params.kelly_fraction *= adjustments['kelly_fraction_factor']
            params.entry_z_score *= adjustments['entry_z_score_factor']
            params.exit_z_score *= adjustments['exit_z_score_factor']
            params.execution_urgency *= adjustments['execution_urgency_factor']
            
            # Ensure parameters stay within reasonable bounds
            params = self._enforce_parameter_bounds(params)
            
            self.logger.info(f"Applied performance learning for {regime.value}: Sharpe={sharpe_ratio:.2f}, WinRate={win_rate:.2f}")
            
        except Exception as e:
            self.logger.warning(f"Error applying performance learning: {e}")
        
        return params
    
    def _apply_transition_adjustments(self, 
                                    params: StrategyParameters, 
                                    classification: RegimeClassification) -> StrategyParameters:
        """Apply adjustments during regime transitions."""
        
        if not classification.previous_regime:
            return params
        
        transition_prob = classification.transition_probability
        
        # During uncertain transitions, be more conservative
        if transition_prob < 0.7:
            uncertainty_factor = 1.0 - transition_prob
            
            # Reduce position sizes during uncertain transitions
            params.max_position_size *= (1.0 - 0.3 * uncertainty_factor)
            params.position_sizing_factor *= (1.0 - 0.2 * uncertainty_factor)
            
            # Increase execution urgency to adapt quickly
            params.execution_urgency = min(1.0, params.execution_urgency * (1.0 + 0.2 * uncertainty_factor))
            
            # Shorter rebalance frequency during transitions
            params.rebalance_frequency = int(params.rebalance_frequency * (1.0 - 0.3 * uncertainty_factor))
            
            self.logger.info(f"Applied transition adjustments: uncertainty_factor={uncertainty_factor:.3f}")
        
        return params
    
    def _enforce_parameter_bounds(self, params: StrategyParameters) -> StrategyParameters:
        """Enforce reasonable bounds on parameters."""
        
        # Entry/exit thresholds
        params.entry_z_score = max(0.5, min(5.0, params.entry_z_score))
        params.exit_z_score = max(0.1, min(2.0, params.exit_z_score))
        params.stop_loss_z_score = max(2.0, min(10.0, params.stop_loss_z_score))
        
        # Position sizing
        params.max_position_size = max(0.1, min(2.0, params.max_position_size))
        params.position_sizing_factor = max(0.1, min(2.0, params.position_sizing_factor))
        params.kelly_fraction = max(0.01, min(0.5, params.kelly_fraction))
        
        # Risk management
        params.max_drawdown_threshold = max(0.01, min(0.2, params.max_drawdown_threshold))
        params.correlation_threshold = max(0.3, min(0.95, params.correlation_threshold))
        
        # Execution
        params.execution_urgency = max(0.1, min(1.0, params.execution_urgency))
        params.slippage_tolerance = max(0.0001, min(0.01, params.slippage_tolerance))
        
        # Timing
        params.holding_period_target = max(30, min(1440, params.holding_period_target))  # 30 min to 24 hours
        params.rebalance_frequency = max(15, min(360, params.rebalance_frequency))  # 15 min to 6 hours
        
        return params
    
    def _record_parameter_change(self, 
                               params: StrategyParameters, 
                               classification: RegimeClassification):
        """Record parameter changes in history."""
        
        record = {
            'timestamp': classification.timestamp.isoformat(),
            'regime': classification.regime.value,
            'confidence': classification.confidence,
            'parameters': params.to_dict(),
            'previous_regime': classification.previous_regime.value if classification.previous_regime else None,
            'transition_probability': classification.transition_probability
        }
        
        self.parameter_history.append(record)
        
        # Limit history size
        if len(self.parameter_history) > 1000:
            self.parameter_history = self.parameter_history[-500:]
    
    def get_current_parameters(self) -> Optional[StrategyParameters]:
        """Get current strategy parameters."""
        return self.current_parameters
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get summary of current parameters and recent changes."""
        
        if not self.current_parameters:
            return {'status': 'No parameters set'}
        
        # Recent parameter changes
        recent_changes = self.parameter_history[-10:] if self.parameter_history else []
        
        # Parameter comparison with base
        base_params = self.base_parameters.get(self.current_regime)
        parameter_deltas = {}
        
        if base_params:
            current_dict = self.current_parameters.to_dict()
            base_dict = base_params.to_dict()
            
            for key in current_dict:
                if key in base_dict and isinstance(current_dict[key], (int, float)):
                    current_val = current_dict[key]
                    base_val = base_dict[key]
                    if base_val != 0:
                        delta_pct = ((current_val - base_val) / base_val) * 100
                        parameter_deltas[key] = {
                            'current': current_val,
                            'base': base_val,
                            'delta_pct': delta_pct
                        }
        
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'current_parameters': self.current_parameters.to_dict(),
            'parameter_deltas': parameter_deltas,
            'recent_changes': recent_changes,
            'total_adaptations': len(self.parameter_history),
            'regime_performance': {k.value: v for k, v in self.regime_performance.items()}
        }
    
    def update_base_parameters(self, 
                             regime: MarketRegime, 
                             new_parameters: StrategyParameters):
        """Update base parameters for a specific regime."""
        
        self.base_parameters[regime] = new_parameters
        self.logger.info(f"Updated base parameters for regime: {regime.value}")
    
    def reset_learning(self, regime: Optional[MarketRegime] = None):
        """Reset performance learning adjustments."""
        
        if regime:
            # Reset specific regime
            self.parameter_adjustments[regime] = {
                'entry_z_score_factor': 1.0,
                'exit_z_score_factor': 1.0,
                'position_size_factor': 1.0,
                'kelly_fraction_factor': 1.0,
                'execution_urgency_factor': 1.0
            }
            if regime in self.regime_performance:
                del self.regime_performance[regime]
            
            self.logger.info(f"Reset learning for regime: {regime.value}")
        else:
            # Reset all regimes
            self._initialize_adjustments()
            self.regime_performance.clear()
            self.logger.info("Reset learning for all regimes")
    
    def get_regime_risk_assessment(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get risk assessment for a specific regime."""
        
        risk_level = get_regime_risk_level(regime)
        is_high_vol = is_high_volatility_regime(regime)
        is_trending = is_trending_regime(regime)
        
        # Get parameters for this regime
        params = self.base_parameters.get(regime)
        
        assessment = {
            'regime': regime.value,
            'risk_level': risk_level,
            'is_high_volatility': is_high_vol,
            'is_trending': is_trending,
            'recommended_max_position': params.max_position_size if params else 0.5,
            'recommended_kelly_fraction': params.kelly_fraction if params else 0.15,
            'execution_urgency': params.execution_urgency if params else 0.5
        }
        
        # Add performance data if available
        if regime in self.regime_performance:
            assessment['historical_performance'] = self.regime_performance[regime]
        
        return assessment
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for backup/sharing."""
        
        return {
            'base_parameters': {k.value: v.to_dict() for k, v in self.base_parameters.items()},
            'parameter_adjustments': {k.value: v for k, v in self.parameter_adjustments.items()},
            'adaptation_speed': self.adaptation_speed,
            'min_confidence_threshold': self.min_confidence_threshold,
            'regime_performance': {k.value: v for k, v in self.regime_performance.items()},
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import configuration from backup."""
        
        try:
            # Import base parameters
            if 'base_parameters' in config:
                self.base_parameters = {}
                for regime_str, params_dict in config['base_parameters'].items():
                    regime = MarketRegime(regime_str)
                    params = StrategyParameters(regime=regime, **params_dict)
                    self.base_parameters[regime] = params
            
            # Import adjustments
            if 'parameter_adjustments' in config:
                self.parameter_adjustments = {}
                for regime_str, adjustments in config['parameter_adjustments'].items():
                    regime = MarketRegime(regime_str)
                    self.parameter_adjustments[regime] = adjustments
            
            # Import settings
            if 'adaptation_speed' in config:
                self.adaptation_speed = config['adaptation_speed']
            
            if 'min_confidence_threshold' in config:
                self.min_confidence_threshold = config['min_confidence_threshold']
            
            # Import performance data
            if 'regime_performance' in config:
                self.regime_performance = {}
                for regime_str, perf_data in config['regime_performance'].items():
                    regime = MarketRegime(regime_str)
                    self.regime_performance[regime] = perf_data
            
            self.logger.info("Configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            raise
