"""
Champion-Challenger Management Logic for EVOP Framework.

This module implements the core logic for managing parallel strategy execution,
including challenger generation, performance monitoring, and promotion decisions.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import numpy as np
import json

from .models import (
    StrategyInstance, StrategyParameters, StrategyStatus, 
    PromotionEvent, PromotionReason, EVOPConfiguration,
    PerformanceMetrics
)
from .performance_comparator import PerformanceComparator, ComparisonResult


class ChampionChallengerManager:
    """
    Manages the champion-challenger framework for evolutionary strategy optimization.
    
    This class handles:
    - Maintaining a champion strategy and multiple challenger strategies
    - Generating challenger variations with mutated parameters
    - Monitoring performance and triggering evaluations
    - Promoting successful challengers to champion status
    - Managing strategy lifecycle (start, stop, restart)
    """
    
    def __init__(
        self,
        config: EVOPConfiguration,
        strategy_executor: Optional[Callable] = None,
        performance_tracker: Optional[Callable] = None
    ):
        """
        Initialize the Champion-Challenger Manager.
        
        Args:
            config: EVOP configuration parameters
            strategy_executor: Function to execute a strategy instance
            performance_tracker: Function to track strategy performance
        """
        self.config = config
        self.strategy_executor = strategy_executor
        self.performance_tracker = performance_tracker
        
        # Strategy management
        self.champion: Optional[StrategyInstance] = None
        self.challengers: Dict[str, StrategyInstance] = {}
        self.strategy_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance comparison
        self.comparator = PerformanceComparator(
            confidence_level=config.required_confidence_level
        )
        
        # Event tracking
        self.promotion_history: List[PromotionEvent] = []
        self.last_evaluation: Optional[datetime] = None
        
        # State management
        self.is_running = False
        self.total_allocated_capital = Decimal('0.0')
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_champion(
        self,
        parameters: StrategyParameters,
        initial_capital: Decimal,
        name: str = "Champion"
    ) -> StrategyInstance:
        """
        Initialize the champion strategy.
        
        Args:
            parameters: Initial strategy parameters
            initial_capital: Capital allocation for champion
            name: Name for the champion strategy
            
        Returns:
            The created champion strategy instance
        """
        self.champion = StrategyInstance(
            name=name,
            is_champion=True,
            parameters=parameters,
            allocated_capital=initial_capital,
            status=StrategyStatus.PENDING,
            description="Initial champion strategy"
        )
        
        self.total_allocated_capital = initial_capital
        
        self.logger.info(
            f"Initialized champion strategy '{name}' with capital ${initial_capital}"
        )
        
        return self.champion
    
    async def start_champion(self) -> bool:
        """
        Start the champion strategy execution.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.champion is None:
            self.logger.error("No champion strategy initialized")
            return False
        
        if self.champion.status == StrategyStatus.RUNNING:
            self.logger.warning("Champion is already running")
            return True
        
        try:
            # Start champion execution
            self.champion.status = StrategyStatus.RUNNING
            self.champion.started_at = datetime.now()
            
            if self.strategy_executor:
                task = asyncio.create_task(
                    self.strategy_executor(self.champion)
                )
                self.strategy_tasks[self.champion.instance_id] = task
            
            self.logger.info(f"Started champion strategy: {self.champion.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start champion: {e}")
            self.champion.status = StrategyStatus.FAILED
            return False
    
    async def generate_challengers(self, count: Optional[int] = None) -> List[StrategyInstance]:
        """
        Generate challenger strategies with mutated parameters.
        
        Args:
            count: Number of challengers to generate (defaults to config.max_challengers)
            
        Returns:
            List of generated challenger strategies
        """
        if self.champion is None:
            raise ValueError("Cannot generate challengers without a champion")
        
        if count is None:
            count = self.config.max_challengers
        
        generated_challengers = []
        
        for i in range(count):
            # Generate mutated parameters
            challenger_params = self._mutate_parameters(self.champion.parameters)
            
            # Calculate capital allocation
            challenger_capital = (
                self.champion.allocated_capital * 
                Decimal(str(self.config.challenger_capital_fraction))
            )
            
            # Create challenger instance
            challenger = StrategyInstance(
                name=f"Challenger-{i+1}",
                is_champion=False,
                parameters=challenger_params,
                allocated_capital=challenger_capital,
                status=StrategyStatus.PENDING,
                description=f"Auto-generated challenger {i+1}",
                tags=["auto-generated", "challenger"]
            )
            
            self.challengers[challenger.instance_id] = challenger
            generated_challengers.append(challenger)
            
            self.logger.info(
                f"Generated challenger '{challenger.name}' with capital ${challenger_capital}"
            )
        
        return generated_challengers
    
    async def start_challengers(self) -> int:
        """
        Start all pending challenger strategies.
        
        Returns:
            Number of challengers started successfully
        """
        started_count = 0
        
        for challenger in self.challengers.values():
            if challenger.status == StrategyStatus.PENDING:
                if await self._start_challenger(challenger):
                    started_count += 1
        
        self.logger.info(f"Started {started_count} challenger strategies")
        return started_count
    
    async def _start_challenger(self, challenger: StrategyInstance) -> bool:
        """Start a single challenger strategy."""
        try:
            challenger.status = StrategyStatus.RUNNING
            challenger.started_at = datetime.now()
            
            if self.strategy_executor:
                task = asyncio.create_task(
                    self.strategy_executor(challenger)
                )
                self.strategy_tasks[challenger.instance_id] = task
            
            self.logger.info(f"Started challenger strategy: {challenger.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start challenger {challenger.name}: {e}")
            challenger.status = StrategyStatus.FAILED
            return False
    
    def _mutate_parameters(self, base_params: StrategyParameters) -> StrategyParameters:
        """
        Create mutated parameters for a challenger strategy.
        
        Args:
            base_params: Base parameters to mutate
            
        Returns:
            New parameters with random mutations
        """
        # Create a copy of base parameters
        new_params = StrategyParameters(**base_params.to_dict())
        
        mutation_rate = self.config.parameter_mutation_rate
        mutation_std = self.config.parameter_mutation_std
        
        # Define parameter mutation rules
        mutation_rules = {
            'entry_z_score': {'min': 1.0, 'max': 4.0, 'type': 'continuous'},
            'exit_z_score': {'min': 0.1, 'max': 2.0, 'type': 'continuous'},
            'stop_loss_z_score': {'min': 2.0, 'max': 8.0, 'type': 'continuous'},
            'max_position_size': {'min': 1000.0, 'max': 100000.0, 'type': 'continuous'},
            'max_drawdown_percent': {'min': 5.0, 'max': 25.0, 'type': 'continuous'},
            'kelly_fraction': {'min': 0.1, 'max': 0.5, 'type': 'continuous'},
            'kalman_process_noise': {'min': 1e-6, 'max': 1e-3, 'type': 'log'},
            'kalman_observation_noise': {'min': 1e-4, 'max': 1e-1, 'type': 'log'},
            'garch_window_size': {'min': 100, 'max': 1000, 'type': 'integer'},
            'bocd_hazard_rate': {'min': 0.001, 'max': 0.01, 'type': 'continuous'},
            'order_timeout_seconds': {'min': 10, 'max': 60, 'type': 'integer'},
            'max_slippage_bps': {'min': 5, 'max': 25, 'type': 'integer'},
        }
        
        for param_name, rules in mutation_rules.items():
            if random.random() < mutation_rate:
                current_value = getattr(new_params, param_name)
                new_value = self._mutate_value(current_value, rules, mutation_std)
                setattr(new_params, param_name, new_value)
                
                self.logger.debug(
                    f"Mutated {param_name}: {current_value} -> {new_value}"
                )
        
        return new_params
    
    def _mutate_value(self, current_value: float, rules: Dict[str, Any], std: float) -> float:
        """Apply mutation to a single parameter value."""
        if rules['type'] == 'continuous':
            # Gaussian mutation with bounds
            noise = np.random.normal(0, std * current_value)
            new_value = current_value + noise
            
        elif rules['type'] == 'log':
            # Log-space mutation for parameters that span orders of magnitude
            log_current = np.log10(current_value)
            noise = np.random.normal(0, std)
            new_value = 10 ** (log_current + noise)
            
        elif rules['type'] == 'integer':
            # Integer mutation
            noise = np.random.normal(0, std * current_value)
            new_value = int(round(current_value + noise))
            
        else:
            return current_value
        
        # Apply bounds
        new_value = max(rules['min'], min(rules['max'], new_value))
        
        return new_value
    
    async def evaluate_promotions(self) -> List[PromotionEvent]:
        """
        Evaluate all challengers and perform promotions if warranted.
        
        Returns:
            List of promotion events that occurred
        """
        if self.champion is None:
            self.logger.warning("No champion to evaluate against")
            return []
        
        # Check if enough time has passed since last evaluation
        if self.last_evaluation is not None:
            time_since_eval = datetime.now() - self.last_evaluation
            min_interval = timedelta(hours=self.config.evaluation_frequency_hours)
            
            if time_since_eval < min_interval:
                self.logger.debug("Too soon for next evaluation")
                return []
        
        promotion_events = []
        
        # Evaluate each challenger against the champion
        for challenger in self.challengers.values():
            if challenger.status != StrategyStatus.RUNNING:
                continue
            
            # Perform comparison
            try:
                comparison = self.comparator.compare_strategies(
                    self.champion, 
                    challenger, 
                    self.config.min_evaluation_period_days
                )
                
                if comparison.should_promote and comparison.confidence_score >= self.config.required_confidence_level:
                    # Execute promotion
                    promotion_event = await self._promote_challenger(challenger, comparison)
                    if promotion_event:
                        promotion_events.append(promotion_event)
                        break  # Only one promotion per evaluation cycle
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate challenger {challenger.name}: {e}")
        
        self.last_evaluation = datetime.now()
        return promotion_events
    
    async def _promote_challenger(
        self,
        challenger: StrategyInstance,
        comparison: ComparisonResult
    ) -> Optional[PromotionEvent]:
        """
        Promote a challenger to champion status.
        
        Args:
            challenger: The challenger to promote
            comparison: The comparison result that triggered the promotion
            
        Returns:
            PromotionEvent if successful, None otherwise
        """
        if self.champion is None:
            return None
        
        try:
            self.logger.info(
                f"Promoting challenger '{challenger.name}' to champion "
                f"(confidence: {comparison.confidence_score:.3f}, reason: {comparison.reason.value})"
            )
            
            # Create promotion event
            promotion_event = PromotionEvent(
                old_champion_id=self.champion.instance_id,
                new_champion_id=challenger.instance_id,
                old_champion_name=self.champion.name,
                new_champion_name=challenger.name,
                reason=comparison.reason,
                confidence_score=comparison.confidence_score,
                old_champion_performance=self.champion.performance.to_dict(),
                new_champion_performance=challenger.performance.to_dict(),
                evaluation_period_days=(
                    (datetime.now() - challenger.started_at).days 
                    if challenger.started_at else 0
                ),
                notes=f"Automatic promotion based on {comparison.reason.value}"
            )
            
            # Stop old champion
            await self._stop_strategy(self.champion)
            
            # Promote challenger
            old_champion = self.champion
            challenger.is_champion = True
            challenger.name = f"Champion (promoted from {challenger.name})"
            
            # Transfer capital allocation
            challenger.allocated_capital = old_champion.allocated_capital
            
            # Update references
            self.champion = challenger
            del self.challengers[challenger.instance_id]
            
            # Stop other challengers and regenerate
            await self._stop_all_challengers()
            await self.generate_challengers()
            await self.start_challengers()
            
            # Record promotion
            self.promotion_history.append(promotion_event)
            
            self.logger.info(f"Successfully promoted challenger to champion")
            return promotion_event
            
        except Exception as e:
            self.logger.error(f"Failed to promote challenger: {e}")
            return None
    
    async def _stop_strategy(self, strategy: StrategyInstance) -> bool:
        """Stop a strategy execution."""
        try:
            strategy.status = StrategyStatus.STOPPED
            strategy.stopped_at = datetime.now()
            
            # Cancel the execution task
            if strategy.instance_id in self.strategy_tasks:
                task = self.strategy_tasks[strategy.instance_id]
                task.cancel()
                del self.strategy_tasks[strategy.instance_id]
            
            self.logger.info(f"Stopped strategy: {strategy.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop strategy {strategy.name}: {e}")
            return False
    
    async def _stop_all_challengers(self) -> int:
        """Stop all challenger strategies."""
        stopped_count = 0
        
        for challenger in list(self.challengers.values()):
            if await self._stop_strategy(challenger):
                stopped_count += 1
        
        self.challengers.clear()
        return stopped_count
    
    async def update_performance(
        self,
        strategy_id: str,
        performance_data: Dict[str, Any]
    ) -> bool:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy to update
            performance_data: Performance data dictionary
            
        Returns:
            True if updated successfully
        """
        strategy = None
        
        if self.champion and self.champion.instance_id == strategy_id:
            strategy = self.champion
        elif strategy_id in self.challengers:
            strategy = self.challengers[strategy_id]
        
        if strategy is None:
            self.logger.warning(f"Strategy {strategy_id} not found for performance update")
            return False
        
        try:
            # Update performance metrics
            perf = strategy.performance
            
            # Update basic metrics
            if 'total_pnl' in performance_data:
                perf.total_pnl = Decimal(str(performance_data['total_pnl']))
            if 'realized_pnl' in performance_data:
                perf.realized_pnl = Decimal(str(performance_data['realized_pnl']))
            if 'unrealized_pnl' in performance_data:
                perf.unrealized_pnl = Decimal(str(performance_data['unrealized_pnl']))
            
            # Update ratio metrics
            for metric in ['sharpe_ratio', 'calmar_ratio', 'total_return', 'max_drawdown', 
                          'current_drawdown', 'volatility', 'win_rate', 'avg_win', 'avg_loss']:
                if metric in performance_data:
                    setattr(perf, metric, performance_data[metric])
            
            # Update trade counts
            for metric in ['total_trades', 'winning_trades', 'losing_trades']:
                if metric in performance_data:
                    setattr(perf, metric, performance_data[metric])
            
            # Update return series
            if 'daily_returns' in performance_data:
                perf.daily_returns = performance_data['daily_returns']
            if 'monthly_returns' in performance_data:
                perf.monthly_returns = performance_data['monthly_returns']
            
            # Update derived metrics
            perf.update_trade_metrics()
            perf.last_updated = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update performance for {strategy_id}: {e}")
            return False
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the current EVOP status."""
        champion_info = None
        if self.champion:
            champion_info = {
                'id': self.champion.instance_id,
                'name': self.champion.name,
                'status': self.champion.status.value,
                'runtime_hours': (
                    (datetime.now() - self.champion.started_at).total_seconds() / 3600
                    if self.champion.started_at else 0
                ),
                'performance': self.champion.performance.to_dict()
            }
        
        challenger_info = []
        for challenger in self.challengers.values():
            challenger_info.append({
                'id': challenger.instance_id,
                'name': challenger.name,
                'status': challenger.status.value,
                'runtime_hours': (
                    (datetime.now() - challenger.started_at).total_seconds() / 3600
                    if challenger.started_at else 0
                ),
                'performance': challenger.performance.to_dict()
            })
        
        return {
            'champion': champion_info,
            'challengers': challenger_info,
            'active_strategies': len([s for s in self.challengers.values() if s.is_active()]) + (1 if self.champion and self.champion.is_active() else 0),
            'total_allocated_capital': float(self.total_allocated_capital),
            'last_evaluation': self.last_evaluation.isoformat() if self.last_evaluation else None,
            'promotion_count': len(self.promotion_history),
            'config': self.config.to_dict()
        }
    
    async def emergency_stop(self, reason: str = "Emergency stop triggered") -> bool:
        """
        Emergency stop all strategies.
        
        Args:
            reason: Reason for the emergency stop
            
        Returns:
            True if all strategies stopped successfully
        """
        self.logger.warning(f"Emergency stop triggered: {reason}")
        
        success = True
        
        # Stop champion
        if self.champion and self.champion.is_active():
            if not await self._stop_strategy(self.champion):
                success = False
        
        # Stop all challengers
        stopped_count = await self._stop_all_challengers()
        
        self.logger.info(
            f"Emergency stop completed. Champion: {'stopped' if self.champion else 'none'}, "
            f"Challengers stopped: {stopped_count}"
        )
        
        return success
