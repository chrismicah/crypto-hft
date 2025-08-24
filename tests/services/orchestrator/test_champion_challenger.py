"""
Unit tests for ChampionChallengerManager class.

Tests the core logic for managing parallel strategy execution,
challenger generation, and promotion decisions.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from services.orchestrator.champion_challenger import ChampionChallengerManager
from services.orchestrator.models import (
    StrategyInstance, StrategyParameters, StrategyStatus, 
    EVOPConfiguration, PromotionReason
)
from services.orchestrator.performance_comparator import ComparisonResult


class TestChampionChallengerManager:
    """Test the ChampionChallengerManager class."""
    
    @pytest.fixture
    def config(self):
        """Create a test EVOP configuration."""
        return EVOPConfiguration(
            max_challengers=2,
            challenger_capital_fraction=0.3,
            min_evaluation_period_days=3,
            required_confidence_level=0.8,
            parameter_mutation_rate=0.2,
            parameter_mutation_std=0.1,
            evaluation_frequency_hours=1
        )
    
    @pytest.fixture
    def mock_strategy_executor(self):
        """Mock strategy executor function."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_performance_tracker(self):
        """Mock performance tracker function."""
        return AsyncMock()
    
    @pytest.fixture
    def manager(self, config, mock_strategy_executor, mock_performance_tracker):
        """Create a ChampionChallengerManager instance for testing."""
        return ChampionChallengerManager(
            config=config,
            strategy_executor=mock_strategy_executor,
            performance_tracker=mock_performance_tracker
        )
    
    @pytest.fixture
    def base_parameters(self):
        """Create base strategy parameters for testing."""
        return StrategyParameters(
            entry_z_score=2.0,
            exit_z_score=0.5,
            max_position_size=10000.0,
            kelly_fraction=0.25
        )
    
    @pytest.mark.asyncio
    async def test_initialize_champion(self, manager, base_parameters):
        """Test champion initialization."""
        initial_capital = Decimal('100000.0')
        
        champion = await manager.initialize_champion(
            parameters=base_parameters,
            initial_capital=initial_capital,
            name="Test Champion"
        )
        
        assert champion is not None
        assert champion.name == "Test Champion"
        assert champion.is_champion is True
        assert champion.allocated_capital == initial_capital
        assert champion.status == StrategyStatus.PENDING
        assert manager.champion == champion
        assert manager.total_allocated_capital == initial_capital
    
    @pytest.mark.asyncio
    async def test_start_champion(self, manager, base_parameters, mock_strategy_executor):
        """Test starting the champion strategy."""
        # Initialize champion first
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        
        # Start champion
        success = await manager.start_champion()
        
        assert success is True
        assert manager.champion.status == StrategyStatus.RUNNING
        assert manager.champion.started_at is not None
        
        # Verify strategy executor was called
        mock_strategy_executor.assert_called_once_with(manager.champion)
    
    @pytest.mark.asyncio
    async def test_start_champion_without_initialization(self, manager):
        """Test starting champion without initialization should fail."""
        success = await manager.start_champion()
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_generate_challengers(self, manager, base_parameters):
        """Test challenger generation with parameter mutations."""
        # Initialize champion first
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        
        # Generate challengers
        challengers = await manager.generate_challengers(count=3)
        
        assert len(challengers) == 3
        assert len(manager.challengers) == 3
        
        for challenger in challengers:
            assert challenger.is_champion is False
            assert challenger.status == StrategyStatus.PENDING
            assert challenger.allocated_capital == Decimal('30000.0')  # 30% of champion
            assert "Challenger" in challenger.name
            
            # Parameters should be different from champion (due to mutation)
            # Note: With low mutation rate, some parameters might be the same
            # We'll just check that the structure is correct
            assert isinstance(challenger.parameters.entry_z_score, float)
            assert isinstance(challenger.parameters.exit_z_score, float)
    
    @pytest.mark.asyncio
    async def test_start_challengers(self, manager, base_parameters, mock_strategy_executor):
        """Test starting challenger strategies."""
        # Initialize champion and generate challengers
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.generate_challengers(count=2)
        
        # Start challengers
        started_count = await manager.start_challengers()
        
        assert started_count == 2
        
        for challenger in manager.challengers.values():
            assert challenger.status == StrategyStatus.RUNNING
            assert challenger.started_at is not None
        
        # Strategy executor should be called for each challenger
        assert mock_strategy_executor.call_count == 2
    
    def test_parameter_mutation(self, manager, base_parameters):
        """Test parameter mutation logic."""
        # Test multiple mutations to verify variety
        mutations = []
        for _ in range(10):
            mutated = manager._mutate_parameters(base_parameters)
            mutations.append(mutated)
        
        # Check that at least some parameters were mutated
        entry_z_scores = [m.entry_z_score for m in mutations]
        exit_z_scores = [m.exit_z_score for m in mutations]
        
        # With mutation rate of 0.2 and 10 samples, we should see some variation
        assert len(set(entry_z_scores)) > 1 or len(set(exit_z_scores)) > 1
        
        # Check bounds are respected
        for mutation in mutations:
            assert 1.0 <= mutation.entry_z_score <= 4.0
            assert 0.1 <= mutation.exit_z_score <= 2.0
            assert 1000.0 <= mutation.max_position_size <= 100000.0
    
    def test_mutate_value_continuous(self, manager):
        """Test continuous value mutation."""
        rules = {'min': 1.0, 'max': 5.0, 'type': 'continuous'}
        original_value = 2.0
        
        # Test multiple mutations
        mutations = []
        for _ in range(100):
            mutated = manager._mutate_value(original_value, rules, 0.1)
            mutations.append(mutated)
        
        # Check bounds
        assert all(1.0 <= m <= 5.0 for m in mutations)
        
        # Should have some variation
        assert len(set(mutations)) > 1
    
    def test_mutate_value_integer(self, manager):
        """Test integer value mutation."""
        rules = {'min': 10, 'max': 100, 'type': 'integer'}
        original_value = 50
        
        mutations = []
        for _ in range(100):
            mutated = manager._mutate_value(original_value, rules, 0.1)
            mutations.append(mutated)
        
        # Check all are integers within bounds
        assert all(isinstance(m, int) for m in mutations)
        assert all(10 <= m <= 100 for m in mutations)
    
    def test_mutate_value_log(self, manager):
        """Test logarithmic value mutation."""
        rules = {'min': 1e-6, 'max': 1e-2, 'type': 'log'}
        original_value = 1e-4
        
        mutations = []
        for _ in range(100):
            mutated = manager._mutate_value(original_value, rules, 0.1)
            mutations.append(mutated)
        
        # Check bounds
        assert all(1e-6 <= m <= 1e-2 for m in mutations)
        
        # Should have some variation
        assert len(set(mutations)) > 1
    
    @pytest.mark.asyncio
    async def test_update_performance(self, manager, base_parameters):
        """Test updating strategy performance."""
        # Initialize champion
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        
        performance_data = {
            'total_pnl': 5000.0,
            'realized_pnl': 3000.0,
            'unrealized_pnl': 2000.0,
            'sharpe_ratio': 1.5,
            'total_trades': 25,
            'winning_trades': 15,
            'daily_returns': [0.01, 0.02, -0.005, 0.015]
        }
        
        # Update performance
        success = await manager.update_performance(
            manager.champion.instance_id,
            performance_data
        )
        
        assert success is True
        
        # Verify performance was updated
        perf = manager.champion.performance
        assert perf.total_pnl == Decimal('5000.0')
        assert perf.realized_pnl == Decimal('3000.0')
        assert perf.unrealized_pnl == Decimal('2000.0')
        assert perf.sharpe_ratio == 1.5
        assert perf.total_trades == 25
        assert perf.winning_trades == 15
        assert perf.daily_returns == [0.01, 0.02, -0.005, 0.015]
        
        # Verify derived metrics were updated
        assert perf.win_rate == 0.6  # 15/25
    
    @pytest.mark.asyncio
    async def test_update_performance_invalid_strategy(self, manager):
        """Test updating performance for non-existent strategy."""
        performance_data = {'total_pnl': 1000.0}
        
        success = await manager.update_performance('invalid_id', performance_data)
        
        assert success is False
    
    @pytest.mark.asyncio
    @patch('services.orchestrator.champion_challenger.ChampionChallengerManager._promote_challenger')
    async def test_evaluate_promotions_with_promotion(self, mock_promote, manager, base_parameters):
        """Test evaluation that results in a promotion."""
        # Setup champion and challenger
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.generate_challengers(count=1)
        
        # Set up started times to meet evaluation criteria
        manager.champion.started_at = datetime.now() - timedelta(days=10)
        challenger = list(manager.challengers.values())[0]
        challenger.started_at = datetime.now() - timedelta(days=10)
        challenger.status = StrategyStatus.RUNNING
        
        # Mock the comparison to return a positive result
        mock_comparison = ComparisonResult(
            should_promote=True,
            confidence_score=0.95,
            reason=PromotionReason.SUPERIOR_SHARPE
        )
        
        with patch.object(manager.comparator, 'compare_strategies', return_value=mock_comparison):
            # Mock the promotion method to return a promotion event
            from services.orchestrator.models import PromotionEvent
            mock_event = PromotionEvent(
                old_champion_id=manager.champion.instance_id,
                new_champion_id=challenger.instance_id,
                reason=PromotionReason.SUPERIOR_SHARPE,
                confidence_score=0.95
            )
            mock_promote.return_value = mock_event
            
            # Evaluate promotions
            events = await manager.evaluate_promotions()
            
            assert len(events) == 1
            assert events[0] == mock_event
            mock_promote.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_promotions_no_promotion(self, manager, base_parameters):
        """Test evaluation that does not result in promotion."""
        # Setup champion and challenger
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.generate_challengers(count=1)
        
        # Set up started times
        manager.champion.started_at = datetime.now() - timedelta(days=10)
        challenger = list(manager.challengers.values())[0]
        challenger.started_at = datetime.now() - timedelta(days=10)
        challenger.status = StrategyStatus.RUNNING
        
        # Mock the comparison to return a negative result
        mock_comparison = ComparisonResult(
            should_promote=False,
            confidence_score=0.3,
            reason=PromotionReason.SUPERIOR_SHARPE
        )
        
        with patch.object(manager.comparator, 'compare_strategies', return_value=mock_comparison):
            events = await manager.evaluate_promotions()
            
            assert len(events) == 0
    
    @pytest.mark.asyncio
    async def test_stop_strategy(self, manager, base_parameters):
        """Test stopping a strategy."""
        # Initialize and start champion
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.start_champion()
        
        # Stop the strategy
        success = await manager._stop_strategy(manager.champion)
        
        assert success is True
        assert manager.champion.status == StrategyStatus.STOPPED
        assert manager.champion.stopped_at is not None
    
    @pytest.mark.asyncio
    async def test_stop_all_challengers(self, manager, base_parameters):
        """Test stopping all challenger strategies."""
        # Setup champion and challengers
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.generate_challengers(count=3)
        await manager.start_challengers()
        
        # Stop all challengers
        stopped_count = await manager._stop_all_challengers()
        
        assert stopped_count == 3
        assert len(manager.challengers) == 0
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, manager, base_parameters):
        """Test emergency stop functionality."""
        # Setup champion and challengers
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.generate_challengers(count=2)
        await manager.start_champion()
        await manager.start_challengers()
        
        # Emergency stop
        success = await manager.emergency_stop("Test emergency")
        
        assert success is True
        assert manager.champion.status == StrategyStatus.STOPPED
        assert len(manager.challengers) == 0
    
    def test_get_status_summary(self, manager, base_parameters):
        """Test status summary generation."""
        # Get status with no strategies
        status = manager.get_status_summary()
        
        assert status['champion'] is None
        assert status['challengers'] == []
        assert status['active_strategies'] == 0
        assert status['total_allocated_capital'] == 0.0
        assert status['promotion_count'] == 0
        assert isinstance(status['config'], dict)
    
    @pytest.mark.asyncio
    async def test_get_status_summary_with_strategies(self, manager, base_parameters):
        """Test status summary with active strategies."""
        # Setup strategies
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        await manager.generate_challengers(count=2)
        await manager.start_champion()
        await manager.start_challengers()
        
        status = manager.get_status_summary()
        
        assert status['champion'] is not None
        assert status['champion']['name'] == "Champion"  # Default name from fixture
        assert status['champion']['status'] == StrategyStatus.RUNNING.value
        assert len(status['challengers']) == 2
        assert status['active_strategies'] == 3  # 1 champion + 2 challengers
        assert status['total_allocated_capital'] == 100000.0
    
    @pytest.mark.asyncio
    async def test_promote_challenger_full_flow(self, manager, base_parameters):
        """Test the complete challenger promotion flow."""
        # Setup champion and challenger
        await manager.initialize_champion(base_parameters, Decimal('100000.0'))
        challengers = await manager.generate_challengers(count=1)
        challenger = challengers[0]
        
        # Set up performance data
        challenger.performance.sharpe_ratio = 2.0  # Better than default
        challenger.started_at = datetime.now() - timedelta(days=10)
        
        # Mock comparison result
        comparison = ComparisonResult(
            should_promote=True,
            confidence_score=0.95,
            reason=PromotionReason.SUPERIOR_SHARPE
        )
        
        # Execute promotion
        with patch.object(manager, '_stop_strategy', return_value=True):
            with patch.object(manager, '_stop_all_challengers', return_value=0):
                with patch.object(manager, 'generate_challengers', return_value=[]):
                    with patch.object(manager, 'start_challengers', return_value=0):
                        promotion_event = await manager._promote_challenger(challenger, comparison)
        
        assert promotion_event is not None
        assert promotion_event.reason == PromotionReason.SUPERIOR_SHARPE
        assert promotion_event.confidence_score == 0.95
        assert promotion_event.new_champion_id == challenger.instance_id
        
        # Verify challenger became champion
        assert manager.champion == challenger
        assert challenger.is_champion is True
        assert "Champion" in challenger.name
        assert challenger.instance_id not in manager.challengers
        
        # Verify promotion was recorded
        assert len(manager.promotion_history) == 1
        assert manager.promotion_history[0] == promotion_event


class TestEVOPConfiguration:
    """Test the EVOPConfiguration data class."""
    
    def test_evop_configuration_creation(self):
        """Test creation of EVOP configuration."""
        config = EVOPConfiguration(
            max_challengers=5,
            challenger_capital_fraction=0.15,
            required_confidence_level=0.99
        )
        
        assert config.max_challengers == 5
        assert config.challenger_capital_fraction == 0.15
        assert config.required_confidence_level == 0.99
        
        # Test defaults
        assert config.min_evaluation_period_days == 7
        assert config.parameter_mutation_rate == 0.1
    
    def test_evop_configuration_to_dict(self):
        """Test conversion of configuration to dictionary."""
        config = EVOPConfiguration()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'max_challengers' in config_dict
        assert 'challenger_capital_fraction' in config_dict
        assert 'required_confidence_level' in config_dict
        assert config_dict['max_challengers'] == 3  # Default value
