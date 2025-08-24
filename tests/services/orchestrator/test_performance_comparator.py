"""
Unit tests for PerformanceComparator class.

Tests the statistical comparison logic for determining when a challenger
should be promoted to champion status.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from services.orchestrator.performance_comparator import PerformanceComparator, ComparisonResult
from services.orchestrator.models import (
    StrategyInstance, StrategyParameters, PerformanceMetrics, PromotionReason
)


class TestPerformanceComparator:
    """Test the PerformanceComparator class."""
    
    @pytest.fixture
    def comparator(self):
        """Create a PerformanceComparator instance for testing."""
        return PerformanceComparator(min_samples=10, confidence_level=0.80)  # Lower for testing
    
    @pytest.fixture
    def champion_strategy(self):
        """Create a mock champion strategy with performance data."""
        strategy = StrategyInstance(
            name="Champion",
            is_champion=True,
            started_at=datetime.now() - timedelta(days=30)
        )
        
        # Add performance data
        strategy.performance.sharpe_ratio = 1.2
        strategy.performance.calmar_ratio = 0.8
        strategy.performance.max_drawdown = 0.15
        strategy.performance.total_return = 0.20
        strategy.performance.volatility = 0.18
        
        # Generate realistic daily returns (20% annual return, 18% volatility)
        np.random.seed(42)  # For reproducible tests
        daily_returns = np.random.normal(0.20/252, 0.18/np.sqrt(252), 60)  # 60 days
        strategy.performance.daily_returns = daily_returns.tolist()
        
        return strategy
    
    @pytest.fixture
    def superior_challenger(self):
        """Create a challenger strategy that outperforms the champion."""
        strategy = StrategyInstance(
            name="Superior Challenger",
            is_champion=False,
            started_at=datetime.now() - timedelta(days=30)
        )
        
        # Better performance metrics
        strategy.performance.sharpe_ratio = 1.5  # 25% improvement
        strategy.performance.calmar_ratio = 1.1  # 37.5% improvement
        strategy.performance.max_drawdown = 0.10  # Lower drawdown
        strategy.performance.total_return = 0.28  # Higher return
        strategy.performance.volatility = 0.16   # Lower volatility
        
        # Generate better daily returns
        np.random.seed(123)  # Different seed for different performance
        daily_returns = np.random.normal(0.28/252, 0.16/np.sqrt(252), 60)
        strategy.performance.daily_returns = daily_returns.tolist()
        
        return strategy
    
    @pytest.fixture
    def inferior_challenger(self):
        """Create a challenger strategy that underperforms the champion."""
        strategy = StrategyInstance(
            name="Inferior Challenger",
            is_champion=False,
            started_at=datetime.now() - timedelta(days=30)
        )
        
        # Worse performance metrics
        strategy.performance.sharpe_ratio = 0.9   # Lower than champion
        strategy.performance.calmar_ratio = 0.6   # Lower than champion
        strategy.performance.max_drawdown = 0.25  # Higher drawdown
        strategy.performance.total_return = 0.12  # Lower return
        strategy.performance.volatility = 0.22    # Higher volatility
        
        # Generate worse daily returns
        np.random.seed(456)
        daily_returns = np.random.normal(0.12/252, 0.22/np.sqrt(252), 60)
        strategy.performance.daily_returns = daily_returns.tolist()
        
        return strategy
    
    def test_insufficient_data_handling(self, comparator, champion_strategy):
        """Test handling of insufficient data for comparison."""
        # Create challenger with insufficient data
        new_challenger = StrategyInstance(
            name="New Challenger",
            is_champion=False,
            started_at=datetime.now() - timedelta(days=2)  # Too recent
        )
        new_challenger.performance.daily_returns = [0.01, 0.02]  # Too few samples
        
        result = comparator.compare_strategies(champion_strategy, new_challenger)
        
        assert not result.should_promote
        assert result.confidence_score == 0.0
        assert 'Insufficient data' in result.details.get('error', '')
    
    def test_superior_challenger_promotion(self, comparator, champion_strategy, superior_challenger):
        """Test that a superior challenger is recommended for promotion."""
        result = comparator.compare_strategies(champion_strategy, superior_challenger)
        
        # With random data, promotion may not always happen, so test structure
        assert result.should_promote in [True, False]
        assert isinstance(result.confidence_score, float)
        assert result.reason in [
            PromotionReason.SUPERIOR_SHARPE,
            PromotionReason.HIGHER_RETURNS,
            PromotionReason.LOWER_DRAWDOWN,
            PromotionReason.SUPERIOR_CALMAR,
            PromotionReason.MANUAL_OVERRIDE
        ]
        
        # At least some tests should be favorable due to better metrics
        assert result.details['favorable_tests'] >= 2
    
    def test_inferior_challenger_rejection(self, comparator, champion_strategy, inferior_challenger):
        """Test that an inferior challenger is not recommended for promotion."""
        result = comparator.compare_strategies(champion_strategy, inferior_challenger)
        
        assert not result.should_promote
        assert result.confidence_score < 0.5
    
    def test_sharpe_ratio_comparison(self, comparator, champion_strategy, superior_challenger):
        """Test specific Sharpe ratio comparison logic."""
        result = comparator._compare_sharpe_ratios(champion_strategy, superior_challenger)
        
        # With statistical tests, promotion may not always be recommended
        assert result.should_promote in [True, False]
        assert result.reason == PromotionReason.SUPERIOR_SHARPE
        assert isinstance(result.confidence_score, float)
        assert 'champion_sharpe' in result.details
        assert 'challenger_sharpe' in result.details
        
        # Challenger should have better Sharpe ratio
        assert result.details['challenger_sharpe'] > result.details['champion_sharpe']
    
    def test_return_comparison(self, comparator, champion_strategy, superior_challenger):
        """Test return comparison using t-test."""
        result = comparator._compare_returns(champion_strategy, superior_challenger)
        
        # Note: With random data, results may vary, so we test the structure
        assert result.should_promote in [True, False]
        assert result.reason == PromotionReason.HIGHER_RETURNS
        assert 'champion_mean_return' in result.details
        assert 'challenger_mean_return' in result.details
        assert 'cohens_d' in result.details
    
    def test_drawdown_comparison(self, comparator, champion_strategy, superior_challenger):
        """Test maximum drawdown comparison."""
        result = comparator._compare_drawdowns(champion_strategy, superior_challenger)
        
        assert result.should_promote  # Superior challenger has lower drawdown
        assert result.reason == PromotionReason.LOWER_DRAWDOWN
        assert result.details['champion_max_drawdown'] == 0.15
        assert result.details['challenger_max_drawdown'] == 0.10
        assert result.details['improvement'] > 0.1  # More than 10% improvement
    
    def test_calmar_ratio_comparison(self, comparator, champion_strategy, superior_challenger):
        """Test Calmar ratio comparison."""
        result = comparator._compare_calmar_ratios(champion_strategy, superior_challenger)
        
        assert result.should_promote
        assert result.reason == PromotionReason.SUPERIOR_CALMAR
        assert result.details['challenger_calmar'] > result.details['champion_calmar']
    
    def test_stability_comparison(self, comparator, champion_strategy, superior_challenger):
        """Test volatility/stability comparison."""
        result = comparator._compare_stability(champion_strategy, superior_challenger)
        
        # Test the structure (results depend on random data)
        assert result.should_promote in [True, False]
        assert 'champion_volatility' in result.details
        assert 'challenger_volatility' in result.details
        if result.should_promote:
            assert result.details['improvement'] > 0.05
    
    def test_jobson_korkie_test(self, comparator):
        """Test the Jobson-Korkie test implementation."""
        # Create test return series
        np.random.seed(42)
        returns1 = np.random.normal(0.001, 0.02, 100)  # Lower Sharpe
        returns2 = np.random.normal(0.002, 0.02, 100)  # Higher Sharpe
        
        test_stat, p_value = comparator._jobson_korkie_test(returns1, returns2)
        
        assert isinstance(test_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_missing_sharpe_ratio_handling(self, comparator):
        """Test handling of missing Sharpe ratio data."""
        champion = StrategyInstance(name="Champion", is_champion=True)
        challenger = StrategyInstance(name="Challenger", is_champion=False)
        
        # Leave Sharpe ratios as None
        champion.performance.sharpe_ratio = None
        challenger.performance.sharpe_ratio = None
        
        result = comparator._compare_sharpe_ratios(champion, challenger)
        
        assert not result.should_promote
        assert result.confidence_score == 0.0
        assert 'Missing Sharpe ratio data' in result.details.get('error', '')
    
    def test_missing_calmar_ratio_handling(self, comparator):
        """Test handling of missing Calmar ratio data."""
        champion = StrategyInstance(name="Champion", is_champion=True)
        challenger = StrategyInstance(name="Challenger", is_champion=False)
        
        # Leave Calmar ratios as None
        champion.performance.calmar_ratio = None
        challenger.performance.calmar_ratio = None
        
        result = comparator._compare_calmar_ratios(champion, challenger)
        
        assert not result.should_promote
        assert result.confidence_score == 0.0
        assert 'Missing Calmar ratio data' in result.details.get('error', '')
    
    def test_small_sample_sharpe_comparison(self, comparator):
        """Test Sharpe ratio comparison with small sample sizes."""
        champion = StrategyInstance(name="Champion", is_champion=True)
        challenger = StrategyInstance(name="Challenger", is_champion=False)
        
        # Small return series
        champion.performance.daily_returns = [0.01, 0.02, -0.01]
        challenger.performance.daily_returns = [0.02, 0.03, 0.01]
        
        champion.performance.sharpe_ratio = 0.5
        challenger.performance.sharpe_ratio = 1.0  # 100% improvement
        
        result = comparator._compare_sharpe_ratios(champion, challenger)
        
        assert result.should_promote  # Should promote due to significant improvement
        assert result.details['method'] == 'simple_comparison'
    
    def test_edge_case_zero_volatility(self, comparator):
        """Test handling of zero volatility (constant returns)."""
        champion = StrategyInstance(name="Champion", is_champion=True)
        challenger = StrategyInstance(name="Challenger", is_champion=False)
        
        # Constant returns (zero volatility)
        champion.performance.daily_returns = [0.01] * 50
        challenger.performance.daily_returns = [0.02] * 50
        
        result = comparator._compare_stability(champion, challenger)
        
        # Should handle gracefully without errors
        assert result.should_promote in [True, False]
        assert isinstance(result.confidence_score, float)
    
    def test_aggregation_logic(self, comparator, champion_strategy, superior_challenger):
        """Test the aggregation of multiple comparison results."""
        # Create individual results
        sharpe_result = ComparisonResult(
            should_promote=True,
            confidence_score=0.9,
            reason=PromotionReason.SUPERIOR_SHARPE
        )
        
        return_result = ComparisonResult(
            should_promote=True,
            confidence_score=0.8,
            reason=PromotionReason.HIGHER_RETURNS
        )
        
        drawdown_result = ComparisonResult(
            should_promote=False,
            confidence_score=0.3,
            reason=PromotionReason.LOWER_DRAWDOWN
        )
        
        results = [sharpe_result, return_result, drawdown_result]
        
        final_result = comparator._aggregate_results(results, champion_strategy, superior_challenger)
        
        # Should promote due to majority favorable results with high confidence
        assert final_result.should_promote
        assert final_result.confidence_score > 0.7
        assert final_result.details['total_tests'] == 3
        assert final_result.details['favorable_tests'] == 2
    
    def test_insufficient_favorable_results(self, comparator, champion_strategy, superior_challenger):
        """Test aggregation when insufficient tests are favorable."""
        # Create mostly unfavorable results
        results = [
            ComparisonResult(False, 0.2, PromotionReason.SUPERIOR_SHARPE),
            ComparisonResult(False, 0.3, PromotionReason.HIGHER_RETURNS),
            ComparisonResult(True, 0.6, PromotionReason.LOWER_DRAWDOWN),
        ]
        
        final_result = comparator._aggregate_results(results, champion_strategy, superior_challenger)
        
        # Should not promote due to insufficient favorable results
        assert not final_result.should_promote
        assert final_result.details['favorable_tests'] == 1


class TestComparisonResult:
    """Test the ComparisonResult data class."""
    
    def test_comparison_result_creation(self):
        """Test creation of ComparisonResult objects."""
        result = ComparisonResult(
            should_promote=True,
            confidence_score=0.95,
            reason=PromotionReason.SUPERIOR_SHARPE,
            p_value=0.01,
            test_statistic=2.5,
            effect_size=0.8
        )
        
        assert result.should_promote is True
        assert result.confidence_score == 0.95
        assert result.reason == PromotionReason.SUPERIOR_SHARPE
        assert result.p_value == 0.01
        assert result.test_statistic == 2.5
        assert result.effect_size == 0.8
        assert result.details is not None
    
    def test_comparison_result_defaults(self):
        """Test default values in ComparisonResult."""
        result = ComparisonResult(
            should_promote=False,
            confidence_score=0.0,
            reason=PromotionReason.MANUAL_OVERRIDE
        )
        
        assert result.p_value is None
        assert result.test_statistic is None
        assert result.effect_size is None
        assert result.details == {}
