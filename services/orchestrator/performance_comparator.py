"""
Performance Comparison Logic for EVOP Framework.

This module implements statistical methods for comparing strategy performance
and determining when a challenger should be promoted to champion status.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from scipy import stats
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .models import StrategyInstance, PromotionReason, PerformanceMetrics


@dataclass
class ComparisonResult:
    """Result of a performance comparison between strategies."""
    
    should_promote: bool
    confidence_score: float
    reason: PromotionReason
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    effect_size: Optional[float] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PerformanceComparator:
    """
    Compares strategy performance using statistical methods.
    
    This class implements various statistical tests and metrics to determine
    when a challenger strategy has sufficiently outperformed the champion
    to warrant promotion.
    """
    
    def __init__(self, min_samples: int = 30, confidence_level: float = 0.95):
        """
        Initialize the performance comparator.
        
        Args:
            min_samples: Minimum number of return samples for statistical tests
            confidence_level: Required confidence level for promotion decisions
        """
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def compare_strategies(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance,
        min_evaluation_days: int = 7
    ) -> ComparisonResult:
        """
        Compare two strategies and determine if challenger should be promoted.
        
        Args:
            champion: Current champion strategy
            challenger: Challenger strategy
            min_evaluation_days: Minimum days of data required for comparison
            
        Returns:
            ComparisonResult indicating whether promotion should occur
        """
        # Check if sufficient data exists
        if not self._has_sufficient_data(champion, challenger, min_evaluation_days):
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.MANUAL_OVERRIDE,
                details={'error': 'Insufficient data for comparison'}
            )
        
        # Perform multiple comparison tests
        results = []
        
        # 1. Sharpe ratio comparison
        sharpe_result = self._compare_sharpe_ratios(champion, challenger)
        results.append(sharpe_result)
        
        # 2. Return comparison
        return_result = self._compare_returns(champion, challenger)
        results.append(return_result)
        
        # 3. Drawdown comparison
        drawdown_result = self._compare_drawdowns(champion, challenger)
        results.append(drawdown_result)
        
        # 4. Calmar ratio comparison
        calmar_result = self._compare_calmar_ratios(champion, challenger)
        results.append(calmar_result)
        
        # 5. Stability comparison (volatility of returns)
        stability_result = self._compare_stability(champion, challenger)
        results.append(stability_result)
        
        # Aggregate results
        return self._aggregate_results(results, champion, challenger)
    
    def _has_sufficient_data(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance,
        min_days: int
    ) -> bool:
        """Check if both strategies have sufficient data for comparison."""
        now = datetime.now()
        
        # Check champion data
        if champion.started_at is None:
            return False
        
        champion_runtime = now - champion.started_at
        if champion_runtime.days < min_days:
            return False
        
        # Check challenger data
        if challenger.started_at is None:
            return False
        
        challenger_runtime = now - challenger.started_at
        if challenger_runtime.days < min_days:
            return False
        
        # Check if both have sufficient return samples
        champion_samples = len(champion.performance.daily_returns)
        challenger_samples = len(challenger.performance.daily_returns)
        
        return (champion_samples >= self.min_samples and 
                challenger_samples >= self.min_samples)
    
    def _compare_sharpe_ratios(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance
    ) -> ComparisonResult:
        """Compare Sharpe ratios between strategies."""
        champion_sharpe = champion.performance.sharpe_ratio
        challenger_sharpe = challenger.performance.sharpe_ratio
        
        if champion_sharpe is None or challenger_sharpe is None:
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.SUPERIOR_SHARPE,
                details={'error': 'Missing Sharpe ratio data'}
            )
        
        # Use Jobson-Korkie test for Sharpe ratio comparison
        champion_returns = np.array(champion.performance.daily_returns)
        challenger_returns = np.array(challenger.performance.daily_returns)
        
        # Align return series (use overlapping period)
        min_length = min(len(champion_returns), len(challenger_returns))
        champion_returns = champion_returns[-min_length:]
        challenger_returns = challenger_returns[-min_length:]
        
        # Calculate test statistic
        sharpe_diff = challenger_sharpe - champion_sharpe
        
        if len(champion_returns) < 10:  # Fallback for small samples
            # Simple comparison
            improvement_threshold = 0.1  # 10% improvement required
            relative_improvement = sharpe_diff / abs(champion_sharpe) if champion_sharpe != 0 else 0
            
            should_promote = (challenger_sharpe > champion_sharpe and 
                            relative_improvement > improvement_threshold)
            
            confidence = min(0.9, relative_improvement * 2) if should_promote else 0.0
            
            return ComparisonResult(
                should_promote=should_promote,
                confidence_score=confidence,
                reason=PromotionReason.SUPERIOR_SHARPE,
                details={
                    'champion_sharpe': champion_sharpe,
                    'challenger_sharpe': challenger_sharpe,
                    'improvement': relative_improvement,
                    'method': 'simple_comparison'
                }
            )
        
        # Jobson-Korkie test
        try:
            test_stat, p_value = self._jobson_korkie_test(
                champion_returns, challenger_returns
            )
            
            should_promote = (challenger_sharpe > champion_sharpe and 
                            p_value < (1 - self.confidence_level))
            
            confidence = 1 - p_value if should_promote else 0.0
            
            return ComparisonResult(
                should_promote=should_promote,
                confidence_score=confidence,
                reason=PromotionReason.SUPERIOR_SHARPE,
                p_value=p_value,
                test_statistic=test_stat,
                details={
                    'champion_sharpe': champion_sharpe,
                    'challenger_sharpe': challenger_sharpe,
                    'method': 'jobson_korkie'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Jobson-Korkie test failed: {e}")
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.SUPERIOR_SHARPE,
                details={'error': str(e)}
            )
    
    def _compare_returns(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance
    ) -> ComparisonResult:
        """Compare raw returns between strategies using t-test."""
        champion_returns = np.array(champion.performance.daily_returns)
        challenger_returns = np.array(challenger.performance.daily_returns)
        
        if len(champion_returns) < 2 or len(challenger_returns) < 2:
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.HIGHER_RETURNS,
                details={'error': 'Insufficient return data'}
            )
        
        # Align return series
        min_length = min(len(champion_returns), len(challenger_returns))
        champion_returns = champion_returns[-min_length:]
        challenger_returns = challenger_returns[-min_length:]
        
        # Perform Welch's t-test (unequal variances)
        try:
            t_stat, p_value = stats.ttest_ind(
                challenger_returns, champion_returns, equal_var=False
            )
            
            challenger_mean = np.mean(challenger_returns)
            champion_mean = np.mean(champion_returns)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(champion_returns) + np.var(challenger_returns)) / 2
            )
            cohens_d = (challenger_mean - champion_mean) / pooled_std if pooled_std > 0 else 0
            
            should_promote = (challenger_mean > champion_mean and 
                            p_value < (1 - self.confidence_level) and
                            cohens_d > 0.2)  # Small effect size threshold
            
            confidence = 1 - p_value if should_promote else 0.0
            
            return ComparisonResult(
                should_promote=should_promote,
                confidence_score=confidence,
                reason=PromotionReason.HIGHER_RETURNS,
                p_value=p_value,
                test_statistic=t_stat,
                effect_size=cohens_d,
                details={
                    'champion_mean_return': champion_mean,
                    'challenger_mean_return': challenger_mean,
                    'cohens_d': cohens_d,
                    'method': 'welch_ttest'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Return comparison t-test failed: {e}")
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.HIGHER_RETURNS,
                details={'error': str(e)}
            )
    
    def _compare_drawdowns(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance
    ) -> ComparisonResult:
        """Compare maximum drawdowns between strategies."""
        champion_dd = champion.performance.max_drawdown
        challenger_dd = challenger.performance.max_drawdown
        
        # Lower drawdown is better
        dd_improvement = champion_dd - challenger_dd
        relative_improvement = dd_improvement / champion_dd if champion_dd > 0 else 0
        
        # Require at least 10% improvement in drawdown
        should_promote = (challenger_dd < champion_dd and 
                         relative_improvement > 0.1)
        
        confidence = min(0.9, relative_improvement) if should_promote else 0.0
        
        return ComparisonResult(
            should_promote=should_promote,
            confidence_score=confidence,
            reason=PromotionReason.LOWER_DRAWDOWN,
            details={
                'champion_max_drawdown': champion_dd,
                'challenger_max_drawdown': challenger_dd,
                'improvement': relative_improvement,
                'method': 'simple_comparison'
            }
        )
    
    def _compare_calmar_ratios(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance
    ) -> ComparisonResult:
        """Compare Calmar ratios between strategies."""
        champion_calmar = champion.performance.calmar_ratio
        challenger_calmar = challenger.performance.calmar_ratio
        
        if champion_calmar is None or challenger_calmar is None:
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.SUPERIOR_CALMAR,
                details={'error': 'Missing Calmar ratio data'}
            )
        
        calmar_improvement = challenger_calmar - champion_calmar
        relative_improvement = calmar_improvement / abs(champion_calmar) if champion_calmar != 0 else 0
        
        should_promote = (challenger_calmar > champion_calmar and 
                         relative_improvement > 0.1)  # 10% improvement required
        
        confidence = min(0.9, relative_improvement) if should_promote else 0.0
        
        return ComparisonResult(
            should_promote=should_promote,
            confidence_score=confidence,
            reason=PromotionReason.SUPERIOR_CALMAR,
            details={
                'champion_calmar': champion_calmar,
                'challenger_calmar': challenger_calmar,
                'improvement': relative_improvement,
                'method': 'simple_comparison'
            }
        )
    
    def _compare_stability(
        self,
        champion: StrategyInstance,
        challenger: StrategyInstance
    ) -> ComparisonResult:
        """Compare stability (volatility) of returns."""
        champion_returns = np.array(champion.performance.daily_returns)
        challenger_returns = np.array(challenger.performance.daily_returns)
        
        if len(champion_returns) < 2 or len(challenger_returns) < 2:
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.HIGHER_RETURNS,
                details={'error': 'Insufficient data for stability comparison'}
            )
        
        champion_vol = np.std(champion_returns)
        challenger_vol = np.std(challenger_returns)
        
        # Use F-test to compare variances
        try:
            champion_var = np.var(champion_returns)
            challenger_var = np.var(challenger_returns)
            
            # Handle zero variance case
            if champion_var == 0 and challenger_var == 0:
                return ComparisonResult(
                    should_promote=False,
                    confidence_score=0.0,
                    reason=PromotionReason.HIGHER_RETURNS,
                    details={
                        'champion_volatility': champion_vol,
                        'challenger_volatility': challenger_vol,
                        'improvement': 0,
                        'method': 'zero_variance'
                    }
                )
            
            if challenger_var == 0:
                # Challenger has zero variance, can't compute F-test
                return ComparisonResult(
                    should_promote=False,
                    confidence_score=0.0,
                    reason=PromotionReason.HIGHER_RETURNS,
                    details={
                        'champion_volatility': champion_vol,
                        'challenger_volatility': challenger_vol,
                        'improvement': -1.0,  # Infinite improvement but suspicious
                        'method': 'zero_challenger_variance'
                    }
                )
            
            f_stat = champion_var / challenger_var
            df1 = len(champion_returns) - 1
            df2 = len(challenger_returns) - 1
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                             1 - stats.f.cdf(f_stat, df1, df2))
            
            # Lower volatility is better (more stable)
            vol_improvement = (champion_vol - challenger_vol) / champion_vol if champion_vol > 0 else 0
            
            should_promote = (challenger_vol < champion_vol and 
                            p_value < (1 - self.confidence_level) and
                            vol_improvement > 0.05)  # 5% improvement in stability
            
            confidence = 1 - p_value if should_promote else 0.0
            
            return ComparisonResult(
                should_promote=should_promote,
                confidence_score=confidence,
                reason=PromotionReason.HIGHER_RETURNS,  # Stability contributes to return quality
                p_value=p_value,
                test_statistic=f_stat,
                details={
                    'champion_volatility': champion_vol,
                    'challenger_volatility': challenger_vol,
                    'improvement': vol_improvement,
                    'method': 'f_test'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Stability comparison failed: {e}")
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.HIGHER_RETURNS,
                details={'error': str(e)}
            )
    
    def _jobson_korkie_test(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Jobson-Korkie test for Sharpe ratio equality.
        
        Returns:
            Tuple of (test_statistic, p_value)
        """
        n = len(returns1)
        
        # Calculate Sharpe ratios
        sr1 = np.mean(returns1) / np.std(returns1) if np.std(returns1) > 0 else 0
        sr2 = np.mean(returns2) / np.std(returns2) if np.std(returns2) > 0 else 0
        
        # Calculate correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        # Calculate test statistic
        var1 = np.var(returns1)
        var2 = np.var(returns2)
        
        if var1 == 0 or var2 == 0:
            return 0.0, 1.0
        
        # Asymptotic variance of Sharpe ratio difference
        term1 = (1 + 0.5 * sr1**2) / n
        term2 = (1 + 0.5 * sr2**2) / n
        term3 = -2 * correlation * np.sqrt((1 + 0.5 * sr1**2) * (1 + 0.5 * sr2**2)) / n
        
        variance = term1 + term2 + term3
        
        if variance <= 0:
            return 0.0, 1.0
        
        test_stat = (sr2 - sr1) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        
        return test_stat, p_value
    
    def _aggregate_results(
        self,
        results: List[ComparisonResult],
        champion: StrategyInstance,
        challenger: StrategyInstance
    ) -> ComparisonResult:
        """Aggregate multiple comparison results into a final decision."""
        # Count favorable results
        favorable_results = [r for r in results if r.should_promote]
        
        if not favorable_results:
            return ComparisonResult(
                should_promote=False,
                confidence_score=0.0,
                reason=PromotionReason.MANUAL_OVERRIDE,
                details={
                    'total_tests': len(results),
                    'favorable_tests': 0,
                    'test_results': [r.details for r in results]
                }
            )
        
        # Calculate overall confidence as weighted average
        # Give higher weight to Sharpe ratio and return comparisons
        weights = {
            PromotionReason.SUPERIOR_SHARPE: 0.3,
            PromotionReason.HIGHER_RETURNS: 0.3,
            PromotionReason.LOWER_DRAWDOWN: 0.2,
            PromotionReason.SUPERIOR_CALMAR: 0.15,
        }
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        primary_reason = PromotionReason.MANUAL_OVERRIDE
        max_confidence = 0.0
        
        for result in favorable_results:
            weight = weights.get(result.reason, 0.05)
            total_weighted_confidence += result.confidence_score * weight
            total_weight += weight
            
            if result.confidence_score > max_confidence:
                max_confidence = result.confidence_score
                primary_reason = result.reason
        
        overall_confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Require multiple favorable tests and high overall confidence
        min_favorable_tests = max(1, len(results) // 2)  # At least half of tests
        min_confidence = 0.8
        
        should_promote = (len(favorable_results) >= min_favorable_tests and 
                         overall_confidence >= min_confidence)
        
        return ComparisonResult(
            should_promote=should_promote,
            confidence_score=overall_confidence,
            reason=primary_reason,
            details={
                'total_tests': len(results),
                'favorable_tests': len(favorable_results),
                'weighted_confidence': overall_confidence,
                'test_results': [r.details for r in results],
                'champion_performance': champion.performance.to_dict(),
                'challenger_performance': challenger.performance.to_dict()
            }
        )
