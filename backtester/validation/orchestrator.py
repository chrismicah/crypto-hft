"""CPCV Orchestrator for running backtests across all cross-validation splits."""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path
import concurrent.futures
from statistics import mean, stdev

from ..engine import BacktestEngine
from ..models import BacktestConfig
from ..data_loader import DataLoader
from ..metrics import PerformanceMetrics
from .cpcv import CPCVSplit, TimeSeriesSplitter, CPCVValidator
from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVResult:
    """Results from a single CPCV split."""
    split_id: int
    split_config: CPCVSplit
    performance_metrics: PerformanceMetrics
    backtest_results: Dict[str, Any]
    execution_time_seconds: float
    train_data_points: int
    test_data_points: int
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class CPCVAggregatedResults:
    """Aggregated results from all CPCV splits."""
    individual_results: List[CPCVResult] = field(default_factory=list)
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'individual_results': [
                {
                    'split_id': result.split_id,
                    'performance_metrics': result.performance_metrics.to_dict(),
                    'execution_time_seconds': result.execution_time_seconds,
                    'train_data_points': result.train_data_points,
                    'test_data_points': result.test_data_points,
                    'success': result.success,
                    'error_message': result.error_message
                }
                for result in self.individual_results
            ],
            'aggregated_metrics': self.aggregated_metrics,
            'validation_results': self.validation_results,
            'execution_summary': self.execution_summary
        }


class CPCVOrchestrator:
    """Orchestrator for running CPCV backtests."""
    
    def __init__(
        self,
        base_config: BacktestConfig,
        data_loader: DataLoader,
        strategy_callback: Callable,
        splitter: Optional[TimeSeriesSplitter] = None,
        max_parallel_jobs: int = 4,
        save_individual_results: bool = True,
        results_directory: Optional[str] = None
    ):
        """
        Initialize CPCV orchestrator.
        
        Args:
            base_config: Base backtesting configuration
            data_loader: Data loader for historical data
            strategy_callback: Strategy callback function
            splitter: Time series splitter (default: creates one)
            max_parallel_jobs: Maximum parallel backtest jobs
            save_individual_results: Whether to save individual split results
            results_directory: Directory to save results
        """
        self.base_config = base_config
        self.data_loader = data_loader
        self.strategy_callback = strategy_callback
        self.splitter = splitter or TimeSeriesSplitter()
        self.max_parallel_jobs = max_parallel_jobs
        self.save_individual_results = save_individual_results
        self.results_directory = Path(results_directory) if results_directory else Path("cpcv_results")
        
        # Create results directory
        self.results_directory.mkdir(exist_ok=True)
        
        # Validator
        self.validator = CPCVValidator()
        
        logger.info("CPCV Orchestrator initialized",
                   max_parallel_jobs=max_parallel_jobs,
                   results_directory=str(self.results_directory))
    
    async def run_cpcv(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> CPCVAggregatedResults:
        """
        Run complete CPCV analysis.
        
        Args:
            data: Historical data for backtesting
            timestamp_col: Name of timestamp column
            
        Returns:
            Aggregated CPCV results
        """
        start_time = datetime.utcnow()
        
        logger.info("Starting CPCV analysis",
                   data_points=len(data),
                   start_date=data[timestamp_col].min(),
                   end_date=data[timestamp_col].max())
        
        try:
            # Generate CPCV splits
            splits = self.splitter.split(data, timestamp_col)
            
            if not splits:
                raise ValueError("No valid CPCV splits generated")
            
            # Validate splits
            validation_results = self.validator.validate_splits(splits, data, timestamp_col)
            
            if validation_results['leakage_detected']:
                logger.warning("Potential information leakage detected in splits")
            
            # Run backtests for all splits
            individual_results = await self._run_all_splits(splits, data, timestamp_col)
            
            # Aggregate results
            aggregated_results = self._aggregate_results(individual_results)
            
            # Create final results object
            results = CPCVAggregatedResults(
                individual_results=individual_results,
                aggregated_metrics=aggregated_results,
                validation_results=validation_results,
                execution_summary={
                    'total_splits': len(splits),
                    'successful_splits': sum(1 for r in individual_results if r.success),
                    'failed_splits': sum(1 for r in individual_results if not r.success),
                    'total_execution_time_seconds': (datetime.utcnow() - start_time).total_seconds(),
                    'average_split_time_seconds': mean([r.execution_time_seconds for r in individual_results if r.success]) if individual_results else 0
                }
            )
            
            # Save results
            if self.save_individual_results:
                await self._save_results(results)
            
            logger.info("CPCV analysis completed",
                       successful_splits=results.execution_summary['successful_splits'],
                       failed_splits=results.execution_summary['failed_splits'],
                       total_time_seconds=results.execution_summary['total_execution_time_seconds'])
            
            return results
            
        except Exception as e:
            logger.error("CPCV analysis failed", error=str(e), exc_info=True)
            raise
    
    async def _run_all_splits(
        self,
        splits: List[CPCVSplit],
        data: pd.DataFrame,
        timestamp_col: str
    ) -> List[CPCVResult]:
        """Run backtests for all splits."""
        logger.info("Running backtests for all splits", total_splits=len(splits))
        
        # Prepare tasks for parallel execution
        tasks = []
        
        for split in splits:
            task = self._run_single_split(split, data, timestamp_col)
            tasks.append(task)
        
        # Execute with limited parallelism
        results = []
        
        # Process in batches to limit parallel jobs
        for i in range(0, len(tasks), self.max_parallel_jobs):
            batch = tasks[i:i + self.max_parallel_jobs]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Split execution failed", error=str(result))
                    # Create failed result
                    failed_result = CPCVResult(
                        split_id=-1,
                        split_config=None,
                        performance_metrics=PerformanceMetrics(),
                        backtest_results={},
                        execution_time_seconds=0,
                        train_data_points=0,
                        test_data_points=0,
                        success=False,
                        error_message=str(result)
                    )
                    results.append(failed_result)
                else:
                    results.append(result)
        
        successful_results = [r for r in results if r.success]
        logger.info("Split execution completed",
                   successful=len(successful_results),
                   failed=len(results) - len(successful_results))
        
        return results
    
    async def _run_single_split(
        self,
        split: CPCVSplit,
        data: pd.DataFrame,
        timestamp_col: str
    ) -> CPCVResult:
        """Run backtest for a single split."""
        split_start_time = datetime.utcnow()
        
        try:
            logger.debug("Running split", split_id=split.split_id)
            
            # Get timestamps
            timestamps = pd.to_datetime(data[timestamp_col])
            
            # Create train and test datasets
            train_mask = split.get_train_mask(timestamps)
            test_mask = split.get_test_mask(timestamps)
            
            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()
            
            if len(train_data) == 0 or len(test_data) == 0:
                raise ValueError(f"Split {split.split_id}: Empty train or test data")
            
            # Create split-specific config
            split_config = self._create_split_config(split, test_data, timestamp_col)
            
            # Create data loader for this split
            split_data_loader = self._create_split_data_loader(train_data, test_data)
            
            # Run backtest
            engine = BacktestEngine(split_config, split_data_loader, self.strategy_callback)
            await engine.run()
            
            # Get results
            backtest_results = engine.get_results()
            performance_metrics = engine.performance_calculator.calculate_metrics()
            
            execution_time = (datetime.utcnow() - split_start_time).total_seconds()
            
            result = CPCVResult(
                split_id=split.split_id,
                split_config=split,
                performance_metrics=performance_metrics,
                backtest_results=backtest_results,
                execution_time_seconds=execution_time,
                train_data_points=len(train_data),
                test_data_points=len(test_data),
                success=True
            )
            
            logger.debug("Split completed successfully",
                        split_id=split.split_id,
                        execution_time=execution_time,
                        total_return=f"{performance_metrics.total_return:.2%}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - split_start_time).total_seconds()
            
            logger.error("Split execution failed",
                        split_id=split.split_id,
                        error=str(e),
                        execution_time=execution_time)
            
            return CPCVResult(
                split_id=split.split_id,
                split_config=split,
                performance_metrics=PerformanceMetrics(),
                backtest_results={},
                execution_time_seconds=execution_time,
                train_data_points=0,
                test_data_points=0,
                success=False,
                error_message=str(e)
            )
    
    def _create_split_config(
        self,
        split: CPCVSplit,
        test_data: pd.DataFrame,
        timestamp_col: str
    ) -> BacktestConfig:
        """Create backtest config for a specific split."""
        test_start = test_data[timestamp_col].min()
        test_end = test_data[timestamp_col].max()
        
        # Create new config based on base config
        split_config = BacktestConfig(
            start_time=test_start,
            end_time=test_end,
            initial_capital=self.base_config.initial_capital,
            symbols=self.base_config.symbols,
            slippage_model=self.base_config.slippage_model,
            latency_model=self.base_config.latency_model,
            fee_model=self.base_config.fee_model,
            market_impact_factor=self.base_config.market_impact_factor,
            enable_short_selling=self.base_config.enable_short_selling,
            max_position_size=self.base_config.max_position_size,
            risk_free_rate=self.base_config.risk_free_rate
        )
        
        return split_config
    
    def _create_split_data_loader(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> DataLoader:
        """Create data loader for a specific split."""
        # For now, we'll use the test data for backtesting
        # In a more sophisticated implementation, we might use train data
        # to fit models and then test on test data
        
        # This is a simplified implementation - in practice, you'd want to
        # create a custom data source that uses the split data
        return self.data_loader
    
    def _aggregate_results(self, individual_results: List[CPCVResult]) -> Dict[str, Any]:
        """Aggregate results from all splits."""
        successful_results = [r for r in individual_results if r.success]
        
        if not successful_results:
            logger.warning("No successful results to aggregate")
            return {}
        
        # Extract metrics from all successful results
        metrics_list = [r.performance_metrics for r in successful_results]
        
        # Calculate statistics for each metric
        aggregated = {}
        
        # Get all metric names from the first result
        if metrics_list:
            sample_metrics = metrics_list[0].to_dict()
            
            for metric_name in sample_metrics.keys():
                values = []
                
                for metrics in metrics_list:
                    metric_dict = metrics.to_dict()
                    if metric_name in metric_dict and metric_dict[metric_name] is not None:
                        try:
                            # Only aggregate numeric values
                            value = float(metric_dict[metric_name])
                            if not (np.isnan(value) or np.isinf(value)):
                                values.append(value)
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    aggregated[metric_name] = {
                        'mean': mean(values),
                        'std': stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'median': sorted(values)[len(values) // 2],
                        'count': len(values)
                    }
        
        # Add additional aggregate statistics
        aggregated['summary'] = {
            'total_splits': len(individual_results),
            'successful_splits': len(successful_results),
            'success_rate': len(successful_results) / len(individual_results) if individual_results else 0,
            'average_execution_time': mean([r.execution_time_seconds for r in successful_results]),
            'total_train_data_points': sum([r.train_data_points for r in successful_results]),
            'total_test_data_points': sum([r.test_data_points for r in successful_results])
        }
        
        # Calculate consistency metrics
        if len(successful_results) > 1:
            returns = [r.performance_metrics.total_return for r in successful_results]
            sharpe_ratios = [r.performance_metrics.sharpe_ratio for r in successful_results if r.performance_metrics.sharpe_ratio != 0]
            
            aggregated['consistency'] = {
                'return_consistency': 1 - (stdev(returns) / abs(mean(returns))) if returns and mean(returns) != 0 else 0,
                'positive_return_rate': sum(1 for r in returns if r > 0) / len(returns),
                'sharpe_consistency': 1 - (stdev(sharpe_ratios) / abs(mean(sharpe_ratios))) if sharpe_ratios and mean(sharpe_ratios) != 0 else 0
            }
        
        logger.info("Results aggregated",
                   successful_splits=len(successful_results),
                   mean_return=f"{aggregated.get('total_return', {}).get('mean', 0):.2%}",
                   mean_sharpe=f"{aggregated.get('sharpe_ratio', {}).get('mean', 0):.2f}")
        
        return aggregated
    
    async def _save_results(self, results: CPCVAggregatedResults) -> None:
        """Save CPCV results to files."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated results
        aggregated_file = self.results_directory / f"cpcv_aggregated_{timestamp}.json"
        
        with open(aggregated_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save individual results if requested
        if self.save_individual_results:
            individual_dir = self.results_directory / f"individual_{timestamp}"
            individual_dir.mkdir(exist_ok=True)
            
            for result in results.individual_results:
                if result.success:
                    result_file = individual_dir / f"split_{result.split_id:03d}.json"
                    
                    result_data = {
                        'split_id': result.split_id,
                        'performance_metrics': result.performance_metrics.to_dict(),
                        'execution_time_seconds': result.execution_time_seconds,
                        'train_data_points': result.train_data_points,
                        'test_data_points': result.test_data_points,
                        'backtest_summary': {
                            'total_trades': len(result.backtest_results.get('orders', [])),
                            'final_portfolio_value': result.backtest_results.get('metrics', {}).get('final_portfolio_value', 0)
                        }
                    }
                    
                    with open(result_file, 'w') as f:
                        json.dump(result_data, f, indent=2, default=str)
        
        logger.info("CPCV results saved",
                   aggregated_file=str(aggregated_file),
                   individual_results_saved=self.save_individual_results)
    
    def generate_report(self, results: CPCVAggregatedResults) -> str:
        """Generate a comprehensive CPCV report."""
        report = f"""
COMBINATORIAL PURGED CROSS-VALIDATION REPORT
============================================

Execution Summary:
- Total Splits: {results.execution_summary['total_splits']}
- Successful Splits: {results.execution_summary['successful_splits']}
- Failed Splits: {results.execution_summary['failed_splits']}
- Success Rate: {results.execution_summary['successful_splits'] / results.execution_summary['total_splits']:.1%}
- Total Execution Time: {results.execution_summary['total_execution_time_seconds']:.1f} seconds
- Average Split Time: {results.execution_summary['average_split_time_seconds']:.1f} seconds

Validation Results:
- Leakage Detected: {'Yes' if results.validation_results.get('leakage_detected', False) else 'No'}
- Valid Splits: {results.validation_results.get('valid_splits', 0)}
- Train Coverage: {results.validation_results.get('coverage_stats', {}).get('train_coverage_pct', 0):.1f}%
- Test Coverage: {results.validation_results.get('coverage_stats', {}).get('test_coverage_pct', 0):.1f}%

Performance Metrics (Aggregated):
"""
        
        # Add aggregated metrics
        if 'total_return' in results.aggregated_metrics:
            return_stats = results.aggregated_metrics['total_return']
            report += f"""
Return Statistics:
- Mean Return: {return_stats['mean']:.2%}
- Std Return: {return_stats['std']:.2%}
- Min Return: {return_stats['min']:.2%}
- Max Return: {return_stats['max']:.2%}
- Median Return: {return_stats['median']:.2%}
"""
        
        if 'sharpe_ratio' in results.aggregated_metrics:
            sharpe_stats = results.aggregated_metrics['sharpe_ratio']
            report += f"""
Sharpe Ratio Statistics:
- Mean Sharpe: {sharpe_stats['mean']:.2f}
- Std Sharpe: {sharpe_stats['std']:.2f}
- Min Sharpe: {sharpe_stats['min']:.2f}
- Max Sharpe: {sharpe_stats['max']:.2f}
- Median Sharpe: {sharpe_stats['median']:.2f}
"""
        
        if 'max_drawdown' in results.aggregated_metrics:
            dd_stats = results.aggregated_metrics['max_drawdown']
            report += f"""
Max Drawdown Statistics:
- Mean Drawdown: {dd_stats['mean']:.2%}
- Std Drawdown: {dd_stats['std']:.2%}
- Min Drawdown: {dd_stats['min']:.2%}
- Max Drawdown: {dd_stats['max']:.2%}
- Median Drawdown: {dd_stats['median']:.2%}
"""
        
        # Add consistency metrics
        if 'consistency' in results.aggregated_metrics:
            consistency = results.aggregated_metrics['consistency']
            report += f"""
Consistency Metrics:
- Return Consistency: {consistency.get('return_consistency', 0):.2f}
- Positive Return Rate: {consistency.get('positive_return_rate', 0):.1%}
- Sharpe Consistency: {consistency.get('sharpe_consistency', 0):.2f}
"""
        
        # Add individual split summary
        successful_results = [r for r in results.individual_results if r.success]
        if successful_results:
            report += f"""
Individual Split Results:
"""
            for result in successful_results[:10]:  # Show first 10 splits
                report += f"- Split {result.split_id}: Return {result.performance_metrics.total_return:.2%}, Sharpe {result.performance_metrics.sharpe_ratio:.2f}\n"
            
            if len(successful_results) > 10:
                report += f"... and {len(successful_results) - 10} more splits\n"
        
        return report
