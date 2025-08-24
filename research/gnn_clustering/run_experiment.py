"""
Main runner for GNN clustering experiments.
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from experiments.cluster_experiments import ClusteringExperiment, ExperimentConfig, HyperparameterOptimization
from visualization.cluster_viz import quick_visualize_results


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'gnn_clustering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def create_config_from_args(args) -> ExperimentConfig:
    """Create experiment configuration from command line arguments."""
    
    # Default symbol list (top 20 crypto assets)
    default_symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
        'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SHIB/USDT',
        'MATIC/USDT', 'UNI/USDT', 'LINK/USDT', 'ATOM/USDT', 'LTC/USDT',
        'BCH/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'FIL/USDT'
    ]
    
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = default_symbols
    
    config = ExperimentConfig(
        # Data parameters
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        correlation_window=args.correlation_window,
        
        # Graph construction
        edge_threshold=args.edge_threshold,
        max_edges=args.max_edges,
        window_size=args.window_size,
        stride=args.stride,
        
        # GNN model parameters
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        dropout=args.dropout,
        use_attention=args.use_attention,
        
        # Training parameters
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        
        # Clustering parameters
        min_cluster_size=args.min_cluster_size,
        max_clusters=args.max_clusters,
        stability_window=args.stability_window,
        coherence_threshold=args.coherence_threshold,
        
        # Experiment tracking
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb
    )
    
    return config


async def run_single_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single clustering experiment."""
    experiment = ClusteringExperiment(config)
    results = await experiment.run_full_experiment()
    return results


async def run_hyperparameter_optimization(base_config: ExperimentConfig, n_trials: int = 50) -> Dict[str, Any]:
    """Run hyperparameter optimization."""
    optimizer = HyperparameterOptimization(base_config, n_trials)
    results = optimizer.optimize()
    return results


def run_quick_demo():
    """Run a quick demo with minimal data."""
    print("🚀 Running GNN Clustering Quick Demo")
    print("====================================")
    
    # Create minimal configuration for demo
    demo_config = ExperimentConfig(
        symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT'],
        start_date=(datetime.now() - timedelta(days=7)).isoformat(),
        end_date=datetime.now().isoformat(),
        timeframe="1h",
        window_size=50,  # Smaller window for quick demo
        num_epochs=20,   # Fewer epochs
        batch_size=16,   # Smaller batch
        experiment_name="gnn_demo",
        save_dir="demo_results"
    )
    
    print(f"Configuration:")
    print(f"  Symbols: {demo_config.symbols}")
    print(f"  Date range: {demo_config.start_date} to {demo_config.end_date}")
    print(f"  Window size: {demo_config.window_size}")
    print(f"  Epochs: {demo_config.num_epochs}")
    
    try:
        # Run experiment
        results = asyncio.run(run_single_experiment(demo_config))
        
        print("\n📊 Results Summary:")
        print(f"  Training completed: {results['training_metrics']['total_epochs']} epochs")
        print(f"  Cluster results: {results['cluster_results']} time steps")
        print(f"  Arbitrage opportunities: {results['arbitrage_opportunities']}")
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"  Average silhouette score: {metrics.get('avg_silhouette_score', 0):.3f}")
            print(f"  Average stability score: {metrics.get('avg_stability_score', 0):.3f}")
            print(f"  Average clusters: {metrics.get('avg_num_clusters', 0):.1f}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Results saved to: {demo_config.save_dir}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GNN Crypto Clustering Experiments')
    
    # Mode selection
    parser.add_argument('--mode', choices=['experiment', 'optimize', 'demo'], default='demo',
                       help='Experiment mode (default: demo)')
    
    # Data parameters
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--start-date', type=str, default=(datetime.now() - timedelta(days=30)).isoformat(),
                       help='Start date (ISO format)')
    parser.add_argument('--end-date', type=str, default=datetime.now().isoformat(),
                       help='End date (ISO format)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--correlation-window', type=int, default=24, help='Correlation window (default: 24)')
    
    # Graph construction
    parser.add_argument('--edge-threshold', type=float, default=0.3, help='Edge threshold (default: 0.3)')
    parser.add_argument('--max-edges', type=int, default=200, help='Max edges (default: 200)')
    parser.add_argument('--window-size', type=int, default=100, help='Window size (default: 100)')
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension (default: 128)')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension (default: 64)')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers (default: 3)')
    parser.add_argument('--conv-type', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE'],
                       help='Convolution type (default: GCN)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanism')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    
    # Clustering parameters
    parser.add_argument('--min-cluster-size', type=int, default=3, help='Min cluster size (default: 3)')
    parser.add_argument('--max-clusters', type=int, default=15, help='Max clusters (default: 15)')
    parser.add_argument('--stability-window', type=int, default=5, help='Stability window (default: 5)')
    parser.add_argument('--coherence-threshold', type=float, default=0.7, help='Coherence threshold (default: 0.7)')
    
    # Optimization parameters
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials (default: 50)')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default='crypto_gnn_clustering',
                       help='Experiment name (default: crypto_gnn_clustering)')
    parser.add_argument('--save-dir', type=str, default='experiments/results',
                       help='Save directory (default: experiments/results)')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    if args.mode == 'demo':
        run_quick_demo()
        return
    
    # Create configuration
    config = create_config_from_args(args)
    
    logger.info(f"Starting GNN clustering experiment in {args.mode} mode")
    logger.info(f"Configuration: {config.to_dict()}")
    
    try:
        if args.mode == 'experiment':
            # Run single experiment
            results = asyncio.run(run_single_experiment(config))
            
            logger.info("Experiment completed successfully")
            logger.info(f"Results summary: {json.dumps(results, indent=2, default=str)}")
            
        elif args.mode == 'optimize':
            # Run hyperparameter optimization
            results = asyncio.run(run_hyperparameter_optimization(config, args.n_trials))
            
            logger.info("Hyperparameter optimization completed")
            logger.info(f"Best parameters: {results['best_params']}")
            logger.info(f"Best score: {results['best_value']:.4f}")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
