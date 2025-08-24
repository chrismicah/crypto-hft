"""
Experimental framework for GNN-based cluster detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
import os
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import wandb
import optuna

from ..models.graph_models import (
    CryptoGNN, DynamicClusterDetector, ArbitrageOpportunityDetector,
    GraphSnapshot, ClusterResult
)
from ..data.graph_constructor import CryptoDataCollector, GraphConstructor


@dataclass
class ExperimentConfig:
    """Configuration for clustering experiments."""
    
    # Data parameters
    symbols: List[str]
    start_date: str
    end_date: str
    timeframe: str = "1h"
    correlation_window: int = 24
    
    # Graph construction
    edge_threshold: float = 0.3
    max_edges: int = 200
    window_size: int = 100
    stride: int = 1
    
    # GNN model parameters
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 3
    conv_type: str = "GCN"
    dropout: float = 0.1
    use_attention: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    
    # Clustering parameters
    min_cluster_size: int = 3
    max_clusters: int = 15
    stability_window: int = 5
    coherence_threshold: float = 0.7
    
    # Experiment tracking
    experiment_name: str = "crypto_gnn_clustering"
    save_dir: str = "experiments/results"
    use_wandb: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ClusteringExperiment:
    """
    Main experimental framework for GNN-based crypto clustering.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_collector = CryptoDataCollector(
            symbols=config.symbols,
            timeframe=config.timeframe
        )
        
        self.graph_constructor = GraphConstructor(
            correlation_window=config.correlation_window,
            edge_threshold=config.edge_threshold,
            max_edges=config.max_edges
        )
        
        self.cluster_detector = DynamicClusterDetector(
            min_cluster_size=config.min_cluster_size,
            max_clusters=config.max_clusters,
            stability_window=config.stability_window,
            coherence_threshold=config.coherence_threshold
        )
        
        self.arbitrage_detector = ArbitrageOpportunityDetector()
        
        # Model and training
        self.model: Optional[CryptoGNN] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.results = {
            'cluster_results': [],
            'arbitrage_opportunities': [],
            'model_performance': {},
            'temporal_stability': [],
            'profitability_analysis': {}
        }
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="crypto-gnn-clustering",
                name=config.experiment_name,
                config=config.to_dict()
            )
    
    async def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete clustering experiment."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        try:
            # Step 1: Data Collection
            self.logger.info("Step 1: Collecting market data")
            market_data = await self._collect_data()
            
            # Step 2: Graph Construction
            self.logger.info("Step 2: Constructing dynamic graphs")
            graph_sequence = self._construct_graphs(market_data)
            
            if not graph_sequence:
                raise ValueError("No graphs could be constructed from the data")
            
            # Step 3: Model Training
            self.logger.info("Step 3: Training GNN model")
            self._initialize_model(graph_sequence[0])
            training_metrics = self._train_model(graph_sequence)
            
            # Step 4: Cluster Analysis
            self.logger.info("Step 4: Performing cluster analysis")
            cluster_results = self._analyze_clusters(graph_sequence)
            
            # Step 5: Arbitrage Detection
            self.logger.info("Step 5: Detecting arbitrage opportunities")
            arbitrage_results = self._detect_arbitrage_opportunities(
                cluster_results, market_data
            )
            
            # Step 6: Performance Evaluation
            self.logger.info("Step 6: Evaluating performance")
            performance_metrics = self._evaluate_performance(
                cluster_results, arbitrage_results, market_data
            )
            
            # Step 7: Save Results
            self._save_results()
            
            self.logger.info("Experiment completed successfully")
            
            return {
                'config': self.config.to_dict(),
                'training_metrics': training_metrics,
                'cluster_results': len(cluster_results),
                'arbitrage_opportunities': len(arbitrage_results),
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
    
    async def _collect_data(self) -> Dict[str, List]:
        """Collect market data for all symbols."""
        start_date = datetime.fromisoformat(self.config.start_date)
        end_date = datetime.fromisoformat(self.config.end_date)
        
        market_data = await self.data_collector.collect_historical_data(
            start_date, end_date
        )
        
        # Filter out symbols with insufficient data
        min_data_points = 100
        filtered_data = {
            symbol: data for symbol, data in market_data.items()
            if len(data) >= min_data_points
        }
        
        self.logger.info(f"Collected data for {len(filtered_data)} symbols")
        return filtered_data
    
    def _construct_graphs(self, market_data: Dict[str, List]) -> List[GraphSnapshot]:
        """Construct graph sequence from market data."""
        
        # Convert to aligned DataFrame
        aligned_df = self.data_collector.get_aligned_dataframe(market_data)
        
        if aligned_df.empty:
            self.logger.error("Failed to create aligned DataFrame")
            return []
        
        # Construct graph sequence
        graph_sequence = self.graph_constructor.construct_graph_sequence(
            aligned_df,
            window_size=self.config.window_size,
            stride=self.config.stride
        )
        
        self.logger.info(f"Constructed {len(graph_sequence)} graph snapshots")
        return graph_sequence
    
    def _initialize_model(self, sample_graph: GraphSnapshot):
        """Initialize GNN model based on sample graph."""
        num_node_features = sample_graph.node_features.shape[1]
        
        self.model = CryptoGNN(
            num_node_features=num_node_features,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim,
            num_layers=self.config.num_layers,
            conv_type=self.config.conv_type,
            dropout=self.config.dropout,
            use_attention=self.config.use_attention
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.logger.info(f"Initialized {self.config.conv_type} model with {num_node_features} input features")
    
    def _train_model(self, graph_sequence: List[GraphSnapshot]) -> Dict[str, float]:
        """Train the GNN model using self-supervised learning."""
        
        # Convert graphs to PyTorch Geometric data
        pyg_data = [graph.to_pyg_data().to(self.device) for graph in graph_sequence]
        
        # Split into train/validation
        split_idx = int(0.8 * len(pyg_data))
        train_data = pyg_data[:split_idx]
        val_data = pyg_data[split_idx:]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_metrics = []
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            
            training_metrics.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model_checkpoint()
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    })
        
        # Load best model
        self._load_model_checkpoint()
        
        return {
            'final_train_loss': training_metrics[-1]['train_loss'],
            'final_val_loss': training_metrics[-1]['val_loss'],
            'best_val_loss': best_val_loss,
            'total_epochs': len(training_metrics)
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr.squeeze() if batch.edge_attr is not None else None,
                batch=batch.batch
            )
            
            # Self-supervised loss (temporal consistency + clustering quality)
            loss = self._calculate_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_weight=batch.edge_attr.squeeze() if batch.edge_attr is not None else None,
                    batch=batch.batch
                )
                
                loss = self._calculate_loss(outputs, batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], batch) -> torch.Tensor:
        """Calculate self-supervised loss."""
        
        # Reconstruction loss (node features)
        node_embeddings = outputs['node_embeddings']
        
        # Temporal consistency loss (if we have sequential data)
        temporal_loss = torch.tensor(0.0, device=self.device)
        
        # Clustering quality loss (encourage diverse but coherent clusters)
        clustering_embeddings = outputs['clustering_embeddings']
        
        # Intra-cluster similarity (encourage similar nodes to be close)
        pairwise_distances = torch.cdist(clustering_embeddings, clustering_embeddings)
        
        # Use edge information as supervision
        if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0:
            edge_idx = batch.edge_index
            edge_distances = pairwise_distances[edge_idx[0], edge_idx[1]]
            similarity_loss = edge_distances.mean()
        else:
            similarity_loss = torch.tensor(0.0, device=self.device)
        
        # Diversity loss (encourage different clusters to be far apart)
        # Use negative sampling to push random pairs apart
        batch_size = clustering_embeddings.shape[0]
        if batch_size > 1:
            random_pairs = torch.randperm(batch_size)[:batch_size//2]
            diversity_loss = -pairwise_distances[random_pairs, random_pairs].mean()
        else:
            diversity_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = similarity_loss + 0.1 * diversity_loss + 0.01 * temporal_loss
        
        return total_loss
    
    def _save_model_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}_best_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict()
        }, checkpoint_path)
    
    def _load_model_checkpoint(self):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}_best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def _analyze_clusters(self, graph_sequence: List[GraphSnapshot]) -> List[ClusterResult]:
        """Analyze clusters in the graph sequence."""
        cluster_results = []
        
        self.model.eval()
        with torch.no_grad():
            for i, graph in enumerate(graph_sequence):
                # Get embeddings from model
                pyg_data = graph.to_pyg_data().to(self.device)
                outputs = self.model(
                    x=pyg_data.x,
                    edge_index=pyg_data.edge_index,
                    edge_weight=pyg_data.edge_attr.squeeze() if pyg_data.edge_attr is not None else None
                )
                
                embeddings = outputs['clustering_embeddings'].cpu().numpy()
                
                # Detect clusters
                cluster_labels, metrics = self.cluster_detector.detect_clusters(
                    embeddings, graph.correlation_matrix
                )
                
                # Calculate temporal stability
                stability_score = self.cluster_detector.calculate_temporal_stability(
                    cluster_labels, self.cluster_detector.cluster_history
                )
                
                # Update history
                self.cluster_detector.update_history(cluster_labels)
                
                # Create cluster result
                cluster_symbols = self._group_symbols_by_cluster(cluster_labels, graph.symbols)
                cluster_result = ClusterResult(
                    timestamp=graph.timestamp,
                    cluster_labels=cluster_labels,
                    cluster_centers=embeddings,  # Use embeddings as centers
                    silhouette_score=metrics.get('silhouette_score', 0.0),
                    num_clusters=metrics.get('num_clusters', 0),
                    stability_score=stability_score,
                    cluster_sizes=[len(symbols) for symbols in cluster_symbols],
                    cluster_symbols=cluster_symbols,
                    cluster_coherence=self._calculate_cluster_coherence(
                        cluster_labels, graph.correlation_matrix
                    ),
                    arbitrage_pairs=[],  # Will be filled later
                    cluster_spreads={}
                )
                
                cluster_results.append(cluster_result)
                
                if i % 100 == 0:
                    self.logger.info(f"Analyzed {i+1} graphs")
        
        self.logger.info(f"Completed cluster analysis for {len(cluster_results)} graphs")
        return cluster_results
    
    def _group_symbols_by_cluster(
        self,
        cluster_labels: np.ndarray,
        symbols: List[str]
    ) -> List[List[str]]:
        """Group symbols by their cluster assignments."""
        cluster_dict = {}
        
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Ignore noise points (-1)
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(symbols[i])
        
        return list(cluster_dict.values())
    
    def _calculate_cluster_coherence(
        self,
        cluster_labels: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> List[float]:
        """Calculate coherence score for each cluster."""
        coherence_scores = []
        
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            
            if len(cluster_indices) > 1:
                # Calculate average correlation within cluster
                cluster_corrs = []
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i != j:
                            cluster_corrs.append(abs(correlation_matrix[i, j]))
                
                coherence = np.mean(cluster_corrs) if cluster_corrs else 0.0
                coherence_scores.append(coherence)
            else:
                coherence_scores.append(1.0)  # Single-node cluster is perfectly coherent
        
        return coherence_scores
    
    def _detect_arbitrage_opportunities(
        self,
        cluster_results: List[ClusterResult],
        market_data: Dict[str, List]
    ) -> List[Dict]:
        """Detect arbitrage opportunities from clustering results."""
        
        # Prepare price and volume data
        price_data = {}
        volume_data = {}
        
        for symbol, data_points in market_data.items():
            if data_points:
                prices = [dp.close for dp in data_points]
                volumes = [dp.volume for dp in data_points]
                price_data[symbol.replace('/', '_')] = np.array(prices)
                volume_data[symbol.replace('/', '_')] = np.array(volumes)
        
        all_opportunities = []
        
        for i, cluster_result in enumerate(cluster_results):
            try:
                opportunities = self.arbitrage_detector.detect_opportunities(
                    cluster_result,
                    price_data,
                    volume_data,
                    cluster_result.cluster_centers  # Use embeddings as correlation proxy
                )
                
                # Add timestamp to opportunities
                for opp in opportunities:
                    opp['timestamp'] = cluster_result.timestamp
                    opp['graph_index'] = i
                
                all_opportunities.extend(opportunities)
                
            except Exception as e:
                self.logger.warning(f"Failed to detect opportunities at {cluster_result.timestamp}: {e}")
                continue
        
        self.logger.info(f"Detected {len(all_opportunities)} arbitrage opportunities")
        return all_opportunities
    
    def _evaluate_performance(
        self,
        cluster_results: List[ClusterResult],
        arbitrage_opportunities: List[Dict],
        market_data: Dict[str, List]
    ) -> Dict[str, float]:
        """Evaluate overall performance of the clustering approach."""
        
        if not cluster_results:
            return {}
        
        # Clustering quality metrics
        avg_silhouette = np.mean([cr.silhouette_score for cr in cluster_results])
        avg_stability = np.mean([cr.stability_score for cr in cluster_results])
        avg_clusters = np.mean([cr.num_clusters for cr in cluster_results])
        
        # Temporal consistency
        stability_variance = np.var([cr.stability_score for cr in cluster_results])
        
        # Arbitrage opportunity quality
        total_opportunities = len(arbitrage_opportunities)
        
        if arbitrage_opportunities:
            avg_expected_return = np.mean([opp['expected_return'] for opp in arbitrage_opportunities])
            avg_confidence = np.mean([opp['confidence'] for opp in arbitrage_opportunities])
        else:
            avg_expected_return = 0.0
            avg_confidence = 0.0
        
        # Coverage metrics
        all_symbols = set()
        clustered_symbols = set()
        
        for cr in cluster_results:
            for cluster_symbols in cr.cluster_symbols:
                clustered_symbols.update(cluster_symbols)
        
        for symbol in market_data.keys():
            all_symbols.add(symbol.replace('/', '_'))
        
        coverage_ratio = len(clustered_symbols) / len(all_symbols) if all_symbols else 0.0
        
        performance_metrics = {
            'avg_silhouette_score': avg_silhouette,
            'avg_stability_score': avg_stability,
            'avg_num_clusters': avg_clusters,
            'stability_variance': stability_variance,
            'total_arbitrage_opportunities': total_opportunities,
            'avg_expected_return': avg_expected_return,
            'avg_opportunity_confidence': avg_confidence,
            'symbol_coverage_ratio': coverage_ratio,
            'opportunities_per_hour': total_opportunities / len(cluster_results) if cluster_results else 0
        }
        
        self.logger.info("Performance evaluation completed")
        for metric, value in performance_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return performance_metrics
    
    def _save_results(self):
        """Save experiment results."""
        results_path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}_results.json")
        
        # Convert results to serializable format
        serializable_results = {
            'config': self.config.to_dict(),
            'results': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_path}")


class HyperparameterOptimization:
    """
    Hyperparameter optimization for GNN clustering.
    """
    
    def __init__(self, base_config: ExperimentConfig, n_trials: int = 50):
        self.base_config = base_config
        self.n_trials = n_trials
        self.logger = logging.getLogger(__name__)
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization using Optuna."""
        
        def objective(trial):
            # Sample hyperparameters
            config = ExperimentConfig(**self.base_config.to_dict())
            
            config.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            config.embedding_dim = trial.suggest_categorical('embedding_dim', [32, 64, 128])
            config.num_layers = trial.suggest_int('num_layers', 2, 5)
            config.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            config.dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
            config.edge_threshold = trial.suggest_uniform('edge_threshold', 0.1, 0.6)
            config.conv_type = trial.suggest_categorical('conv_type', ['GCN', 'GAT', 'SAGE'])
            
            # Update experiment name
            config.experiment_name = f"optuna_trial_{trial.number}"
            
            # Run experiment
            experiment = ClusteringExperiment(config)
            
            try:
                import asyncio
                results = asyncio.run(experiment.run_full_experiment())
                
                # Return metric to optimize (higher is better)
                return results['performance_metrics']['avg_silhouette_score']
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {e}")
                return -1.0  # Return low score for failed trials
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Return best parameters
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
