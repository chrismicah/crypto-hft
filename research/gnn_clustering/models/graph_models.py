"""
Graph data models for crypto market representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score


@dataclass
class GraphSnapshot:
    """Represents a market graph at a specific timestamp."""
    timestamp: datetime
    node_features: torch.Tensor      # [num_nodes, num_features]
    edge_index: torch.Tensor         # [2, num_edges]
    edge_weights: torch.Tensor       # [num_edges]
    symbols: List[str]               # Node symbol mapping
    correlation_matrix: np.ndarray   # Full correlation matrix
    volatility_vector: np.ndarray    # Node volatilities
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        return Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights.unsqueeze(1),
            y=torch.zeros(len(self.symbols))  # Placeholder for labels
        )
    
    def get_networkx_graph(self) -> nx.Graph:
        """Convert to NetworkX graph for analysis."""
        pyg_data = self.to_pyg_data()
        G = to_networkx(pyg_data, to_undirected=True)
        
        # Add node attributes
        for i, symbol in enumerate(self.symbols):
            G.nodes[i]['symbol'] = symbol
            G.nodes[i]['volatility'] = float(self.volatility_vector[i])
        
        return G


@dataclass
class ClusterResult:
    """Results of clustering analysis."""
    timestamp: datetime
    cluster_labels: np.ndarray       # Cluster assignment for each node
    cluster_centers: np.ndarray      # Cluster center embeddings
    silhouette_score: float          # Clustering quality metric
    num_clusters: int                # Number of detected clusters
    stability_score: float           # Temporal stability score
    
    # Cluster-specific metrics
    cluster_sizes: List[int]         # Size of each cluster
    cluster_symbols: List[List[str]] # Symbols in each cluster
    cluster_coherence: List[float]   # Internal coherence of each cluster
    
    # Arbitrage opportunities
    arbitrage_pairs: List[Tuple[str, str, float]]  # (symbol1, symbol2, spread)
    cluster_spreads: Dict[int, float]              # Cluster internal spreads


class CryptoGraphConv(nn.Module):
    """
    Custom Graph Convolutional layer for crypto market data.
    Incorporates market-specific features like volatility and volume.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "GCN",
        dropout: float = 0.1,
        heads: int = 8  # For GAT
    ):
        super().__init__()
        self.conv_type = conv_type
        self.dropout = dropout
        
        if conv_type == "GCN":
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == "GAT":
            self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False)
        elif conv_type == "SAGE":
            self.conv = GraphSAGE(in_channels, out_channels, num_layers=2)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """Forward pass."""
        if self.conv_type == "SAGE":
            # GraphSAGE doesn't use edge weights in the same way
            x = self.conv(x, edge_index)
        else:
            x = self.conv(x, edge_index, edge_weight)
        
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class CryptoGNN(nn.Module):
    """
    Graph Neural Network for crypto market analysis.
    
    Architecture:
    1. Multiple graph convolutional layers
    2. Node-level and graph-level representations
    3. Clustering-aware embeddings
    4. Temporal consistency regularization
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 3,
        conv_type: str = "GCN",
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.conv_layers.append(
                CryptoGraphConv(in_dim, out_dim, conv_type, dropout)
            )
        
        # Attention mechanism for node importance
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Node embedding projection
        self.node_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Graph-level representation
        self.graph_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Clustering head
        self.clustering_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim // 2)
        )
        
        # Temporal consistency predictor
        self.temporal_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # Current + previous embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
            batch: Batch assignment for multiple graphs
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing embeddings and predictions
        """
        # Input projection
        h = self.input_proj(x)
        
        # Graph convolutions
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_weight)
        
        # Attention mechanism
        attention_weights = None
        if self.use_attention:
            # Reshape for attention (add sequence dimension)
            h_att = h.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            h_att, attention_weights = self.attention(h_att, h_att, h_att)
            h = h_att.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Node embeddings
        node_embeddings = self.node_embedding(h)
        
        # Graph-level embeddings (if batch is provided)
        if batch is not None:
            graph_embeddings = global_mean_pool(node_embeddings, batch)
            graph_embeddings = self.graph_embedding(graph_embeddings)
        else:
            # Single graph case
            graph_embeddings = self.graph_embedding(node_embeddings.mean(dim=0, keepdim=True))
        
        # Clustering representations
        clustering_embeddings = self.clustering_head(node_embeddings)
        
        results = {
            'node_embeddings': node_embeddings,
            'graph_embeddings': graph_embeddings,
            'clustering_embeddings': clustering_embeddings,
            'hidden_representations': h
        }
        
        if return_attention and attention_weights is not None:
            results['attention_weights'] = attention_weights
        
        return results
    
    def predict_temporal_consistency(
        self,
        current_embeddings: torch.Tensor,
        previous_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Predict temporal consistency between embeddings."""
        combined = torch.cat([current_embeddings, previous_embeddings], dim=-1)
        return self.temporal_predictor(combined)


class DynamicClusterDetector:
    """
    Dynamic cluster detection for crypto market graphs.
    
    Uses multiple clustering algorithms and temporal consistency
    to identify stable, profitable clusters.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 3,
        max_clusters: int = 20,
        stability_window: int = 5,
        coherence_threshold: float = 0.7
    ):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.stability_window = stability_window
        self.coherence_threshold = coherence_threshold
        
        # Clustering algorithms
        self.clusterers = {
            'kmeans': None,  # Will be initialized with optimal k
            'spectral': SpectralClustering(affinity='rbf'),
            'dbscan': DBSCAN(eps=0.3, min_samples=min_cluster_size)
        }
        
        # Temporal tracking
        self.cluster_history = []
        self.stability_scores = []
    
    def detect_clusters(
        self,
        embeddings: np.ndarray,
        similarity_matrix: Optional[np.ndarray] = None,
        method: str = "ensemble"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detect clusters in node embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            similarity_matrix: Precomputed similarity matrix
            method: Clustering method ('kmeans', 'spectral', 'dbscan', 'ensemble')
            
        Returns:
            Cluster labels and quality metrics
        """
        if method == "ensemble":
            return self._ensemble_clustering(embeddings, similarity_matrix)
        else:
            return self._single_method_clustering(embeddings, method, similarity_matrix)
    
    def _find_optimal_k(self, embeddings: np.ndarray, max_k: Optional[int] = None) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        if max_k is None:
            max_k = min(self.max_clusters, len(embeddings) // 2)
        
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            if k >= len(embeddings):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        return best_k
    
    def _single_method_clustering(
        self,
        embeddings: np.ndarray,
        method: str,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply single clustering method."""
        if method == "kmeans":
            optimal_k = self._find_optimal_k(embeddings)
            clusterer = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            
        elif method == "spectral":
            optimal_k = self._find_optimal_k(embeddings)
            clusterer = SpectralClustering(
                n_clusters=optimal_k,
                affinity='rbf' if similarity_matrix is None else 'precomputed',
                random_state=42
            )
            if similarity_matrix is not None:
                labels = clusterer.fit_predict(similarity_matrix)
            else:
                labels = clusterer.fit_predict(embeddings)
                
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.3, min_samples=self.min_cluster_size)
            labels = clusterer.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate quality metrics
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(embeddings, labels)
        else:
            silhouette = 0.0
        
        metrics = {
            'silhouette_score': silhouette,
            'num_clusters': len(np.unique(labels[labels >= 0])),  # Exclude noise (-1)
            'noise_ratio': np.sum(labels == -1) / len(labels) if method == "dbscan" else 0.0
        }
        
        return labels, metrics
    
    def _ensemble_clustering(
        self,
        embeddings: np.ndarray,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Ensemble clustering using multiple methods."""
        all_labels = []
        all_metrics = []
        
        # Apply each clustering method
        for method in ['kmeans', 'spectral', 'dbscan']:
            try:
                labels, metrics = self._single_method_clustering(
                    embeddings, method, similarity_matrix
                )
                all_labels.append(labels)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Clustering method {method} failed: {e}")
                continue
        
        if not all_labels:
            # Fallback to simple k-means
            optimal_k = self._find_optimal_k(embeddings)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            metrics = {
                'silhouette_score': silhouette_score(embeddings, labels),
                'num_clusters': optimal_k,
                'noise_ratio': 0.0
            }
            return labels, metrics
        
        # Consensus clustering
        final_labels = self._consensus_clustering(all_labels, embeddings)
        
        # Aggregate metrics
        avg_metrics = {
            'silhouette_score': np.mean([m['silhouette_score'] for m in all_metrics]),
            'num_clusters': len(np.unique(final_labels[final_labels >= 0])),
            'noise_ratio': np.sum(final_labels == -1) / len(final_labels),
            'consensus_score': self._calculate_consensus_score(all_labels)
        }
        
        return final_labels, avg_metrics
    
    def _consensus_clustering(
        self,
        all_labels: List[np.ndarray],
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Create consensus clustering from multiple results."""
        n_samples = len(all_labels[0])
        
        # Create co-occurrence matrix
        co_occurrence = np.zeros((n_samples, n_samples))
        
        for labels in all_labels:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j] and labels[i] >= 0:
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1
        
        # Normalize by number of methods
        co_occurrence /= len(all_labels)
        
        # Apply spectral clustering on consensus matrix
        optimal_k = self._find_optimal_k(embeddings)
        spectral = SpectralClustering(
            n_clusters=optimal_k,
            affinity='precomputed',
            random_state=42
        )
        
        # Convert co-occurrence to similarity (0-1 range)
        similarity = co_occurrence
        consensus_labels = spectral.fit_predict(similarity)
        
        return consensus_labels
    
    def _calculate_consensus_score(self, all_labels: List[np.ndarray]) -> float:
        """Calculate agreement between different clustering methods."""
        if len(all_labels) < 2:
            return 1.0
        
        scores = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                try:
                    ari = adjusted_rand_score(all_labels[i], all_labels[j])
                    scores.append(ari)
                except Exception:
                    continue
        
        return np.mean(scores) if scores else 0.0
    
    def calculate_temporal_stability(
        self,
        current_labels: np.ndarray,
        historical_labels: List[np.ndarray]
    ) -> float:
        """Calculate temporal stability of clustering."""
        if not historical_labels:
            return 1.0
        
        # Compare with recent history
        recent_history = historical_labels[-self.stability_window:]
        stability_scores = []
        
        for past_labels in recent_history:
            if len(past_labels) == len(current_labels):
                try:
                    ari = adjusted_rand_score(current_labels, past_labels)
                    stability_scores.append(ari)
                except Exception:
                    continue
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def update_history(self, labels: np.ndarray, max_history: int = 50):
        """Update clustering history for temporal analysis."""
        self.cluster_history.append(labels.copy())
        
        # Keep only recent history
        if len(self.cluster_history) > max_history:
            self.cluster_history = self.cluster_history[-max_history:]


class ArbitrageOpportunityDetector:
    """
    Detect multi-asset arbitrage opportunities within and across clusters.
    """
    
    def __init__(
        self,
        min_spread_threshold: float = 0.02,  # 2% minimum spread
        max_cluster_size: int = 10,          # Max assets in arbitrage cluster
        correlation_threshold: float = 0.8,  # Minimum correlation for pairing
        liquidity_threshold: float = 1000000 # $1M minimum daily volume
    ):
        self.min_spread_threshold = min_spread_threshold
        self.max_cluster_size = max_cluster_size
        self.correlation_threshold = correlation_threshold
        self.liquidity_threshold = liquidity_threshold
    
    def detect_opportunities(
        self,
        cluster_result: ClusterResult,
        price_data: Dict[str, np.ndarray],
        volume_data: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray
    ) -> List[Dict]:
        """
        Detect arbitrage opportunities within clusters.
        
        Args:
            cluster_result: Current clustering results
            price_data: Price time series for each symbol
            volume_data: Volume data for liquidity filtering
            correlation_matrix: Asset correlation matrix
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        # Analyze each cluster
        for cluster_id, symbols in enumerate(cluster_result.cluster_symbols):
            if len(symbols) < 2 or len(symbols) > self.max_cluster_size:
                continue
            
            # Filter by liquidity
            liquid_symbols = self._filter_by_liquidity(symbols, volume_data)
            if len(liquid_symbols) < 2:
                continue
            
            # Find pair opportunities within cluster
            cluster_opps = self._find_cluster_opportunities(
                liquid_symbols, price_data, correlation_matrix, cluster_id
            )
            opportunities.extend(cluster_opps)
        
        # Cross-cluster opportunities
        cross_opps = self._find_cross_cluster_opportunities(
            cluster_result, price_data, volume_data, correlation_matrix
        )
        opportunities.extend(cross_opps)
        
        # Rank by expected profitability
        opportunities = self._rank_opportunities(opportunities)
        
        return opportunities
    
    def _filter_by_liquidity(
        self,
        symbols: List[str],
        volume_data: Dict[str, np.ndarray]
    ) -> List[str]:
        """Filter symbols by liquidity requirements."""
        liquid_symbols = []
        
        for symbol in symbols:
            if symbol in volume_data:
                avg_volume = np.mean(volume_data[symbol][-30:])  # 30-day average
                if avg_volume >= self.liquidity_threshold:
                    liquid_symbols.append(symbol)
        
        return liquid_symbols
    
    def _find_cluster_opportunities(
        self,
        symbols: List[str],
        price_data: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        cluster_id: int
    ) -> List[Dict]:
        """Find arbitrage opportunities within a cluster."""
        opportunities = []
        
        # Get symbol indices for correlation lookup
        symbol_to_idx = {sym: i for i, sym in enumerate(price_data.keys())}
        
        # Check all pairs within cluster
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if sym1 not in symbol_to_idx or sym2 not in symbol_to_idx:
                    continue
                
                idx1, idx2 = symbol_to_idx[sym1], symbol_to_idx[sym2]
                correlation = correlation_matrix[idx1, idx2]
                
                if correlation >= self.correlation_threshold:
                    # Calculate current spread
                    prices1 = price_data[sym1]
                    prices2 = price_data[sym2]
                    
                    # Normalize prices for comparison
                    norm_prices1 = prices1 / prices1[0]
                    norm_prices2 = prices2 / prices2[0]
                    
                    current_spread = abs(norm_prices1[-1] - norm_prices2[-1])
                    
                    # Calculate historical spread statistics
                    spreads = abs(norm_prices1 - norm_prices2)
                    spread_mean = np.mean(spreads)
                    spread_std = np.std(spreads)
                    
                    # Z-score of current spread
                    if spread_std > 0:
                        spread_zscore = (current_spread - spread_mean) / spread_std
                        
                        if abs(spread_zscore) >= 2.0:  # 2 sigma threshold
                            opportunity = {
                                'type': 'intra_cluster',
                                'cluster_id': cluster_id,
                                'symbol1': sym1,
                                'symbol2': sym2,
                                'correlation': correlation,
                                'current_spread': current_spread,
                                'spread_zscore': spread_zscore,
                                'mean_reversion_signal': spread_zscore > 0,  # True if overextended
                                'expected_return': abs(spread_zscore) * spread_std,
                                'confidence': min(correlation, 1.0)
                            }
                            opportunities.append(opportunity)
        
        return opportunities
    
    def _find_cross_cluster_opportunities(
        self,
        cluster_result: ClusterResult,
        price_data: Dict[str, np.ndarray],
        volume_data: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray
    ) -> List[Dict]:
        """Find arbitrage opportunities across different clusters."""
        opportunities = []
        
        # Only consider clusters with reasonable size
        valid_clusters = [
            (i, symbols) for i, symbols in enumerate(cluster_result.cluster_symbols)
            if 2 <= len(symbols) <= self.max_cluster_size
        ]
        
        symbol_to_idx = {sym: i for i, sym in enumerate(price_data.keys())}
        
        # Compare clusters pairwise
        for i, (cluster1_id, cluster1_symbols) in enumerate(valid_clusters):
            for j, (cluster2_id, cluster2_symbols) in enumerate(valid_clusters[i+1:], i+1):
                
                # Find best representative from each cluster
                rep1 = self._find_cluster_representative(
                    cluster1_symbols, volume_data, price_data
                )
                rep2 = self._find_cluster_representative(
                    cluster2_symbols, volume_data, price_data
                )
                
                if rep1 and rep2 and rep1 in symbol_to_idx and rep2 in symbol_to_idx:
                    idx1, idx2 = symbol_to_idx[rep1], symbol_to_idx[rep2]
                    correlation = correlation_matrix[idx1, idx2]
                    
                    # Look for negative correlation (divergence opportunities)
                    if correlation <= -0.5:  # Strong negative correlation
                        
                        prices1 = price_data[rep1]
                        prices2 = price_data[rep2]
                        
                        # Calculate cluster momentum
                        momentum1 = (prices1[-1] / prices1[-10] - 1) if len(prices1) >= 10 else 0
                        momentum2 = (prices2[-1] / prices2[-10] - 1) if len(prices2) >= 10 else 0
                        
                        # Divergence signal
                        momentum_diff = abs(momentum1 - momentum2)
                        
                        if momentum_diff >= 0.05:  # 5% momentum difference
                            opportunity = {
                                'type': 'cross_cluster',
                                'cluster1_id': cluster1_id,
                                'cluster2_id': cluster2_id,
                                'symbol1': rep1,
                                'symbol2': rep2,
                                'correlation': correlation,
                                'momentum1': momentum1,
                                'momentum2': momentum2,
                                'momentum_divergence': momentum_diff,
                                'expected_return': momentum_diff * 0.5,  # Conservative estimate
                                'confidence': abs(correlation)
                            }
                            opportunities.append(opportunity)
        
        return opportunities
    
    def _find_cluster_representative(
        self,
        symbols: List[str],
        volume_data: Dict[str, np.ndarray],
        price_data: Dict[str, np.ndarray]
    ) -> Optional[str]:
        """Find the most liquid/representative symbol in a cluster."""
        best_symbol = None
        best_volume = 0
        
        for symbol in symbols:
            if symbol in volume_data and symbol in price_data:
                avg_volume = np.mean(volume_data[symbol][-30:])
                if avg_volume > best_volume:
                    best_volume = avg_volume
                    best_symbol = symbol
        
        return best_symbol
    
    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by expected profitability and confidence."""
        def score_opportunity(opp):
            base_score = opp['expected_return'] * opp['confidence']
            
            # Bonus for intra-cluster opportunities (more stable)
            if opp['type'] == 'intra_cluster':
                base_score *= 1.2
            
            # Penalty for very large spreads (may indicate fundamental change)
            if 'spread_zscore' in opp and abs(opp['spread_zscore']) > 3:
                base_score *= 0.8
            
            return base_score
        
        return sorted(opportunities, key=score_opportunity, reverse=True)
