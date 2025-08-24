"""
Visualization tools for GNN clustering results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from ..models.graph_models import GraphSnapshot, ClusterResult


class ClusterVisualizer:
    """
    Comprehensive visualization suite for GNN clustering results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = px.colors.qualitative.Set3
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_graph_network(
        self,
        graph: GraphSnapshot,
        cluster_result: Optional[ClusterResult] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize graph network with clusters highlighted.
        
        Args:
            graph: Graph snapshot to visualize
            cluster_result: Optional cluster results for coloring
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Convert to NetworkX graph
        G = graph.get_networkx_graph()
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge weight if available
            if 'weight' in G[edge[0]][edge[1]]:
                edge_weights.append(G[edge[0]][edge[1]]['weight'])
            else:
                edge_weights.append(1.0)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node coordinates and attributes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            symbol = graph.symbols[node]
            volatility = graph.volatility_vector[node] if node < len(graph.volatility_vector) else 0
            
            node_text.append(f"Symbol: {symbol}<br>Volatility: {volatility:.4f}")
            node_sizes.append(max(10, volatility * 1000))  # Scale by volatility
            
            # Color by cluster if available
            if cluster_result is not None and node < len(cluster_result.cluster_labels):
                cluster_id = cluster_result.cluster_labels[node]
                if cluster_id >= 0:
                    color_idx = cluster_id % len(self.color_palette)
                    node_colors.append(self.color_palette[color_idx])
                else:
                    node_colors.append('gray')  # Noise points
            else:
                node_colors.append('blue')
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[graph.symbols[i] for i in range(len(graph.symbols))],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black'),
                opacity=0.8
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Crypto Market Graph - {graph.timestamp.strftime("%Y-%m-%d %H:%M")}',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Node size ~ volatility, Color ~ cluster",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=600
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_cluster_evolution(
        self,
        cluster_results: List[ClusterResult],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot evolution of clusters over time.
        
        Args:
            cluster_results: List of cluster results over time
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Extract time series data
        timestamps = [cr.timestamp for cr in cluster_results]
        num_clusters = [cr.num_clusters for cr in cluster_results]
        silhouette_scores = [cr.silhouette_score for cr in cluster_results]
        stability_scores = [cr.stability_score for cr in cluster_results]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Number of Clusters',
                'Silhouette Score',
                'Temporal Stability'
            ),
            vertical_spacing=0.1
        )
        
        # Number of clusters
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=num_clusters,
                mode='lines+markers',
                name='Clusters',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Silhouette score
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Stability score
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=stability_scores,
                mode='lines+markers',
                name='Stability',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Cluster Evolution Over Time',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_embedding_space(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        symbols: List[str],
        method: str = "tsne",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize embeddings in 2D space.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            cluster_labels: Cluster assignments
            symbols: Symbol names
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Reduce dimensionality
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': cluster_labels,
            'symbol': symbols
        })
        
        # Create scatter plot
        fig = px.scatter(
            df, x='x', y='y',
            color='cluster',
            hover_data=['symbol'],
            title=f'Embedding Space ({method.upper()})',
            color_continuous_scale='viridis'
        )
        
        # Add symbol annotations
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['x'], y=row['y'],
                text=row['symbol'],
                showarrow=False,
                font=dict(size=8)
            )
        
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        symbols: List[str],
        cluster_labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot correlation matrix heatmap with cluster grouping.
        
        Args:
            correlation_matrix: Correlation matrix
            symbols: Symbol names
            cluster_labels: Optional cluster labels for grouping
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Reorder by clusters if available
        if cluster_labels is not None:
            # Sort by cluster labels
            sort_indices = np.argsort(cluster_labels)
            correlation_matrix = correlation_matrix[sort_indices][:, sort_indices]
            symbols = [symbols[i] for i in sort_indices]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            width=800,
            height=600,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_arbitrage_opportunities(
        self,
        opportunities: List[Dict],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize arbitrage opportunities over time.
        
        Args:
            opportunities: List of arbitrage opportunities
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if not opportunities:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No arbitrage opportunities found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(opportunities)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Expected Returns Distribution',
                'Opportunities Over Time',
                'Confidence vs Expected Return',
                'Opportunity Types'
            )
        )
        
        # Expected returns histogram
        fig.add_trace(
            go.Histogram(
                x=df['expected_return'],
                nbinsx=20,
                name='Returns'
            ),
            row=1, col=1
        )
        
        # Opportunities over time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_counts = df.groupby(df['timestamp'].dt.date).size()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    mode='lines+markers',
                    name='Daily Count'
                ),
                row=1, col=2
            )
        
        # Scatter: Confidence vs Expected Return
        fig.add_trace(
            go.Scatter(
                x=df['confidence'],
                y=df['expected_return'],
                mode='markers',
                text=df.apply(lambda x: f"{x['symbol1']}-{x['symbol2']}", axis=1),
                name='Opportunities',
                marker=dict(
                    size=8,
                    color=df['expected_return'],
                    colorscale='viridis',
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # Opportunity types pie chart
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    name="Types"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Arbitrage Opportunities Analysis',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_cluster_composition(
        self,
        cluster_results: List[ClusterResult],
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot cluster composition and symbol co-occurrence.
        
        Args:
            cluster_results: List of cluster results
            top_n: Number of top symbol pairs to show
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Count symbol co-occurrences
        co_occurrence = {}
        
        for cr in cluster_results:
            for cluster_symbols in cr.cluster_symbols:
                if len(cluster_symbols) > 1:
                    for i, sym1 in enumerate(cluster_symbols):
                        for sym2 in cluster_symbols[i+1:]:
                            pair = tuple(sorted([sym1, sym2]))
                            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
        
        # Get top pairs
        top_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not top_pairs:
            fig = go.Figure()
            fig.add_annotation(
                text="No cluster pairs found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Create bar chart
        pair_labels = [f"{pair[0]}-{pair[1]}" for pair, count in top_pairs]
        pair_counts = [count for pair, count in top_pairs]
        
        fig = go.Figure(data=[
            go.Bar(
                x=pair_labels,
                y=pair_counts,
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title='Top Symbol Co-occurrences in Clusters',
            xaxis_title='Symbol Pairs',
            yaxis_title='Co-occurrence Count',
            xaxis_tickangle=45,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard(
        self,
        graph_sequence: List[GraphSnapshot],
        cluster_results: List[ClusterResult],
        arbitrage_opportunities: List[Dict],
        save_path: str = "cluster_dashboard.html"
    ):
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            graph_sequence: List of graph snapshots
            cluster_results: List of cluster results
            arbitrage_opportunities: List of arbitrage opportunities
            save_path: Path to save the dashboard
        """
        # Create individual plots
        plots = {}
        
        if graph_sequence and cluster_results:
            # Network plot for latest graph
            latest_graph = graph_sequence[-1]
            latest_clusters = cluster_results[-1]
            plots['network'] = self.plot_graph_network(latest_graph, latest_clusters)
        
        if cluster_results:
            # Cluster evolution
            plots['evolution'] = self.plot_cluster_evolution(cluster_results)
            
            # Embedding space for latest result
            if cluster_results[-1].cluster_centers.shape[0] > 1:
                plots['embeddings'] = self.plot_embedding_space(
                    cluster_results[-1].cluster_centers,
                    cluster_results[-1].cluster_labels,
                    graph_sequence[-1].symbols if graph_sequence else [f"Asset_{i}" for i in range(len(cluster_results[-1].cluster_labels))]
                )
            
            # Cluster composition
            plots['composition'] = self.plot_cluster_composition(cluster_results)
        
        if arbitrage_opportunities:
            # Arbitrage analysis
            plots['arbitrage'] = self.plot_arbitrage_opportunities(arbitrage_opportunities)
        
        if graph_sequence:
            # Correlation heatmap for latest graph
            latest_graph = graph_sequence[-1]
            latest_clusters = cluster_results[-1] if cluster_results else None
            plots['correlation'] = self.plot_correlation_heatmap(
                latest_graph.correlation_matrix,
                latest_graph.symbols,
                latest_clusters.cluster_labels if latest_clusters else None
            )
        
        # Combine plots into dashboard (save individual HTML files)
        dashboard_dir = save_path.replace('.html', '_dashboard')
        os.makedirs(dashboard_dir, exist_ok=True)
        
        for plot_name, plot_fig in plots.items():
            plot_path = os.path.join(dashboard_dir, f"{plot_name}.html")
            plot_fig.write_html(plot_path)
        
        # Create index file
        index_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GNN Clustering Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot-container {{ margin: 20px 0; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>GNN Crypto Clustering Dashboard</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            {"".join([f'<div class="plot-container"><h2>{name.title()}</h2><iframe src="{name}.html"></iframe></div>' for name in plots.keys()])}
        </body>
        </html>
        """
        
        index_path = os.path.join(dashboard_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        print(f"Dashboard saved to {index_path}")


# Helper function for easy plotting
def quick_visualize_results(
    graph_sequence: List[GraphSnapshot],
    cluster_results: List[ClusterResult],
    arbitrage_opportunities: List[Dict] = None,
    save_dir: str = "visualization_results"
):
    """
    Quick visualization of clustering results.
    
    Args:
        graph_sequence: List of graph snapshots
        cluster_results: List of cluster results
        arbitrage_opportunities: Optional list of arbitrage opportunities
        save_dir: Directory to save results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = ClusterVisualizer()
    
    if arbitrage_opportunities is None:
        arbitrage_opportunities = []
    
    # Create dashboard
    dashboard_path = os.path.join(save_dir, "dashboard.html")
    visualizer.create_dashboard(
        graph_sequence,
        cluster_results,
        arbitrage_opportunities,
        dashboard_path
    )
    
    print(f"Visualizations saved to {save_dir}")
    return dashboard_path
