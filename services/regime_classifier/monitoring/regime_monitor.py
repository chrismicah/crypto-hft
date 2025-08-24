"""
Monitoring and visualization for market regime classification.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..models.regime_models import (
    MarketRegime, RegimeClassification, RegimeTransition, 
    RegimeAlert, RegimePerformanceMetrics
)
from ...common.logger import get_logger


class RegimeMonitor:
    """
    Monitor and visualize market regime classifications and transitions.
    """
    
    def __init__(self):
        self.logger = get_logger("regime_monitor")
        
        # Color scheme for regimes
        self.regime_colors = {
            MarketRegime.LOW_VOL_BULL: '#2E8B57',      # Sea Green
            MarketRegime.LOW_VOL_BEAR: '#DC143C',      # Crimson
            MarketRegime.LOW_VOL_RANGE: '#4682B4',     # Steel Blue
            MarketRegime.HIGH_VOL_BULL: '#32CD32',     # Lime Green
            MarketRegime.HIGH_VOL_BEAR: '#FF4500',     # Orange Red
            MarketRegime.HIGH_VOL_RANGE: '#9370DB',    # Medium Purple
            MarketRegime.STABLE_RANGE: '#20B2AA',      # Light Sea Green
            MarketRegime.TRENDING_UP: '#228B22',       # Forest Green
            MarketRegime.TRENDING_DOWN: '#B22222',     # Fire Brick
            MarketRegime.CRISIS: '#8B0000',            # Dark Red
            MarketRegime.RECOVERY: '#FF8C00',          # Dark Orange
            MarketRegime.UNKNOWN: '#808080'            # Gray
        }
    
    def create_regime_timeline(self, 
                             classifications: List[RegimeClassification],
                             price_data: Optional[pd.DataFrame] = None,
                             title: str = "Market Regime Timeline") -> go.Figure:
        """
        Create a timeline visualization of regime classifications.
        
        Args:
            classifications: List of regime classifications
            price_data: Optional price data to overlay
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not classifications:
                return go.Figure().add_annotation(text="No classification data available")
            
            # Prepare data
            df = pd.DataFrame([
                {
                    'timestamp': c.timestamp,
                    'regime': c.regime.value,
                    'confidence': c.confidence,
                    'regime_enum': c.regime
                }
                for c in classifications
            ])
            
            # Create subplot structure
            if price_data is not None:
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Price & Regime', 'Regime Confidence', 'Regime Distribution'),
                    vertical_spacing=0.08,
                    row_heights=[0.5, 0.25, 0.25]
                )
            else:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Regime Timeline', 'Regime Confidence'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
            
            # Add price data if available
            if price_data is not None and 'close' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='black', width=1),
                        yaxis='y2'
                    ),
                    row=1, col=1
                )
            
            # Add regime timeline
            for regime in MarketRegime:
                regime_data = df[df['regime_enum'] == regime]
                if not regime_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_data['timestamp'],
                            y=[regime.value] * len(regime_data),
                            mode='markers',
                            name=regime.value.replace('_', ' ').title(),
                            marker=dict(
                                color=self.regime_colors[regime],
                                size=8,
                                opacity=0.8
                            ),
                            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Confidence: %{customdata:.3f}<extra></extra>',
                            customdata=regime_data['confidence']
                        ),
                        row=1, col=1
                    )
            
            # Add confidence timeline
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['confidence'],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=2 if price_data is None else 2, col=1
            )
            
            # Add confidence threshold line
            fig.add_hline(
                y=0.7, line_dash="dash", line_color="red",
                annotation_text="High Confidence",
                row=2 if price_data is None else 2, col=1
            )
            
            # Add regime distribution if we have price data
            if price_data is not None:
                regime_counts = df['regime'].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=regime_counts.index,
                        y=regime_counts.values,
                        name='Regime Count',
                        marker_color=[self.regime_colors.get(MarketRegime(r), '#808080') for r in regime_counts.index],
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800 if price_data is not None else 600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Regime", row=1, col=1)
            if price_data is not None:
                fig.update_yaxes(title_text="Price", secondary_y=True, row=1, col=1)
            
            fig.update_yaxes(title_text="Confidence", range=[0, 1], row=2 if price_data is None else 2, col=1)
            
            if price_data is not None:
                fig.update_yaxes(title_text="Count", row=3, col=1)
                fig.update_xaxes(title_text="Regime", row=3, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating regime timeline: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def create_regime_transition_matrix(self, 
                                      transitions: List[RegimeTransition],
                                      title: str = "Regime Transition Matrix") -> go.Figure:
        """
        Create a heatmap of regime transitions.
        
        Args:
            transitions: List of regime transitions
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not transitions:
                return go.Figure().add_annotation(text="No transition data available")
            
            # Create transition matrix
            regimes = list(MarketRegime)
            regime_names = [r.value.replace('_', ' ').title() for r in regimes]
            
            # Initialize matrix
            matrix = np.zeros((len(regimes), len(regimes)))
            
            # Count transitions
            for transition in transitions:
                from_idx = regimes.index(transition.from_regime)
                to_idx = regimes.index(transition.to_regime)
                matrix[from_idx, to_idx] += 1
            
            # Normalize by row (from regime)
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=normalized_matrix,
                x=regime_names,
                y=regime_names,
                colorscale='Blues',
                text=matrix.astype(int),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='From: %{y}<br>To: %{x}<br>Count: %{text}<br>Probability: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="To Regime",
                yaxis_title="From Regime",
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating transition matrix: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def create_regime_performance_chart(self, 
                                      performance_data: Dict[MarketRegime, Dict[str, float]],
                                      title: str = "Regime Performance Metrics") -> go.Figure:
        """
        Create a performance comparison chart across regimes.
        
        Args:
            performance_data: Performance metrics by regime
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not performance_data:
                return go.Figure().add_annotation(text="No performance data available")
            
            # Prepare data
            regimes = []
            sharpe_ratios = []
            returns = []
            max_drawdowns = []
            
            for regime, metrics in performance_data.items():
                regimes.append(regime.value.replace('_', ' ').title())
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0.0))
                returns.append(metrics.get('avg_return', 0.0) * 100)  # Convert to percentage
                max_drawdowns.append(metrics.get('max_drawdown', 0.0) * 100)  # Convert to percentage
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sharpe Ratio', 'Average Return (%)', 'Max Drawdown (%)', 'Risk-Return Scatter'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Sharpe ratio bar chart
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=sharpe_ratios,
                    name='Sharpe Ratio',
                    marker_color=[self.regime_colors.get(MarketRegime(r.lower().replace(' ', '_')), '#808080') for r in regimes],
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Returns bar chart
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=returns,
                    name='Avg Return (%)',
                    marker_color=[self.regime_colors.get(MarketRegime(r.lower().replace(' ', '_')), '#808080') for r in regimes],
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Max drawdown bar chart
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=max_drawdowns,
                    name='Max Drawdown (%)',
                    marker_color=[self.regime_colors.get(MarketRegime(r.lower().replace(' ', '_')), '#808080') for r in regimes],
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Risk-return scatter
            fig.add_trace(
                go.Scatter(
                    x=max_drawdowns,
                    y=returns,
                    mode='markers+text',
                    text=regimes,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=[self.regime_colors.get(MarketRegime(r.lower().replace(' ', '_')), '#808080') for r in regimes]
                    ),
                    name='Risk-Return',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(tickangle=45, row=1, col=2)
            fig.update_xaxes(tickangle=45, row=2, col=1)
            fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=2)
            fig.update_yaxes(title_text="Average Return (%)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance chart: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def create_confidence_distribution(self, 
                                     classifications: List[RegimeClassification],
                                     title: str = "Classification Confidence Distribution") -> go.Figure:
        """
        Create a distribution chart of classification confidence levels.
        
        Args:
            classifications: List of regime classifications
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not classifications:
                return go.Figure().add_annotation(text="No classification data available")
            
            # Extract confidence data by regime
            regime_confidences = {}
            for classification in classifications:
                regime = classification.regime
                if regime not in regime_confidences:
                    regime_confidences[regime] = []
                regime_confidences[regime].append(classification.confidence)
            
            # Create box plots
            fig = go.Figure()
            
            for regime, confidences in regime_confidences.items():
                fig.add_trace(go.Box(
                    y=confidences,
                    name=regime.value.replace('_', ' ').title(),
                    marker_color=self.regime_colors[regime],
                    boxpoints='outliers'
                ))
            
            # Add overall distribution
            all_confidences = [c.confidence for c in classifications]
            fig.add_trace(go.Histogram(
                x=all_confidences,
                name='Overall Distribution',
                opacity=0.7,
                nbinsx=20,
                yaxis='y2',
                marker_color='lightblue'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                yaxis=dict(title="Confidence", domain=[0, 0.7]),
                yaxis2=dict(title="Frequency", domain=[0.75, 1], overlaying='y'),
                xaxis_title="Regime",
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating confidence distribution: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def create_alert_timeline(self, 
                            alerts: List[RegimeAlert],
                            title: str = "Regime Alerts Timeline") -> go.Figure:
        """
        Create a timeline of regime alerts.
        
        Args:
            alerts: List of regime alerts
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not alerts:
                return go.Figure().add_annotation(text="No alert data available")
            
            # Prepare data
            df = pd.DataFrame([
                {
                    'timestamp': alert.timestamp,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'regime': alert.regime.value,
                    'message': alert.message,
                    'risk_level': alert.risk_level
                }
                for alert in alerts
            ])
            
            # Define severity colors
            severity_colors = {
                'low': '#90EE90',      # Light Green
                'medium': '#FFD700',   # Gold
                'high': '#FF6347',     # Tomato
                'critical': '#DC143C'  # Crimson
            }
            
            # Create scatter plot
            fig = go.Figure()
            
            for severity in df['severity'].unique():
                severity_data = df[df['severity'] == severity]
                fig.add_trace(go.Scatter(
                    x=severity_data['timestamp'],
                    y=severity_data['alert_type'],
                    mode='markers',
                    name=severity.title(),
                    marker=dict(
                        color=severity_colors.get(severity, '#808080'),
                        size=12,
                        symbol='diamond' if severity == 'critical' else 'circle'
                    ),
                    hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Severity: ' + severity + '<br>Regime: %{customdata}<br><extra></extra>',
                    customdata=severity_data['regime']
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Alert Type",
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating alert timeline: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def create_regime_duration_analysis(self, 
                                      transitions: List[RegimeTransition],
                                      title: str = "Regime Duration Analysis") -> go.Figure:
        """
        Create an analysis of regime durations.
        
        Args:
            transitions: List of regime transitions
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not transitions:
                return go.Figure().add_annotation(text="No transition data available")
            
            # Calculate durations by regime
            regime_durations = {}
            for transition in transitions:
                regime = transition.from_regime
                duration_hours = transition.duration_in_previous.total_seconds() / 3600
                
                if regime not in regime_durations:
                    regime_durations[regime] = []
                regime_durations[regime].append(duration_hours)
            
            # Create box plots
            fig = go.Figure()
            
            for regime, durations in regime_durations.items():
                fig.add_trace(go.Box(
                    y=durations,
                    name=regime.value.replace('_', ' ').title(),
                    marker_color=self.regime_colors[regime],
                    boxpoints='outliers'
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Regime",
                yaxis_title="Duration (Hours)",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating duration analysis: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def generate_regime_report(self, 
                             classifications: List[RegimeClassification],
                             transitions: List[RegimeTransition],
                             alerts: List[RegimeAlert],
                             performance_data: Optional[Dict[MarketRegime, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive regime analysis report.
        
        Args:
            classifications: List of regime classifications
            transitions: List of regime transitions
            alerts: List of regime alerts
            performance_data: Optional performance data by regime
            
        Returns:
            Comprehensive report dictionary
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {},
                'regime_statistics': {},
                'transition_analysis': {},
                'alert_analysis': {},
                'performance_analysis': {}
            }
            
            # Summary statistics
            if classifications:
                total_classifications = len(classifications)
                unique_regimes = len(set(c.regime for c in classifications))
                avg_confidence = np.mean([c.confidence for c in classifications])
                
                # Current regime info
                current = classifications[-1]
                
                report['summary'] = {
                    'total_classifications': total_classifications,
                    'unique_regimes_observed': unique_regimes,
                    'average_confidence': float(avg_confidence),
                    'current_regime': current.regime.value,
                    'current_confidence': float(current.confidence),
                    'analysis_period_hours': (classifications[-1].timestamp - classifications[0].timestamp).total_seconds() / 3600
                }
            
            # Regime statistics
            if classifications:
                regime_counts = {}
                regime_confidences = {}
                
                for classification in classifications:
                    regime = classification.regime
                    
                    # Count occurrences
                    regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
                    
                    # Collect confidences
                    if regime.value not in regime_confidences:
                        regime_confidences[regime.value] = []
                    regime_confidences[regime.value].append(classification.confidence)
                
                # Calculate statistics
                regime_stats = {}
                for regime, count in regime_counts.items():
                    confidences = regime_confidences[regime]
                    regime_stats[regime] = {
                        'occurrences': count,
                        'percentage': (count / len(classifications)) * 100,
                        'avg_confidence': float(np.mean(confidences)),
                        'min_confidence': float(np.min(confidences)),
                        'max_confidence': float(np.max(confidences)),
                        'std_confidence': float(np.std(confidences))
                    }
                
                report['regime_statistics'] = regime_stats
            
            # Transition analysis
            if transitions:
                total_transitions = len(transitions)
                transition_counts = {}
                avg_duration_by_regime = {}
                
                for transition in transitions:
                    # Count transitions
                    key = f"{transition.from_regime.value} -> {transition.to_regime.value}"
                    transition_counts[key] = transition_counts.get(key, 0) + 1
                    
                    # Duration analysis
                    regime = transition.from_regime.value
                    duration_hours = transition.duration_in_previous.total_seconds() / 3600
                    
                    if regime not in avg_duration_by_regime:
                        avg_duration_by_regime[regime] = []
                    avg_duration_by_regime[regime].append(duration_hours)
                
                # Calculate average durations
                for regime in avg_duration_by_regime:
                    durations = avg_duration_by_regime[regime]
                    avg_duration_by_regime[regime] = {
                        'avg_hours': float(np.mean(durations)),
                        'median_hours': float(np.median(durations)),
                        'std_hours': float(np.std(durations)),
                        'min_hours': float(np.min(durations)),
                        'max_hours': float(np.max(durations))
                    }
                
                report['transition_analysis'] = {
                    'total_transitions': total_transitions,
                    'transition_counts': transition_counts,
                    'average_durations': avg_duration_by_regime,
                    'transitions_per_day': total_transitions / (max(1, len(classifications) / 24)) if classifications else 0
                }
            
            # Alert analysis
            if alerts:
                alert_counts_by_type = {}
                alert_counts_by_severity = {}
                
                for alert in alerts:
                    # By type
                    alert_counts_by_type[alert.alert_type] = alert_counts_by_type.get(alert.alert_type, 0) + 1
                    
                    # By severity
                    alert_counts_by_severity[alert.severity] = alert_counts_by_severity.get(alert.severity, 0) + 1
                
                report['alert_analysis'] = {
                    'total_alerts': len(alerts),
                    'alerts_by_type': alert_counts_by_type,
                    'alerts_by_severity': alert_counts_by_severity,
                    'critical_alerts': alert_counts_by_severity.get('critical', 0),
                    'alert_rate_per_day': len(alerts) / (max(1, len(classifications) / 24)) if classifications else 0
                }
            
            # Performance analysis
            if performance_data:
                best_regime = None
                worst_regime = None
                best_sharpe = float('-inf')
                worst_sharpe = float('inf')
                
                for regime, metrics in performance_data.items():
                    sharpe = metrics.get('sharpe_ratio', 0.0)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_regime = regime.value
                    if sharpe < worst_sharpe:
                        worst_sharpe = sharpe
                        worst_regime = regime.value
                
                report['performance_analysis'] = {
                    'best_performing_regime': best_regime,
                    'best_sharpe_ratio': float(best_sharpe),
                    'worst_performing_regime': worst_regime,
                    'worst_sharpe_ratio': float(worst_sharpe),
                    'regime_performance_data': {k.value: v for k, v in performance_data.items()}
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating regime report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
