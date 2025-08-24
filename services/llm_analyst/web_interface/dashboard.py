"""
Web dashboard for LLM Analyst service.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from ..models.anomaly_models import AnomalyType, AnomalySeverity
from ..main import LLMAnalystService
from ...common.db.client import DatabaseClient
from ...common.logger import get_logger


class LLMAnalystDashboard:
    """Interactive dashboard for LLM Analyst service."""
    
    def __init__(self, service: LLMAnalystService):
        self.service = service
        self.logger = get_logger("llm_dashboard")
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="LLM Analyst Dashboard"
        )
        
        # Setup layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _create_layout(self):
        """Create the dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🧠 LLM Analyst Dashboard", className="text-center mb-4"),
                    html.P("Automated anomaly detection and diagnosis for HFT systems", 
                          className="text-center text-muted")
                ])
            ]),
            
            # Status Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Service Status", className="card-title"),
                            html.H2(id="service-status", className="text-success"),
                            html.P(id="uptime-info", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Anomalies", className="card-title"),
                            html.H2(id="active-anomalies", className="text-warning"),
                            html.P(id="anomaly-info", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Diagnoses", className="card-title"),
                            html.H2(id="total-diagnoses", className="text-info"),
                            html.P(id="diagnosis-info", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("LLM Cost", className="card-title"),
                            html.H2(id="llm-cost", className="text-primary"),
                            html.P(id="cost-info", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main Content Tabs
            dbc.Tabs([
                dbc.Tab(label="Real-time Monitoring", tab_id="monitoring"),
                dbc.Tab(label="Anomaly History", tab_id="history"),
                dbc.Tab(label="On-Demand Analysis", tab_id="analysis"),
                dbc.Tab(label="Performance Metrics", tab_id="metrics")
            ], id="main-tabs", active_tab="monitoring"),
            
            # Tab Content
            html.Div(id="tab-content", className="mt-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('service-status', 'children'),
             Output('uptime-info', 'children'),
             Output('active-anomalies', 'children'),
             Output('anomaly-info', 'children'),
             Output('total-diagnoses', 'children'),
             Output('diagnosis-info', 'children'),
             Output('llm-cost', 'children'),
             Output('cost-info', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_status_cards(n):
            """Update status cards."""
            try:
                status = self.service.get_service_status()
                
                # Service status
                service_status = "🟢 Running" if status['is_running'] else "🔴 Stopped"
                uptime_info = f"Detection interval: {status['detection_interval']}s"
                
                # Active anomalies
                active_count = status['current_diagnoses']
                anomaly_info = f"Currently processing {active_count} anomalies"
                
                # Total diagnoses
                total_diagnoses = status['performance_stats']['total_diagnoses']
                success_rate = 0
                if total_diagnoses > 0:
                    success_rate = status['performance_stats']['successful_diagnoses'] / total_diagnoses
                diagnosis_info = f"Success rate: {success_rate:.1%}"
                
                # LLM cost
                llm_perf = status['llm_performance']
                cost = f"${llm_perf['total_cost_usd']:.2f}"
                cost_info = f"Avg per analysis: ${llm_perf['avg_cost_per_analysis']:.3f}"
                
                return (service_status, uptime_info, str(active_count), anomaly_info,
                       str(total_diagnoses), diagnosis_info, cost, cost_info)
                
            except Exception as e:
                self.logger.error(f"Error updating status cards: {e}")
                return "Error", "Error", "Error", "Error", "Error", "Error", "Error", "Error"
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab')]
        )
        def render_tab_content(active_tab):
            """Render content for active tab."""
            if active_tab == "monitoring":
                return self._create_monitoring_tab()
            elif active_tab == "history":
                return self._create_history_tab()
            elif active_tab == "analysis":
                return self._create_analysis_tab()
            elif active_tab == "metrics":
                return self._create_metrics_tab()
            else:
                return html.Div("Tab not found")
    
    def _create_monitoring_tab(self):
        """Create real-time monitoring tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Recent Anomalies"),
                    html.Div(id="recent-anomalies-list")
                ], width=6),
                dbc.Col([
                    html.H3("System Health"),
                    dcc.Graph(id="system-health-chart")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Detection Timeline"),
                    dcc.Graph(id="detection-timeline")
                ], width=12)
            ], className="mt-4")
        ])
    
    def _create_history_tab(self):
        """Create anomaly history tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Diagnosis History"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id="diagnosis-history-table")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    html.H3("Anomaly Statistics"),
                    dcc.Graph(id="anomaly-stats-chart")
                ], width=4)
            ])
        ])
    
    def _create_analysis_tab(self):
        """Create on-demand analysis tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("On-Demand Analysis"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Anomaly Type"),
                                    dcc.Dropdown(
                                        id="anomaly-type-dropdown",
                                        options=[
                                            {"label": "Performance Drop", "value": "performance_drop"},
                                            {"label": "Sudden Loss", "value": "sudden_loss"},
                                            {"label": "Volume Spike", "value": "volume_spike"},
                                            {"label": "Correlation Break", "value": "correlation_break"},
                                            {"label": "System Error", "value": "system_error"},
                                            {"label": "Funding Rate Anomaly", "value": "funding_rate_anomaly"}
                                        ],
                                        value="performance_drop"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Time Range"),
                                    dcc.Dropdown(
                                        id="time-range-dropdown",
                                        options=[
                                            {"label": "Last 1 Hour", "value": 1},
                                            {"label": "Last 6 Hours", "value": 6},
                                            {"label": "Last 24 Hours", "value": 24},
                                            {"label": "Last 7 Days", "value": 168}
                                        ],
                                        value=24
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Run Analysis",
                                        id="run-analysis-btn",
                                        color="primary",
                                        className="mb-3"
                                    )
                                ])
                            ]),
                            html.Div(id="analysis-results")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def _create_metrics_tab(self):
        """Create performance metrics tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("LLM Performance"),
                    dcc.Graph(id="llm-performance-chart")
                ], width=6),
                dbc.Col([
                    html.H3("Detection Accuracy"),
                    dcc.Graph(id="detection-accuracy-chart")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Response Times"),
                    dcc.Graph(id="response-times-chart")
                ], width=12)
            ], className="mt-4")
        ])
    
    def create_recent_anomalies_list(self) -> html.Div:
        """Create list of recent anomalies."""
        try:
            recent_diagnoses = self.service.get_recent_diagnoses(limit=5)
            
            if not recent_diagnoses:
                return html.P("No recent anomalies detected", className="text-muted")
            
            anomaly_cards = []
            for diagnosis in recent_diagnoses:
                # Determine severity color
                severity_colors = {
                    "low": "success",
                    "medium": "warning", 
                    "high": "danger",
                    "critical": "danger"
                }
                
                # Get severity from diagnosis (would need to track original anomaly)
                severity_color = "info"  # Default
                
                card = dbc.Card([
                    dbc.CardBody([
                        html.H6(diagnosis.primary_hypothesis.title, className="card-title"),
                        html.P(f"Confidence: {diagnosis.primary_hypothesis.confidence:.2f}", 
                              className="text-muted small"),
                        html.P(diagnosis.executive_summary[:100] + "...", 
                              className="card-text small"),
                        html.Small(f"Analyzed: {diagnosis.timestamp.strftime('%H:%M:%S')}", 
                                 className="text-muted")
                    ])
                ], color=severity_color, outline=True, className="mb-2")
                
                anomaly_cards.append(card)
            
            return html.Div(anomaly_cards)
            
        except Exception as e:
            self.logger.error(f"Error creating anomalies list: {e}")
            return html.P("Error loading anomalies", className="text-danger")
    
    def create_system_health_chart(self) -> go.Figure:
        """Create system health chart."""
        try:
            # Get system status
            status = self.service.get_service_status()
            
            # Create gauge chart for overall health
            health_score = 85  # Would calculate based on various metrics
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "System Health Score"},
                delta = {'reference': 90},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating health chart: {e}")
            return go.Figure()
    
    def create_detection_timeline(self) -> go.Figure:
        """Create detection timeline chart."""
        try:
            # Get detection statistics
            detection_stats = self.service.anomaly_detector.get_detection_statistics()
            
            # Create timeline of detections (mock data for now)
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                                 end=datetime.now(), freq='H')
            
            # Simulate detection counts
            import numpy as np
            detection_counts = np.random.poisson(2, len(dates))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=detection_counts,
                mode='lines+markers',
                name='Anomalies Detected',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Anomaly Detection Timeline",
                xaxis_title="Time",
                yaxis_title="Anomalies Detected",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating timeline: {e}")
            return go.Figure()
    
    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
        """Run the dashboard server."""
        self.logger.info(f"Starting LLM Analyst Dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# Standalone dashboard runner
async def create_dashboard():
    """Create dashboard with service instance."""
    try:
        # Initialize database client
        db_client = DatabaseClient()
        await db_client.initialize_db()
        
        # Create service instance
        service = LLMAnalystService(
            db_client=db_client,
            detection_interval=300
        )
        
        # Create dashboard
        dashboard = LLMAnalystDashboard(service)
        
        return dashboard
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        raise


def main():
    """Main entry point for dashboard."""
    import os
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some features may not work.")
    
    try:
        # Create and run dashboard
        dashboard = asyncio.run(create_dashboard())
        dashboard.run(debug=True)
        
    except Exception as e:
        print(f"Error running dashboard: {e}")


if __name__ == "__main__":
    main()
