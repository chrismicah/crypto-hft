"""Streamlit dashboard for HFT system ad-hoc analysis and P&L visualization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Any

# Configure Streamlit page
st.set_page_config(
    page_title="HFT Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit-card {
        border-left-color: #2ca02c;
    }
    .loss-card {
        border-left-color: #d62728;
    }
    .warning-card {
        border-left-color: #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)


class PrometheusClient:
    """Client for querying Prometheus metrics."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """Initialize Prometheus client."""
        self.base_url = prometheus_url
        self.session = requests.Session()
    
    def query(self, query: str, time_param: Optional[str] = None) -> Dict[str, Any]:
        """Execute a Prometheus query."""
        try:
            params = {'query': query}
            if time_param:
                params['time'] = time_param
            
            response = self.session.get(
                f"{self.base_url}/api/v1/query",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error querying Prometheus: {str(e)}")
            return {"status": "error", "data": {"result": []}}
    
    def query_range(
        self,
        query: str,
        start: str,
        end: str,
        step: str = "15s"
    ) -> Dict[str, Any]:
        """Execute a Prometheus range query."""
        try:
            params = {
                'query': query,
                'start': start,
                'end': end,
                'step': step
            }
            
            response = self.session.get(
                f"{self.base_url}/api/v1/query_range",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error querying Prometheus range: {str(e)}")
            return {"status": "error", "data": {"result": []}}


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_prometheus_data(query: str, client: PrometheusClient) -> pd.DataFrame:
    """Get data from Prometheus and convert to DataFrame."""
    result = client.query(query)
    
    if result["status"] != "success" or not result["data"]["result"]:
        return pd.DataFrame()
    
    data = []
    for item in result["data"]["result"]:
        row = item["metric"].copy()
        row["value"] = float(item["value"][1])
        row["timestamp"] = datetime.fromtimestamp(float(item["value"][0]))
        data.append(row)
    
    return pd.DataFrame(data)


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_time_series_data(
    query: str,
    hours_back: int,
    client: PrometheusClient
) -> pd.DataFrame:
    """Get time series data from Prometheus."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)
    
    result = client.query_range(
        query=query,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        step="1m"
    )
    
    if result["status"] != "success" or not result["data"]["result"]:
        return pd.DataFrame()
    
    all_data = []
    for series in result["data"]["result"]:
        metric_labels = series["metric"]
        for timestamp, value in series["values"]:
            row = metric_labels.copy()
            row["timestamp"] = datetime.fromtimestamp(float(timestamp))
            row["value"] = float(value)
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values("timestamp")
    
    return df


def create_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Create P&L time series chart."""
    if df.empty:
        return go.Figure().add_annotation(
            text="No P&L data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Portfolio Value", "P&L Components"),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Portfolio value
    portfolio_df = df[df.get('__name__') == 'portfolio_value_total']
    if not portfolio_df.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df['timestamp'],
                y=portfolio_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Realized P&L
    realized_df = df[df.get('__name__') == 'realized_pnl_total']
    for symbol in realized_df.get('symbol', pd.Series()).unique():
        symbol_data = realized_df[realized_df['symbol'] == symbol]
        fig.add_trace(
            go.Scatter(
                x=symbol_data['timestamp'],
                y=symbol_data['value'],
                mode='lines',
                name=f'Realized P&L - {symbol}',
                line=dict(width=1)
            ),
            row=2, col=1
        )
    
    # Unrealized P&L
    unrealized_df = df[df.get('__name__') == 'unrealized_pnl_total']
    for symbol in unrealized_df.get('symbol', pd.Series()).unique():
        symbol_data = unrealized_df[unrealized_df['symbol'] == symbol]
        fig.add_trace(
            go.Scatter(
                x=symbol_data['timestamp'],
                y=symbol_data['value'],
                mode='lines',
                name=f'Unrealized P&L - {symbol}',
                line=dict(dash='dash', width=1)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="Portfolio Performance",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    
    return fig


def create_position_chart(positions_df: pd.DataFrame) -> go.Figure:
    """Create positions pie chart."""
    if positions_df.empty:
        return go.Figure().add_annotation(
            text="No position data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Group by symbol and sum position sizes
    position_summary = positions_df.groupby(['symbol', 'side'])['value'].sum().reset_index()
    position_summary['label'] = position_summary['symbol'] + ' (' + position_summary['side'] + ')'
    
    fig = go.Figure(data=[
        go.Pie(
            labels=position_summary['label'],
            values=position_summary['value'],
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Current Positions",
        height=400,
        showlegend=True
    )
    
    return fig


def create_risk_metrics_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Create risk metrics chart."""
    if risk_df.empty:
        return go.Figure().add_annotation(
            text="No risk data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Drawdown", "Changepoint Probability", "Risk State", "Position Concentration"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Drawdown
    drawdown_df = risk_df[risk_df.get('__name__') == 'current_drawdown_percent']
    if not drawdown_df.empty:
        fig.add_trace(
            go.Scatter(
                x=drawdown_df['timestamp'],
                y=drawdown_df['value'],
                mode='lines',
                name='Current Drawdown',
                line=dict(color='red')
            ),
            row=1, col=1
        )
    
    # Changepoint probability
    cp_df = risk_df[risk_df.get('__name__') == 'changepoint_probability']
    for symbol in cp_df.get('symbol', pd.Series()).unique():
        symbol_data = cp_df[cp_df['symbol'] == symbol]
        fig.add_trace(
            go.Scatter(
                x=symbol_data['timestamp'],
                y=symbol_data['value'],
                mode='lines',
                name=f'CP Prob - {symbol}'
            ),
            row=1, col=2
        )
    
    # Risk state
    risk_state_df = risk_df[risk_df.get('__name__') == 'risk_state']
    if not risk_state_df.empty:
        fig.add_trace(
            go.Scatter(
                x=risk_state_df['timestamp'],
                y=risk_state_df['value'],
                mode='lines+markers',
                name='Risk State',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
    
    # Position concentration
    conc_df = risk_df[risk_df.get('__name__') == 'concentration_risk_percent']
    for symbol in conc_df.get('symbol', pd.Series()).unique():
        symbol_data = conc_df[conc_df['symbol'] == symbol]
        fig.add_trace(
            go.Scatter(
                x=symbol_data['timestamp'],
                y=symbol_data['value'],
                mode='lines',
                name=f'Concentration - {symbol}'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Risk Metrics",
        height=600,
        showlegend=True
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.title("🚀 HFT Analytics Dashboard")
    st.markdown("Real-time analysis and monitoring for the HFT crypto trading system")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    prometheus_url = st.sidebar.text_input(
        "Prometheus URL",
        value="http://localhost:9090",
        help="URL of the Prometheus server"
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        options=[1, 4, 12, 24, 48],
        index=2,
        format_func=lambda x: f"{x} hours",
        help="Time range for historical data"
    )
    
    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh",
        value=True,
        help="Automatically refresh data every 30 seconds"
    )
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=30,
            step=10
        )
    
    # Initialize Prometheus client
    client = PrometheusClient(prometheus_url)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        portfolio_data = get_prometheus_data("portfolio_value_total", client)
        if not portfolio_data.empty:
            portfolio_value = portfolio_data['value'].iloc[0]
            st.metric(
                label="Portfolio Value",
                value=f"${portfolio_value:,.2f}",
                delta=None
            )
        else:
            st.metric("Portfolio Value", "N/A")
    
    with col2:
        pnl_data = get_prometheus_data("sum(realized_pnl_total)", client)
        if not pnl_data.empty:
            total_pnl = pnl_data['value'].iloc[0]
            st.metric(
                label="Total Realized P&L",
                value=f"${total_pnl:,.2f}",
                delta=None
            )
        else:
            st.metric("Total Realized P&L", "N/A")
    
    with col3:
        drawdown_data = get_prometheus_data("current_drawdown_percent", client)
        if not drawdown_data.empty:
            drawdown = drawdown_data['value'].iloc[0]
            st.metric(
                label="Current Drawdown",
                value=f"{drawdown:.2f}%",
                delta=None
            )
        else:
            st.metric("Current Drawdown", "N/A")
    
    with col4:
        positions_data = get_prometheus_data("count(position_size > 0)", client)
        if not positions_data.empty:
            active_positions = int(positions_data['value'].iloc[0])
            st.metric(
                label="Active Positions",
                value=str(active_positions),
                delta=None
            )
        else:
            st.metric("Active Positions", "N/A")
    
    # Charts
    st.header("📊 Performance Analysis")
    
    # P&L Chart
    pnl_queries = [
        "portfolio_value_total",
        "realized_pnl_total",
        "unrealized_pnl_total"
    ]
    
    pnl_data_frames = []
    for query in pnl_queries:
        df = get_time_series_data(query, time_range, client)
        if not df.empty:
            df['__name__'] = query
            pnl_data_frames.append(df)
    
    if pnl_data_frames:
        combined_pnl_df = pd.concat(pnl_data_frames, ignore_index=True)
        pnl_chart = create_pnl_chart(combined_pnl_df)
        st.plotly_chart(pnl_chart, use_container_width=True)
    else:
        st.warning("No P&L data available")
    
    # Positions and Risk
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Positions")
        positions_df = get_prometheus_data("position_size", client)
        if not positions_df.empty:
            position_chart = create_position_chart(positions_df)
            st.plotly_chart(position_chart, use_container_width=True)
        else:
            st.info("No position data available")
    
    with col2:
        st.subheader("Trading Activity")
        trades_df = get_time_series_data("rate(trades_total[5m]) * 300", time_range, client)
        if not trades_df.empty:
            fig = px.line(
                trades_df,
                x='timestamp',
                y='value',
                color='symbol' if 'symbol' in trades_df.columns else None,
                title="Trade Rate (5min average)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trading activity data available")
    
    # Risk Metrics
    st.header("⚠️ Risk Management")
    
    risk_queries = [
        "current_drawdown_percent",
        "changepoint_probability",
        "risk_state",
        "concentration_risk_percent"
    ]
    
    risk_data_frames = []
    for query in risk_queries:
        df = get_time_series_data(query, time_range, client)
        if not df.empty:
            df['__name__'] = query
            risk_data_frames.append(df)
    
    if risk_data_frames:
        combined_risk_df = pd.concat(risk_data_frames, ignore_index=True)
        risk_chart = create_risk_metrics_chart(combined_risk_df)
        st.plotly_chart(risk_chart, use_container_width=True)
    else:
        st.warning("No risk data available")
    
    # System Health
    st.header("🔧 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Service Status")
        service_status = get_prometheus_data("service_up", client)
        if not service_status.empty:
            for _, row in service_status.iterrows():
                service = row.get('service', 'Unknown')
                status = "🟢 UP" if row['value'] == 1 else "🔴 DOWN"
                st.write(f"**{service}**: {status}")
        else:
            st.info("No service status data available")
    
    with col2:
        st.subheader("Error Rates")
        error_data = get_time_series_data("rate(errors_total[5m])", 1, client)
        if not error_data.empty:
            fig = px.bar(
                error_data.groupby('service')['value'].sum().reset_index(),
                x='service',
                y='value',
                title="Error Rate by Service"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error data available")
    
    with col3:
        st.subheader("Latency")
        latency_data = get_time_series_data(
            "histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m])) * 1000",
            1,
            client
        )
        if not latency_data.empty:
            fig = px.bar(
                latency_data.groupby('service')['value'].mean().reset_index(),
                x='service',
                y='value',
                title="95th Percentile Latency (ms)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available")
    
    # Raw Data Explorer
    with st.expander("🔍 Raw Data Explorer"):
        st.subheader("Custom Prometheus Query")
        
        custom_query = st.text_input(
            "Enter Prometheus Query",
            value="up",
            help="Enter any valid Prometheus query"
        )
        
        if st.button("Execute Query"):
            if custom_query:
                result = client.query(custom_query)
                if result["status"] == "success" and result["data"]["result"]:
                    df = pd.DataFrame([
                        {**item["metric"], "value": float(item["value"][1])}
                        for item in result["data"]["result"]
                    ])
                    st.dataframe(df)
                else:
                    st.warning("No data returned from query")
            else:
                st.warning("Please enter a query")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
