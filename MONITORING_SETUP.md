# HFT Monitoring Stack Setup Guide

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
# Install monitoring dependencies
pip install -r requirements.txt
```

### 2. **Start the Monitoring Stack**
```bash
# Start all monitoring services
docker-compose up -d prometheus grafana alertmanager streamlit-app node-exporter cadvisor

# Check services are running
docker-compose ps
```

### 3. **Access Dashboards**
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093  
- **Streamlit**: http://localhost:8501

## 📊 Dashboard Configuration

### Grafana Setup
1. **Login**: Navigate to http://localhost:3000
   - Username: `admin`
   - Password: `admin123`

2. **Verify Datasource**: 
   - Go to Configuration → Data Sources
   - Prometheus should be automatically configured
   - Test connection should show "Data source is working"

3. **Import Dashboards**:
   - The HFT Trading Dashboard should be automatically loaded
   - Navigate to Dashboards → Browse → HFT Trading System

### Prometheus Verification
1. **Check Targets**: http://localhost:9090/targets
   - All HFT services should show as "UP"
   - If services show "DOWN", ensure they're running and exposing `/metrics`

2. **Test Queries**:
   ```promql
   # Check service health
   hft_service_up
   
   # Check request rates
   rate(hft_requests_total[5m])
   
   # Check portfolio value (when trading service is running)
   portfolio_value_total
   ```

## 🔧 Service Integration

### Add Metrics to Your Services

1. **Import the metrics framework**:
```python
from common.metrics import TradingMetrics, ServiceMetrics
from prometheus_client import start_http_server

# Initialize metrics
metrics = TradingMetrics("your-service-name")

# Start metrics server
start_http_server(8000)  # Expose on /metrics
```

2. **Update your services**:
```python
# Track requests
with metrics.track_request("GET", "/health"):
    # Your request handling code
    pass

# Record trading activity
metrics.record_trade("BTCUSDT", "BUY", 1.0, 50000.0, 25.0, 1000.0)

# Update portfolio metrics
metrics.update_pnl("BTCUSDT", realized_pnl=1000.0, unrealized_pnl=500.0)
```

3. **Update Docker Compose**:
```yaml
your-service:
  # ... existing config
  ports:
    - "8000:8000"  # Expose metrics port
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:8000/metrics || exit 1"]
```

## 🚨 Alerting Setup

### Configure Slack Notifications
1. **Create Slack Webhook**:
   - Go to your Slack workspace
   - Create an incoming webhook
   - Copy the webhook URL

2. **Update AlertManager**:
```bash
# Edit alertmanager config
vim monitoring/alertmanager/alertmanager.yml

# Replace YOUR_SLACK_WEBHOOK_URL_HERE with your actual webhook URL
```

3. **Restart AlertManager**:
```bash
docker-compose restart alertmanager
```

### Configure Email Alerts
1. **Update SMTP Settings** in `monitoring/alertmanager/alertmanager.yml`:
```yaml
global:
  smtp_smarthost: 'your-smtp-server:587'
  smtp_from: 'alerts@your-domain.com'
  smtp_auth_username: 'your-email@your-domain.com'
  smtp_auth_password: 'your-password'
```

2. **Update Email Recipients**:
```yaml
email_configs:
  - to: 'your-team@your-domain.com'
```

## 📈 Streamlit Analytics

### Custom Analysis
1. **Access Streamlit**: http://localhost:8501
2. **Configure Prometheus URL**: Should auto-connect to http://prometheus:9090
3. **Custom Queries**: Use the "Raw Data Explorer" for custom PromQL queries

### Example Queries
```promql
# Portfolio performance over time
portfolio_value_total

# Trading activity by symbol
sum(rate(trades_total[5m])) by (symbol)

# Risk metrics
current_drawdown_percent

# System health
hft_service_up
```

## 🔍 Troubleshooting

### Common Issues

1. **Services Not Showing in Prometheus**:
   ```bash
   # Check if service is exposing metrics
   curl http://localhost:8000/metrics
   
   # Check Prometheus config
   docker-compose logs prometheus
   ```

2. **Grafana Dashboard Empty**:
   - Verify Prometheus datasource connection
   - Check if services are generating metrics
   - Ensure time range is appropriate

3. **Alerts Not Firing**:
   ```bash
   # Check AlertManager logs
   docker-compose logs alertmanager
   
   # Verify alert rules
   curl http://localhost:9090/api/v1/rules
   ```

4. **Streamlit Connection Issues**:
   - Verify Prometheus is accessible
   - Check network connectivity between containers
   - Review Streamlit logs: `docker-compose logs streamlit-app`

### Health Checks
```bash
# Check all monitoring services
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health # Grafana  
curl http://localhost:9093/-/healthy  # AlertManager
curl http://localhost:8501/_stcore/health # Streamlit
```

## 🎯 Next Steps

1. **Start Your HFT Services**: Ensure they include metrics endpoints
2. **Monitor Dashboard**: Watch real-time metrics in Grafana
3. **Test Alerts**: Trigger test alerts to verify notification channels
4. **Customize Dashboards**: Add service-specific panels and metrics
5. **Set Up Retention**: Configure long-term storage if needed

## 📋 Service Ports Reference

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Metrics collection & queries |
| Grafana | 3000 | Dashboards & visualization |
| AlertManager | 9093 | Alert management |
| Streamlit | 8501 | Ad-hoc analysis |
| Node Exporter | 9100 | System metrics |
| cAdvisor | 8080 | Container metrics |

## 🔐 Security Notes

- Change default Grafana password in production
- Configure proper authentication for external access
- Use HTTPS in production environments
- Restrict network access to monitoring ports
- Regular backup of dashboard configurations

---

**Need Help?** Check the logs: `docker-compose logs [service-name]`
