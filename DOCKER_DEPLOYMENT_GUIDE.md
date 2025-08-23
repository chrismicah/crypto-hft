# 🐳 Docker Deployment Guide

## 🚀 Quick Start

### Prerequisites
1. **Docker Desktop** installed and running
2. **Docker Compose** v2.0+ installed
3. **Git** for cloning the repository

### 1. Environment Setup
```bash
# Copy environment files
cp env.dev.example .env.dev
cp env.prod.example .env.prod

# Edit environment files with your API keys
vim .env.dev  # For development
vim .env.prod # For production
```

### 2. Development Deployment
```bash
# Start development stack
./scripts/deploy.sh -e development up

# Or using docker-compose directly
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 3. Production Deployment
```bash
# Start production stack
./scripts/deploy.sh -e production up

# Or using docker-compose directly
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📋 Service Architecture

### Core HFT Services
| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| `ingestion-service` | 8000 | Market data ingestion | `/health` |
| `kalman-filter-service` | 8001 | Dynamic hedge ratios | `/health` |
| `garch-volatility-service` | 8002 | Volatility forecasting | `/health` |
| `execution-service` | 8003 | Trading execution | `/health` |
| `risk-manager-service` | 8004 | Risk management & BOCD | `/health` |

### Infrastructure Services
| Service | Port | Purpose |
|---------|------|---------|
| `kafka` | 9092 | Message broker |
| `zookeeper` | 2181 | Kafka coordination |
| `redis` | 6379 | Caching & service registry |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards |
| `alertmanager` | 9093 | Alert management |
| `streamlit-app` | 8501 | Analysis dashboard |

## 🛠️ Build & Deploy Commands

### Using Deployment Script
```bash
# Build all services
./scripts/docker-build.sh

# Build specific service
./scripts/docker-build.sh ingestion

# Deploy development environment
./scripts/deploy.sh -e development up

# Deploy production environment
./scripts/deploy.sh -e production up

# Check service health
./scripts/deploy.sh health

# View logs
./scripts/deploy.sh logs execution-service

# Open shell in container
./scripts/deploy.sh shell execution-service
```

### Manual Docker Commands
```bash
# Build all images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Check status
docker-compose ps

# Stop services
docker-compose down

# Clean up
docker-compose down -v --remove-orphans
```

## 🔧 Configuration

### Environment Variables

#### Development (.env.dev)
- `BINANCE_TESTNET=true` - Use Binance testnet
- `LOG_LEVEL=DEBUG` - Verbose logging
- `SYMBOLS=BTCUSDT,ETHUSDT` - Limited symbols for testing
- `MAX_POSITION_SIZE=1000` - Small position sizes

#### Production (.env.prod)
- `BINANCE_TESTNET=false` - **LIVE TRADING** ⚠️
- `LOG_LEVEL=INFO` - Production logging
- `SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT,BNBUSDT,SOLUSDT` - Full symbol set
- `MAX_POSITION_SIZE=50000` - Production position sizes

### Docker Compose Profiles

#### Development Profile
```bash
# Start with development tools
docker-compose --profile dev-tools up -d

# Includes: Jupyter, Adminer, Redis Commander
```

#### Production Profile
```bash
# Production deployment (excludes debug tools)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📊 Monitoring & Health Checks

### Service Health
```bash
# Check all service health
curl http://localhost:8000/health  # Ingestion
curl http://localhost:8001/health  # Kalman Filter
curl http://localhost:8002/health  # GARCH Model
curl http://localhost:8003/health  # Execution
curl http://localhost:8004/health  # Risk Manager
```

### Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Streamlit**: http://localhost:8501
- **Kafka UI**: http://localhost:8090

### Log Aggregation
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f execution-service

# Follow logs with timestamps
docker-compose logs -f -t execution-service
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check Docker daemon
docker info

# Check compose file syntax
docker-compose config

# Check service dependencies
docker-compose ps
```

#### 2. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000

# Kill process using port
sudo kill -9 $(lsof -t -i:8000)
```

#### 3. Memory Issues
```bash
# Check container resource usage
docker stats

# Increase Docker memory limit in Docker Desktop
# Settings > Resources > Memory
```

#### 4. Network Issues
```bash
# Check Docker networks
docker network ls

# Inspect network
docker network inspect crypto-bot_crypto-bot-network

# Test connectivity between containers
docker-compose exec execution-service ping kafka
```

### Debug Mode

#### Enable Debug Logging
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or edit .env file
echo "LOG_LEVEL=DEBUG" >> .env.dev
```

#### Access Container Shell
```bash
# Open bash shell in container
docker-compose exec execution-service /bin/bash

# Run commands inside container
docker-compose exec execution-service python -c "import sys; print(sys.path)"
```

## 🚨 Production Considerations

### Security
- [ ] Change default passwords in `.env.prod`
- [ ] Use secrets management (Docker Secrets, Vault)
- [ ] Enable SSL/TLS for external access
- [ ] Restrict network access with firewall rules
- [ ] Regular security updates

### Performance
- [ ] Tune JVM settings for Kafka
- [ ] Configure resource limits per service
- [ ] Monitor memory and CPU usage
- [ ] Set up log rotation
- [ ] Configure persistent volumes

### Reliability
- [ ] Set up health checks for all services
- [ ] Configure restart policies
- [ ] Implement graceful shutdown
- [ ] Set up backup procedures
- [ ] Monitor disk space

### Monitoring
- [ ] Configure Prometheus retention
- [ ] Set up alerting rules
- [ ] Configure notification channels (Slack, email)
- [ ] Set up log aggregation (ELK stack)
- [ ] Monitor business metrics

## 📈 Scaling

### Horizontal Scaling
```yaml
# Scale specific services
docker-compose up -d --scale kalman-filter-service=3

# Load balancer configuration needed for multiple instances
```

### Vertical Scaling
```yaml
# Increase resource limits in docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy HFT Bot
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Deploy
        run: |
          ./scripts/docker-build.sh
          ./scripts/deploy.sh -e production up
```

### Docker Registry
```bash
# Tag and push images
docker tag crypto-bot-execution:latest registry.com/crypto-bot-execution:v1.0.0
docker push registry.com/crypto-bot-execution:v1.0.0
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Grafana Documentation](https://grafana.com/docs/)

---

## 🆘 Need Help?

1. **Check logs**: `docker-compose logs -f [service]`
2. **Verify health**: `./scripts/deploy.sh health`
3. **Check resources**: `docker stats`
4. **Network issues**: `docker network inspect crypto-bot_crypto-bot-network`
5. **Clean start**: `docker-compose down -v && docker-compose up -d`
