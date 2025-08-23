# Crypto HFT Arbitrage Bot

A high-frequency trading (HFT) arbitrage bot for cryptocurrency markets, implementing **HFT-001: Binance WebSocket Order Book Ingestor** as the foundation component.

## 🚀 Features

- **Real-time Order Book Ingestion**: WebSocket connection to Binance with automatic reconnection
- **Kafka Integration**: Publishes order book updates to Kafka for downstream processing
- **Data Validation**: Checksum validation and order book integrity checks
- **High Performance**: Optimized for low-latency HFT requirements
- **Comprehensive Testing**: Unit, integration, and performance tests
- **Production Ready**: Docker containerization with monitoring stack

## 📋 Requirements

- Python 3.11+
- Docker & Docker Compose
- Binance API credentials (testnet supported)
- Kafka cluster

## 🛠 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd crypto-bot
cp config.env.example config.env
```

### 2. Configure Environment

Edit `config.env` with your settings:

```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=true
SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT
```

### 3. Start with Docker

```bash
# Start all services
make docker-up

# Create Kafka topics
make kafka-topics

# View logs
make logs
```

### 4. Development Setup

```bash
# Install dependencies
make install

# Run tests
make test

# Run with coverage
make test-cov
```

## 🏗 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Binance API   │───▶│  WebSocket       │───▶│  Order Book     │
│   (WebSocket)   │    │  Client          │    │  Manager        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Downstream    │◀───│     Kafka        │◀───│  Kafka          │
│   Consumers     │    │    Cluster       │    │  Producer       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Monitoring

- **Prometheus**: Metrics collection at `:9090`
- **Grafana**: Visualization at `:3000` (admin/admin)
- **Kafka UI**: Kafka monitoring at `:8080`
- **Health Check**: Application health at `:8001/health`

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-performance   # Performance tests

# Generate coverage report
make test-cov
```

## 📈 Performance Targets

- **Message Processing**: < 1ms per order book update
- **WebSocket Reconnection**: < 5 seconds
- **Order Book Validation**: < 0.1ms
- **Kafka Publishing**: < 2ms end-to-end

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Binance API key | Required |
| `BINANCE_SECRET_KEY` | Binance secret key | Required |
| `BINANCE_TESTNET` | Use testnet | `true` |
| `SYMBOLS` | Trading symbols | `BTCUSDT,ETHUSDT` |
| `ORDER_BOOK_DEPTH` | Order book depth | `20` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka servers | `localhost:9092` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Kafka Topics

- `order_book_updates`: Order book snapshots and updates

## 🐳 Docker Services

- **crypto-bot**: Main application
- **kafka**: Message broker
- **zookeeper**: Kafka coordination
- **redis**: Caching layer
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboard
- **kafka-ui**: Kafka management

## 📝 API Endpoints

- `GET /health`: Health check endpoint
- `GET /metrics`: Prometheus metrics

## 🔍 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   ```bash
   # Check network connectivity
   curl -I https://testnet.binance.vision
   
   # Verify API credentials
   docker-compose logs crypto-bot
   ```

2. **Kafka Connection Issues**
   ```bash
   # Check Kafka health
   make kafka-list-topics
   
   # Restart Kafka
   docker-compose restart kafka
   ```

3. **High Memory Usage**
   ```bash
   # Monitor container resources
   docker stats
   
   # Check for memory leaks in logs
   docker-compose logs crypto-bot | grep -i memory
   ```

## 🚦 Development Workflow

1. **Make Changes**: Edit source code in `src/`
2. **Run Tests**: `make test`
3. **Check Linting**: `make lint`
4. **Format Code**: `make format`
5. **Test Locally**: `make docker-up`
6. **Check Logs**: `make logs`

## 📊 Metrics

The application exposes Prometheus metrics:

- `websocket_messages_total`: Total WebSocket messages
- `errors_total`: Total errors by component
- `order_book_update_duration_seconds`: Update processing time
- `active_connections`: Active connection count

## 🔐 Security

- Non-root Docker containers
- Environment variable configuration
- API key protection
- Network isolation

## 📚 Project Structure

```
crypto-bot/
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models
│   ├── websocket_client.py # WebSocket client
│   ├── kafka_producer.py  # Kafka producer
│   ├── order_book_manager.py # Order book management
│   └── main.py            # Application entry point
├── tests/                 # Test suite
├── monitoring/            # Monitoring configuration
├── docker-compose.yml     # Docker services
├── Dockerfile            # Application container
├── requirements.txt      # Python dependencies
└── Makefile              # Development commands
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Open an issue with detailed information

---

**⚠️ Disclaimer**: This software is for educational purposes. Trading cryptocurrencies involves risk. Use at your own discretion.
