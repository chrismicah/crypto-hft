# Crypto HFT Arbitrage Bot Makefile

.PHONY: help install test lint format clean docker-build docker-up docker-down logs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo "  docker-down  - Stop services"
	@echo "  logs         - View application logs"
	@echo "  kafka-topics - Create Kafka topics"

# Development setup
install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

test-performance:
	pytest tests/ -v -m "slow"

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache

# Docker operations
docker-build:
	docker build -t crypto-bot:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f crypto-bot

logs:
	docker-compose logs -f

# Kafka management
kafka-topics:
	docker-compose exec kafka kafka-topics --create --topic order_book_updates --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

kafka-list-topics:
	docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

kafka-describe-topic:
	docker-compose exec kafka kafka-topics --describe --topic order_book_updates --bootstrap-server localhost:9092

# Development
dev-setup: install kafka-topics
	@echo "Development environment ready!"

# Production
prod-up:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Monitoring
prometheus:
	open http://localhost:9090

grafana:
	open http://localhost:3000

kafka-ui:
	open http://localhost:8080

# Health checks
health:
	curl -f http://localhost:8001/health || echo "Service not healthy"

metrics:
	curl -s http://localhost:8000/metrics | head -20
