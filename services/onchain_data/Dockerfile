# Multi-stage Dockerfile for On-Chain Data Service
FROM hft-crypto-bot-base:latest AS production

# Set working directory
WORKDIR /app

# Copy service-specific requirements if any
# COPY services/onchain_data/requirements.txt ./requirements-onchain.txt
# RUN pip install --no-cache-dir -r requirements-onchain.txt

# Copy the on-chain data service code
COPY services/onchain_data/ ./services/onchain_data/
COPY common/ ./common/

# Create data directory for models and cache
RUN mkdir -p /app/data/models /app/data/cache && \
    chown -R 1001:1001 /app/data

# Set environment variables
ENV SERVICE_NAME=onchain-data
ENV SERVICE_PORT=8006
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

# Run as non-root user
USER 1001

# Start the service
CMD ["python", "-m", "services.onchain_data.main"]
