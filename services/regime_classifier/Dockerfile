# Use the base image
FROM hft-crypto-bot-base:latest AS production

# Set working directory
WORKDIR /app

# Copy regime classifier service files
COPY services/regime_classifier/ services/regime_classifier/
COPY services/common/ services/common/

# Create models directory
RUN mkdir -p models

# Set Python path
ENV PYTHONPATH=/app

# Expose health check port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8007/health', timeout=5)" || exit 1

# Run the service
CMD ["python", "-m", "services.regime_classifier.main"]
