# Kalman Filter Service Dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd -r hftuser && useradd -r -g hftuser hftuser

# Install runtime dependencies (including math libraries)
RUN apt-get update && apt-get install -y \
    curl \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY common/ ./common/
COPY services/kalman_filter/ ./services/kalman_filter/

# Change ownership
RUN chown -R hftuser:hftuser /app

# Switch to non-root user
USER hftuser

# Expose metrics port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=kalman-filter
ENV METRICS_PORT=8001

# Run the service
CMD ["python", "-m", "services.kalman_filter.main"]
