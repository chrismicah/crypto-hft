"""Configuration management for the ingestion service."""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class IngestionSettings(BaseSettings):
    """Ingestion service settings loaded from environment variables."""
    
    # Binance Configuration
    binance_api_key: str = Field(..., env="BINANCE_API_KEY")
    binance_secret_key: str = Field(..., env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(True, env="BINANCE_TESTNET")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_order_book: str = Field("order_book_updates", env="KAFKA_TOPIC_ORDER_BOOK")
    kafka_topic_trade_ticks: str = Field("trade-ticks", env="KAFKA_TOPIC_TRADE_TICKS")
    kafka_client_id: str = Field("ingestion-service", env="KAFKA_CLIENT_ID")
    
    # Trading Configuration
    symbols: str = Field("BTCUSDT,ETHUSDT,ADAUSDT", env="SYMBOLS")
    order_book_depth: int = Field(20, env="ORDER_BOOK_DEPTH")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    # Monitoring
    prometheus_port: int = Field(8000, env="PROMETHEUS_PORT")
    health_check_port: int = Field(8001, env="HEALTH_CHECK_PORT")
    
    # Service Discovery
    service_name: str = Field("ingestion-service", env="SERVICE_NAME")
    service_version: str = Field("1.0.0", env="SERVICE_VERSION")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    class Config:
        env_file = "config.env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @property
    def binance_ws_base_url(self) -> str:
        """Get the appropriate Binance WebSocket URL based on testnet setting."""
        if self.binance_testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"
    
    @property
    def binance_api_base_url(self) -> str:
        """Get the appropriate Binance API URL based on testnet setting."""
        if self.binance_testnet:
            return "https://testnet.binance.vision/api"
        return "https://api.binance.com/api"
    
    @property
    def symbols_list(self) -> List[str]:
        """Get the list of symbols to track."""
        return [s.strip() for s in self.symbols.split(',') if s.strip()]


# Global settings instance
settings = IngestionSettings()
