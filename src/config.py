"""Configuration management for the crypto HFT bot."""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Binance Configuration
    binance_api_key: str = Field(..., env="BINANCE_API_KEY")
    binance_secret_key: str = Field(..., env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(True, env="BINANCE_TESTNET")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_order_book: str = Field("order_book_updates", env="KAFKA_TOPIC_ORDER_BOOK")
    kafka_client_id: str = Field("crypto_hft_bot", env="KAFKA_CLIENT_ID")
    
    # Trading Configuration
    symbols: str = Field("BTCUSDT,ETHUSDT", env="SYMBOLS")
    order_book_depth: int = Field(20, env="ORDER_BOOK_DEPTH")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    # Monitoring
    prometheus_port: int = Field(8000, env="PROMETHEUS_PORT")
    health_check_port: int = Field(8001, env="HEALTH_CHECK_PORT")
    
    class Config:
        env_file = "config.env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @property
    def binance_ws_base_url(self) -> str:
        """Get the appropriate Binance WebSocket URL based on testnet setting."""
        if self.binance_testnet:
            # Binance US testnet WebSocket URL
            return "wss://testnet.binance.us/ws"
        # Binance US production WebSocket URL  
        return "wss://stream.binance.us:9443/ws"
    
    @property
    def binance_api_base_url(self) -> str:
        """Get the appropriate Binance API URL based on testnet setting."""
        if self.binance_testnet:
            # Binance US testnet API URL
            return "https://testnet.binance.us/api"
        # Binance US production API URL
        return "https://api.binance.us/api"
    
    @property
    def symbols_list(self) -> List[str]:
        """Get symbols as a list."""
        return [s.strip() for s in self.symbols.split(',') if s.strip()]


# Global settings instance
settings = Settings()
