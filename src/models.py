"""Data models for the crypto HFT bot."""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import hashlib
import json


class PriceLevel(BaseModel):
    """Represents a single price level in the order book."""
    price: Decimal
    quantity: Decimal
    
    @field_validator('price', 'quantity', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert string values to Decimal for precision."""
        if isinstance(v, str):
            return Decimal(v)
        return v


class OrderBook(BaseModel):
    """Represents a complete order book snapshot."""
    symbol: str
    bids: List[PriceLevel] = Field(default_factory=list)
    asks: List[PriceLevel] = Field(default_factory=list)
    last_update_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for order book validation."""
        # Sort bids (highest price first) and asks (lowest price first)
        sorted_bids = sorted(self.bids, key=lambda x: x.price, reverse=True)
        sorted_asks = sorted(self.asks, key=lambda x: x.price)
        
        # Create checksum string
        checksum_data = []
        
        # Add top 10 bids and asks for checksum
        for i in range(min(10, len(sorted_bids))):
            bid = sorted_bids[i]
            checksum_data.append(f"{bid.price}:{bid.quantity}")
            
        for i in range(min(10, len(sorted_asks))):
            ask = sorted_asks[i]
            checksum_data.append(f"{ask.price}:{ask.quantity}")
        
        checksum_string = "|".join(checksum_data)
        return hashlib.md5(checksum_string.encode()).hexdigest()
    
    def get_best_bid(self) -> Optional[PriceLevel]:
        """Get the best bid (highest price)."""
        if not self.bids:
            return None
        return max(self.bids, key=lambda x: x.price)
    
    def get_best_ask(self) -> Optional[PriceLevel]:
        """Get the best ask (lowest price)."""
        if not self.asks:
            return None
        return min(self.asks, key=lambda x: x.price)
    
    def get_spread(self) -> Optional[Decimal]:
        """Calculate the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None


class OrderBookUpdate(BaseModel):
    """Represents an incremental order book update."""
    symbol: str
    first_update_id: int
    final_update_id: int
    bids: List[PriceLevel] = Field(default_factory=list)
    asks: List[PriceLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KafkaMessage(BaseModel):
    """Wrapper for Kafka messages."""
    topic: str
    key: str
    value: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_json(self) -> str:
        """Convert to JSON string for Kafka."""
        return json.dumps({
            "topic": self.topic,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat()
        }, default=str)


class WebSocketMessage(BaseModel):
    """Represents a WebSocket message from Binance."""
    stream: str
    data: Dict[str, Any]
    
    @classmethod
    def from_binance_depth(cls, message: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from Binance depth stream message."""
        return cls(
            stream=message.get("stream", ""),
            data=message.get("data", {})
        )


class HealthStatus(BaseModel):
    """Health status of the application."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    websocket_connected: bool = False
    kafka_connected: bool = False
    last_message_time: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0
