"""Binance WebSocket client for ingestion service."""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
import structlog
from datetime import datetime, timedelta

from .config import settings
from common.logger import get_logger, get_performance_logger

logger = get_logger(__name__)
perf_logger = get_performance_logger("ingestion-service")


class WebSocketMessage:
    """WebSocket message wrapper."""
    
    def __init__(self, stream: str, data: Dict[str, Any]):
        self.stream = stream
        self.data = data
        self.timestamp = datetime.utcnow()
    
    @classmethod
    def from_binance_depth(cls, raw_data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create WebSocketMessage from Binance depth stream data."""
        return cls(
            stream=raw_data.get('stream', ''),
            data=raw_data.get('data', raw_data)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'stream': self.stream,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }


class PriceLevel:
    """Order book price level."""
    
    def __init__(self, price: str, quantity: str):
        self.price = float(price)
        self.quantity = float(quantity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'price': self.price,
            'quantity': self.quantity
        }


class OrderBookUpdate:
    """Order book update message."""
    
    def __init__(
        self,
        symbol: str,
        first_update_id: int,
        final_update_id: int,
        bids: List[PriceLevel],
        asks: List[PriceLevel]
    ):
        self.symbol = symbol
        self.first_update_id = first_update_id
        self.final_update_id = final_update_id
        self.bids = bids
        self.asks = asks
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'first_update_id': self.first_update_id,
            'final_update_id': self.final_update_id,
            'bids': [bid.to_dict() for bid in self.bids],
            'asks': [ask.to_dict() for ask in self.asks],
            'timestamp': self.timestamp.isoformat()
        }


class BinanceWebSocketClient:
    """WebSocket client for Binance order book streams."""
    
    def __init__(self, symbols: List[str], on_message_callback: Callable[[WebSocketMessage], None]):
        self.symbols = [symbol.lower() for symbol in symbols]
        self.on_message_callback = on_message_callback
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60  # Max 60 seconds
        self.last_message_time: Optional[datetime] = None
        self.message_count = 0
        self.error_count = 0
        
        # Health monitoring
        self.heartbeat_interval = 30  # seconds
        self.message_timeout = 60  # seconds
        
    def _build_stream_url(self) -> str:
        """Build the WebSocket stream URL for multiple symbols."""
        # Create depth streams for all symbols
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol}@depth{settings.order_book_depth}@100ms")
        
        stream_names = "/".join(streams)
        return f"{settings.binance_ws_base_url}/{stream_names}"
    
    async def connect(self) -> bool:
        """Connect to Binance WebSocket."""
        start_time = datetime.utcnow()
        
        try:
            url = self._build_stream_url()
            logger.info("Connecting to Binance WebSocket", url=url)
            
            self.websocket = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10**7,  # 10MB max message size
                compression=None  # Disable compression for lower latency
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            self.reconnect_delay = 1
            self.last_message_time = datetime.utcnow()
            
            connect_time = (datetime.utcnow() - start_time).total_seconds()
            perf_logger.log_processing_time("websocket_connect", connect_time * 1000)
            
            logger.info("Connected to Binance WebSocket", symbols=self.symbols)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Binance WebSocket", error=str(e))
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.connected = False
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Disconnected from Binance WebSocket")
            except Exception as e:
                logger.error("Error disconnecting from WebSocket", error=str(e))
    
    async def listen(self):
        """Listen for WebSocket messages with automatic reconnection."""
        while True:
            try:
                if not self.connected:
                    if not await self._reconnect():
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                
                # Listen for messages
                await self._listen_messages()
                
            except Exception as e:
                logger.error("Error in WebSocket listener", error=str(e))
                self.error_count += 1
                self.connected = False
                await asyncio.sleep(1)
    
    async def _listen_messages(self):
        """Listen for incoming WebSocket messages."""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                start_time = datetime.utcnow()
                
                try:
                    self.last_message_time = datetime.utcnow()
                    self.message_count += 1
                    
                    # Parse JSON message
                    data = json.loads(message)
                    logger.debug("Raw WebSocket message received", data=data)
                    
                    # Handle different message types
                    if isinstance(data, dict):
                        if "stream" in data and "data" in data:
                            # Single stream message
                            logger.debug("Processing single stream message", stream=data.get("stream"))
                            ws_message = WebSocketMessage.from_binance_depth(data)
                            await self._process_message(ws_message)
                        elif "result" in data or "id" in data:
                            # Subscription confirmation or error
                            logger.info("WebSocket response", data=data)
                        else:
                            # Direct depth data (no stream wrapper)
                            logger.debug("Processing direct depth data", data_keys=list(data.keys()))
                            # Try to create a WebSocket message directly
                            try:
                                ws_message = WebSocketMessage(
                                    stream=f"{self.symbols[0]}@depth",
                                    data=data
                                )
                                await self._process_message(ws_message)
                            except Exception as e:
                                logger.error("Failed to process direct depth data", error=str(e))
                    elif isinstance(data, list):
                        # Multiple messages
                        for item in data:
                            if "stream" in item and "data" in item:
                                ws_message = WebSocketMessage.from_binance_depth(item)
                                await self._process_message(ws_message)
                    
                    # Log processing time
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    perf_logger.log_processing_time("websocket_message_processing", processing_time * 1000)
                    
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse WebSocket message", error=str(e), message=message)
                    self.error_count += 1
                except Exception as e:
                    logger.error("Error processing WebSocket message", error=str(e))
                    self.error_count += 1
                    
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
        except WebSocketException as e:
            logger.error("WebSocket exception", error=str(e))
            self.connected = False
            self.error_count += 1
    
    async def _process_message(self, ws_message: WebSocketMessage):
        """Process a WebSocket message."""
        try:
            # Call the callback function
            if self.on_message_callback:
                await asyncio.create_task(self._safe_callback(ws_message))
                
        except Exception as e:
            logger.error("Error in message callback", error=str(e))
            self.error_count += 1
    
    async def _safe_callback(self, ws_message: WebSocketMessage):
        """Safely execute the callback function."""
        try:
            if asyncio.iscoroutinefunction(self.on_message_callback):
                await self.on_message_callback(ws_message)
            else:
                self.on_message_callback(ws_message)
        except Exception as e:
            logger.error("Error in message callback execution", error=str(e))
            raise
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached", 
                        attempts=self.reconnect_attempts)
            return False
        
        self.reconnect_attempts += 1
        
        logger.info("Attempting to reconnect", 
                   attempt=self.reconnect_attempts,
                   delay=self.reconnect_delay)
        
        await asyncio.sleep(self.reconnect_delay)
        
        success = await self.connect()
        
        if not success:
            # Exponential backoff
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
        return success
    
    async def health_check(self) -> bool:
        """Check the health of the WebSocket connection."""
        if not self.connected:
            return False
        
        # Check if we've received messages recently
        if self.last_message_time:
            time_since_last_message = datetime.utcnow() - self.last_message_time
            if time_since_last_message > timedelta(seconds=self.message_timeout):
                logger.warning("No messages received recently", 
                              seconds_since_last=time_since_last_message.total_seconds())
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self.connected,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "reconnect_attempts": self.reconnect_attempts,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "symbols": self.symbols
        }


def parse_order_book_update(ws_message: WebSocketMessage) -> Optional[OrderBookUpdate]:
    """Parse a WebSocket message into an OrderBookUpdate."""
    try:
        data = ws_message.data
        
        # Extract symbol from stream name
        stream_parts = ws_message.stream.split('@')
        if len(stream_parts) < 2:
            return None
        
        symbol = stream_parts[0].upper()
        
        # Parse bids and asks - Binance uses 'bids' and 'asks' keys
        bids = []
        for bid_data in data.get('bids', []):
            if len(bid_data) >= 2:
                price, quantity = bid_data[0], bid_data[1]
                if float(quantity) > 0:  # Only include non-zero quantities
                    bids.append(PriceLevel(price=price, quantity=quantity))
        
        asks = []
        for ask_data in data.get('asks', []):
            if len(ask_data) >= 2:
                price, quantity = ask_data[0], ask_data[1]
                if float(quantity) > 0:  # Only include non-zero quantities
                    asks.append(PriceLevel(price=price, quantity=quantity))
        
        return OrderBookUpdate(
            symbol=symbol,
            first_update_id=data.get('U', data.get('lastUpdateId', 0)),
            final_update_id=data.get('u', data.get('lastUpdateId', 0)),
            bids=bids,
            asks=asks
        )
        
    except Exception as e:
        logger.error("Error parsing order book update", error=str(e), message=ws_message.to_dict())
        return None
