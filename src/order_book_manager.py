"""Order book manager with validation and checksum verification."""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import structlog
import aiohttp

from .models import OrderBook, OrderBookUpdate, PriceLevel, WebSocketMessage
from .websocket_client import BinanceWebSocketClient, parse_order_book_update
from .kafka_producer import OrderBookKafkaProducer
from .config import settings

logger = structlog.get_logger(__name__)


class OrderBookManager:
    """Manages order books for multiple symbols with validation."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.order_books: Dict[str, OrderBook] = {}
        self.last_update_ids: Dict[str, int] = {}
        
        # Components
        self.websocket_client = BinanceWebSocketClient(symbols, self._on_websocket_message)
        self.kafka_producer = OrderBookKafkaProducer()
        
        # Validation settings
        self.snapshot_interval = timedelta(minutes=5)  # Refresh snapshots every 5 minutes
        self.last_snapshot_times: Dict[str, datetime] = {}
        
        # Statistics
        self.update_counts: Dict[str, int] = {symbol: 0 for symbol in symbols}
        self.error_counts: Dict[str, int] = {symbol: 0 for symbol in symbols}
        
    async def start(self):
        """Start the order book manager."""
        logger.info("Starting Order Book Manager", symbols=self.symbols)
        
        # Connect to Kafka
        kafka_connected = await self.kafka_producer.connect()
        if not kafka_connected:
            logger.error("Failed to connect to Kafka")
            return False
        
        # Initialize order books with snapshots
        for symbol in self.symbols:
            success = await self._initialize_order_book(symbol)
            if not success:
                logger.error("Failed to initialize order book", symbol=symbol)
                return False
        
        # Start WebSocket connection
        websocket_connected = await self.websocket_client.connect()
        if not websocket_connected:
            logger.error("Failed to connect to WebSocket")
            return False
        
        # Start background tasks
        asyncio.create_task(self.websocket_client.listen())
        asyncio.create_task(self._periodic_snapshot_refresh())
        asyncio.create_task(self._health_monitor())
        
        logger.info("Order Book Manager started successfully")
        return True
    
    async def stop(self):
        """Stop the order book manager."""
        logger.info("Stopping Order Book Manager")
        
        await self.websocket_client.disconnect()
        await self.kafka_producer.disconnect()
        
        logger.info("Order Book Manager stopped")
    
    async def _initialize_order_book(self, symbol: str) -> bool:
        """Initialize order book with a snapshot from REST API."""
        try:
            logger.info("Initializing order book", symbol=symbol)
            
            # Fetch snapshot from Binance REST API
            snapshot_data = await self._fetch_order_book_snapshot(symbol)
            if not snapshot_data:
                return False
            
            # Create order book
            bids = [PriceLevel(price=bid[0], quantity=bid[1]) for bid in snapshot_data['bids']]
            asks = [PriceLevel(price=ask[0], quantity=ask[1]) for ask in snapshot_data['asks']]
            
            order_book = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                last_update_id=snapshot_data['lastUpdateId']
            )
            
            # Validate order book
            if not self._validate_order_book(order_book):
                logger.error("Order book validation failed", symbol=symbol)
                return False
            
            # Store order book
            self.order_books[symbol] = order_book
            self.last_update_ids[symbol] = order_book.last_update_id
            self.last_snapshot_times[symbol] = datetime.utcnow()
            
            # Publish initial snapshot
            await self.kafka_producer.publish_order_book_snapshot(order_book)
            
            logger.info("Order book initialized", 
                       symbol=symbol, 
                       last_update_id=order_book.last_update_id,
                       checksum=order_book.calculate_checksum())
            
            return True
            
        except Exception as e:
            logger.error("Error initializing order book", symbol=symbol, error=str(e))
            return False
    
    async def _fetch_order_book_snapshot(self, symbol: str) -> Optional[Dict]:
        """Fetch order book snapshot from Binance REST API."""
        try:
            url = f"{settings.binance_api_base_url}/v3/depth"
            params = {
                'symbol': symbol,
                'limit': settings.order_book_depth
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error("Failed to fetch order book snapshot", 
                                   symbol=symbol, status=response.status)
                        return None
                        
        except Exception as e:
            logger.error("Error fetching order book snapshot", symbol=symbol, error=str(e))
            return None
    
    async def _on_websocket_message(self, ws_message: WebSocketMessage):
        """Handle incoming WebSocket messages."""
        try:
            # Parse order book update
            update = parse_order_book_update(ws_message)
            if not update:
                return
            
            # Process the update
            await self._process_order_book_update(update)
            
        except Exception as e:
            logger.error("Error processing WebSocket message", error=str(e))
    
    async def _process_order_book_update(self, update: OrderBookUpdate):
        """Process an order book update."""
        try:
            symbol = update.symbol
            
            if symbol not in self.order_books:
                logger.warning("Received update for uninitialized symbol", symbol=symbol)
                return
            
            current_order_book = self.order_books[symbol]
            
            # Validate update sequence
            if not self._validate_update_sequence(symbol, update):
                logger.warning("Invalid update sequence, refreshing snapshot", 
                             symbol=symbol,
                             expected=self.last_update_ids[symbol] + 1,
                             received_first=update.first_update_id,
                             received_final=update.final_update_id)
                await self._initialize_order_book(symbol)
                return
            
            # Apply update to order book
            updated_order_book = self._apply_update(current_order_book, update)
            
            # Validate updated order book
            if not self._validate_order_book(updated_order_book):
                logger.error("Updated order book validation failed", symbol=symbol)
                self.error_counts[symbol] += 1
                return
            
            # Update stored order book
            self.order_books[symbol] = updated_order_book
            self.last_update_ids[symbol] = update.final_update_id
            self.update_counts[symbol] += 1
            
            # Publish update to Kafka
            await self.kafka_producer.publish_order_book_update(update)
            
            # Periodically publish full snapshots
            if self._should_publish_snapshot(symbol):
                await self.kafka_producer.publish_order_book_snapshot(updated_order_book)
                self.last_snapshot_times[symbol] = datetime.utcnow()
            
        except Exception as e:
            logger.error("Error processing order book update", 
                        symbol=update.symbol, error=str(e))
            self.error_counts[update.symbol] += 1
    
    def _validate_update_sequence(self, symbol: str, update: OrderBookUpdate) -> bool:
        """Validate that the update sequence is correct."""
        last_update_id = self.last_update_ids.get(symbol, 0)
        
        # First update ID should be last_update_id + 1
        # Final update ID should be >= first update ID
        return (update.first_update_id == last_update_id + 1 and 
                update.final_update_id >= update.first_update_id)
    
    def _apply_update(self, order_book: OrderBook, update: OrderBookUpdate) -> OrderBook:
        """Apply an update to an order book."""
        # Create copies of current bids and asks
        current_bids = {bid.price: bid.quantity for bid in order_book.bids}
        current_asks = {ask.price: ask.quantity for ask in order_book.asks}
        
        # Apply bid updates
        for bid in update.bids:
            if bid.quantity == 0:
                # Remove price level
                current_bids.pop(bid.price, None)
            else:
                # Update price level
                current_bids[bid.price] = bid.quantity
        
        # Apply ask updates
        for ask in update.asks:
            if ask.quantity == 0:
                # Remove price level
                current_asks.pop(ask.price, None)
            else:
                # Update price level
                current_asks[ask.price] = ask.quantity
        
        # Convert back to PriceLevel objects
        new_bids = [PriceLevel(price=price, quantity=quantity) 
                   for price, quantity in current_bids.items()]
        new_asks = [PriceLevel(price=price, quantity=quantity) 
                   for price, quantity in current_asks.items()]
        
        # Sort bids (highest first) and asks (lowest first)
        new_bids.sort(key=lambda x: x.price, reverse=True)
        new_asks.sort(key=lambda x: x.price)
        
        # Limit to configured depth
        new_bids = new_bids[:settings.order_book_depth]
        new_asks = new_asks[:settings.order_book_depth]
        
        return OrderBook(
            symbol=order_book.symbol,
            bids=new_bids,
            asks=new_asks,
            last_update_id=update.final_update_id
        )
    
    def _validate_order_book(self, order_book: OrderBook) -> bool:
        """Validate order book integrity."""
        try:
            # Check that bids are sorted in descending order
            for i in range(len(order_book.bids) - 1):
                if order_book.bids[i].price <= order_book.bids[i + 1].price:
                    logger.error("Bids not properly sorted", symbol=order_book.symbol)
                    return False
            
            # Check that asks are sorted in ascending order
            for i in range(len(order_book.asks) - 1):
                if order_book.asks[i].price >= order_book.asks[i + 1].price:
                    logger.error("Asks not properly sorted", symbol=order_book.symbol)
                    return False
            
            # Check that best bid < best ask (no crossed book)
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if best_bid and best_ask and best_bid.price >= best_ask.price:
                logger.error("Crossed order book detected", 
                           symbol=order_book.symbol,
                           best_bid=str(best_bid.price),
                           best_ask=str(best_ask.price))
                return False
            
            # Check for negative quantities
            for bid in order_book.bids:
                if bid.quantity <= 0:
                    logger.error("Negative or zero bid quantity", symbol=order_book.symbol)
                    return False
            
            for ask in order_book.asks:
                if ask.quantity <= 0:
                    logger.error("Negative or zero ask quantity", symbol=order_book.symbol)
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating order book", symbol=order_book.symbol, error=str(e))
            return False
    
    def _should_publish_snapshot(self, symbol: str) -> bool:
        """Check if we should publish a full snapshot."""
        last_snapshot = self.last_snapshot_times.get(symbol)
        if not last_snapshot:
            return True
        
        return datetime.utcnow() - last_snapshot >= self.snapshot_interval
    
    async def _periodic_snapshot_refresh(self):
        """Periodically refresh order book snapshots."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for symbol in self.symbols:
                    if self._should_publish_snapshot(symbol):
                        logger.info("Refreshing order book snapshot", symbol=symbol)
                        await self._initialize_order_book(symbol)
                        
            except Exception as e:
                logger.error("Error in periodic snapshot refresh", error=str(e))
    
    async def _health_monitor(self):
        """Monitor the health of the order book manager."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check WebSocket health
                ws_healthy = await self.websocket_client.health_check()
                kafka_healthy = self.kafka_producer.is_connected
                
                if not ws_healthy:
                    logger.warning("WebSocket connection unhealthy")
                
                if not kafka_healthy:
                    logger.warning("Kafka connection unhealthy")
                
                # Log statistics
                logger.info("Order Book Manager Health Check",
                           websocket_healthy=ws_healthy,
                           kafka_healthy=kafka_healthy,
                           update_counts=self.update_counts,
                           error_counts=self.error_counts)
                
            except Exception as e:
                logger.error("Error in health monitor", error=str(e))
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get the current order book for a symbol."""
        return self.order_books.get(symbol)
    
    def get_stats(self) -> Dict:
        """Get manager statistics."""
        return {
            "symbols": self.symbols,
            "order_books_count": len(self.order_books),
            "update_counts": self.update_counts,
            "error_counts": self.error_counts,
            "websocket_stats": self.websocket_client.get_stats(),
            "kafka_connected": self.kafka_producer.is_connected
        }
