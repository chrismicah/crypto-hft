"""CCXT exchange interface wrapper for Binance Testnet integration."""

import ccxt
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import time

logger = structlog.get_logger(__name__)


@dataclass
class OrderResult:
    """Container for order execution results."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float]
    order_type: str  # 'market', 'limit', etc.
    status: str  # 'open', 'closed', 'canceled', etc.
    filled: float
    remaining: float
    cost: float
    fee: Optional[Dict[str, Any]]
    timestamp: datetime
    info: Dict[str, Any]


@dataclass
class PositionInfo:
    """Container for position information."""
    symbol: str
    side: str  # 'long', 'short', or 'none'
    size: float
    entry_price: Optional[float]
    current_price: float
    unrealized_pnl: float
    percentage: float
    timestamp: datetime


@dataclass
class BalanceInfo:
    """Container for account balance information."""
    currency: str
    free: float
    used: float
    total: float


class BinanceTestnetExchange:
    """
    CCXT wrapper for Binance Testnet trading operations.
    
    Provides a simplified interface for placing orders, checking positions,
    and managing account information on Binance Testnet.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the Binance exchange wrapper.
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key
            testnet: Whether to use testnet (default True)
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': testnet,  # Use testnet
            'rateLimit': int(rate_limit_delay * 1000),  # Convert to milliseconds
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Use spot trading
            }
        })
        
        # Cache for market info and symbols
        self._markets_cache: Optional[Dict[str, Any]] = None
        self._symbols_cache: Optional[List[str]] = None
        self._last_market_fetch: Optional[datetime] = None
        self._market_cache_ttl = timedelta(hours=1)
        
        # Order tracking
        self._active_orders: Dict[str, OrderResult] = {}
        self._order_history: List[OrderResult] = []
        
        logger.info(
            "Binance exchange initialized",
            testnet=testnet,
            rate_limit_delay=rate_limit_delay
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the exchange connection and load markets.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load markets
            await self._load_markets()
            
            # Test connection by fetching account balance
            balance = await self.get_balance()
            if balance is None:
                logger.error("Failed to fetch account balance during initialization")
                return False
            
            logger.info(
                "Exchange initialized successfully",
                testnet=self.testnet,
                available_symbols=len(self._symbols_cache) if self._symbols_cache else 0
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize exchange", error=str(e), exc_info=True)
            return False
    
    async def _load_markets(self) -> None:
        """Load market information from the exchange."""
        try:
            if (self._markets_cache is None or 
                self._last_market_fetch is None or 
                datetime.utcnow() - self._last_market_fetch > self._market_cache_ttl):
                
                logger.debug("Loading markets from exchange")
                self._markets_cache = await asyncio.get_event_loop().run_in_executor(
                    None, self.exchange.load_markets
                )
                self._symbols_cache = list(self._markets_cache.keys())
                self._last_market_fetch = datetime.utcnow()
                
                logger.info("Markets loaded", symbol_count=len(self._symbols_cache))
            
        except Exception as e:
            logger.error("Failed to load markets", error=str(e), exc_info=True)
            raise
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for CCXT."""
        # CCXT uses '/' format (e.g., 'BTC/USDT')
        if '/' not in symbol:
            # Convert from Binance format (e.g., 'BTCUSDT' -> 'BTC/USDT')
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                return f"{base}/USDT"
            elif symbol.endswith('BTC'):
                base = symbol[:-3]
                return f"{base}/BTC"
            elif symbol.endswith('ETH'):
                base = symbol[:-3]
                return f"{base}/ETH"
        
        return symbol
    
    def _round_amount(self, symbol: str, amount: float) -> float:
        """Round amount to exchange precision requirements."""
        try:
            if self._markets_cache and symbol in self._markets_cache:
                market = self._markets_cache[symbol]
                precision = market.get('precision', {}).get('amount', 8)
                return float(Decimal(str(amount)).quantize(
                    Decimal('0.1') ** precision, 
                    rounding=ROUND_DOWN
                ))
            
            # Default to 6 decimal places if no market info
            return round(amount, 6)
            
        except Exception as e:
            logger.warning("Failed to round amount", symbol=symbol, amount=amount, error=str(e))
            return round(amount, 6)
    
    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision requirements."""
        try:
            if self._markets_cache and symbol in self._markets_cache:
                market = self._markets_cache[symbol]
                precision = market.get('precision', {}).get('price', 8)
                return round(price, precision)
            
            # Default to 2 decimal places for USDT pairs
            if 'USDT' in symbol:
                return round(price, 2)
            return round(price, 8)
            
        except Exception as e:
            logger.warning("Failed to round price", symbol=symbol, price=price, error=str(e))
            return round(price, 8)
    
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: Optional[str] = None
    ) -> Optional[OrderResult]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT' or 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            client_order_id: Optional client order ID
            
        Returns:
            OrderResult if successful, None otherwise
        """
        try:
            # Normalize symbol and amount
            normalized_symbol = self._normalize_symbol(symbol)
            rounded_amount = self._round_amount(normalized_symbol, amount)
            
            if rounded_amount <= 0:
                logger.error("Invalid order amount", symbol=symbol, amount=amount)
                return None
            
            # Prepare order parameters
            params = {}
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            logger.info(
                "Placing market order",
                symbol=normalized_symbol,
                side=side,
                amount=rounded_amount,
                client_order_id=client_order_id
            )
            
            # Place order
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.create_market_order(
                    normalized_symbol, side, rounded_amount, None, None, params
                )
            )
            
            # Convert to OrderResult
            order_result = self._convert_order_to_result(order)
            
            # Track order
            self._active_orders[order_result.order_id] = order_result
            self._order_history.append(order_result)
            
            logger.info(
                "Market order placed successfully",
                order_id=order_result.order_id,
                symbol=order_result.symbol,
                side=order_result.side,
                amount=order_result.amount,
                status=order_result.status
            )
            
            return order_result
            
        except Exception as e:
            logger.error(
                "Failed to place market order",
                symbol=symbol,
                side=side,
                amount=amount,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        client_order_id: Optional[str] = None,
        time_in_force: str = 'GTC'
    ) -> Optional[OrderResult]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            price: Limit price
            client_order_id: Optional client order ID
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            
        Returns:
            OrderResult if successful, None otherwise
        """
        try:
            # Normalize symbol, amount, and price
            normalized_symbol = self._normalize_symbol(symbol)
            rounded_amount = self._round_amount(normalized_symbol, amount)
            rounded_price = self._round_price(normalized_symbol, price)
            
            if rounded_amount <= 0 or rounded_price <= 0:
                logger.error(
                    "Invalid order parameters",
                    symbol=symbol,
                    amount=amount,
                    price=price
                )
                return None
            
            # Prepare order parameters
            params = {'timeInForce': time_in_force}
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            logger.info(
                "Placing limit order",
                symbol=normalized_symbol,
                side=side,
                amount=rounded_amount,
                price=rounded_price,
                time_in_force=time_in_force
            )
            
            # Place order
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.create_limit_order(
                    normalized_symbol, side, rounded_amount, rounded_price, params
                )
            )
            
            # Convert to OrderResult
            order_result = self._convert_order_to_result(order)
            
            # Track order
            self._active_orders[order_result.order_id] = order_result
            self._order_history.append(order_result)
            
            logger.info(
                "Limit order placed successfully",
                order_id=order_result.order_id,
                symbol=order_result.symbol,
                side=order_result.side,
                amount=order_result.amount,
                price=order_result.price,
                status=order_result.status
            )
            
            return order_result
            
        except Exception as e:
            logger.error(
                "Failed to place limit order",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            
            logger.info("Canceling order", order_id=order_id, symbol=normalized_symbol)
            
            # Cancel order
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.cancel_order(order_id, normalized_symbol)
            )
            
            # Update order tracking
            if order_id in self._active_orders:
                self._active_orders[order_id].status = 'canceled'
                del self._active_orders[order_id]
            
            logger.info("Order canceled successfully", order_id=order_id)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cancel order",
                order_id=order_id,
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[OrderResult]:
        """
        Get the status of an order.
        
        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            
        Returns:
            OrderResult if found, None otherwise
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Fetch order from exchange
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_order(order_id, normalized_symbol)
            )
            
            # Convert to OrderResult
            order_result = self._convert_order_to_result(order)
            
            # Update tracking
            if order_id in self._active_orders:
                self._active_orders[order_id] = order_result
                
                # Remove from active if closed/canceled
                if order_result.status in ['closed', 'canceled']:
                    del self._active_orders[order_id]
            
            return order_result
            
        except Exception as e:
            logger.error(
                "Failed to get order status",
                order_id=order_id,
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def get_balance(self) -> Optional[Dict[str, BalanceInfo]]:
        """
        Get account balance information.
        
        Returns:
            Dictionary of currency -> BalanceInfo, None if failed
        """
        try:
            # Fetch balance
            balance = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_balance
            )
            
            # Convert to BalanceInfo objects
            result = {}
            for currency, info in balance.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    result[currency] = BalanceInfo(
                        currency=currency,
                        free=float(info.get('free', 0)),
                        used=float(info.get('used', 0)),
                        total=float(info.get('total', 0))
                    )
            
            return result
            
        except Exception as e:
            logger.error("Failed to get balance", error=str(e), exc_info=True)
            return None
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker information dict, None if failed
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_ticker(normalized_symbol)
            )
            
            return ticker
            
        except Exception as e:
            logger.error("Failed to get ticker", symbol=symbol, error=str(e), exc_info=True)
            return None
    
    def _convert_order_to_result(self, order: Dict[str, Any]) -> OrderResult:
        """Convert CCXT order dict to OrderResult."""
        return OrderResult(
            order_id=str(order['id']),
            symbol=order['symbol'],
            side=order['side'],
            amount=float(order['amount']),
            price=float(order['price']) if order['price'] else None,
            order_type=order['type'],
            status=order['status'],
            filled=float(order['filled']),
            remaining=float(order['remaining']),
            cost=float(order['cost']),
            fee=order.get('fee'),
            timestamp=datetime.fromtimestamp(order['timestamp'] / 1000) if order['timestamp'] else datetime.utcnow(),
            info=order.get('info', {})
        )
    
    def get_active_orders(self) -> Dict[str, OrderResult]:
        """Get currently active orders."""
        return self._active_orders.copy()
    
    def get_order_history(self) -> List[OrderResult]:
        """Get order history."""
        return self._order_history.copy()
    
    async def close(self) -> None:
        """Close the exchange connection."""
        try:
            if hasattr(self.exchange, 'close'):
                await asyncio.get_event_loop().run_in_executor(None, self.exchange.close)
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error("Error closing exchange connection", error=str(e))


class ExchangeManager:
    """
    High-level manager for exchange operations across multiple trading pairs.
    """
    
    def __init__(self, exchange: BinanceTestnetExchange):
        """Initialize with an exchange instance."""
        self.exchange = exchange
        self.positions: Dict[str, PositionInfo] = {}
        self.last_position_update: Optional[datetime] = None
        
    async def initialize(self) -> bool:
        """Initialize the exchange manager."""
        success = await self.exchange.initialize()
        if success:
            await self.update_positions()
        return success
    
    async def execute_pair_trade(
        self,
        pair_id: str,
        asset1_symbol: str,
        asset2_symbol: str,
        side: str,  # 'long' or 'short'
        hedge_ratio: float,
        notional_amount: float,
        order_type: str = 'market'
    ) -> Tuple[Optional[OrderResult], Optional[OrderResult]]:
        """
        Execute a pairs trade (long one asset, short the other).
        
        Args:
            pair_id: Trading pair identifier
            asset1_symbol: First asset symbol
            asset2_symbol: Second asset symbol
            side: 'long' (buy asset1, sell asset2) or 'short' (sell asset1, buy asset2)
            hedge_ratio: Hedge ratio between assets
            notional_amount: Total notional amount to trade
            order_type: 'market' or 'limit'
            
        Returns:
            Tuple of (asset1_order, asset2_order)
        """
        try:
            logger.info(
                "Executing pairs trade",
                pair_id=pair_id,
                asset1=asset1_symbol,
                asset2=asset2_symbol,
                side=side,
                hedge_ratio=hedge_ratio,
                notional_amount=notional_amount
            )
            
            # Get current prices for position sizing
            ticker1 = await self.exchange.get_ticker(asset1_symbol)
            ticker2 = await self.exchange.get_ticker(asset2_symbol)
            
            if not ticker1 or not ticker2:
                logger.error("Failed to get tickers for pair trade")
                return None, None
            
            price1 = ticker1['last']
            price2 = ticker2['last']
            
            # Calculate position sizes
            if side == 'long':
                # Long asset1, short asset2
                asset1_side = 'buy'
                asset2_side = 'sell'
                asset1_amount = notional_amount / price1
                asset2_amount = (notional_amount * hedge_ratio) / price2
            else:
                # Short asset1, long asset2
                asset1_side = 'sell'
                asset2_side = 'buy'
                asset1_amount = notional_amount / price1
                asset2_amount = (notional_amount * hedge_ratio) / price2
            
            # Place orders
            if order_type == 'market':
                order1 = await self.exchange.place_market_order(
                    asset1_symbol, asset1_side, asset1_amount,
                    client_order_id=f"{pair_id}_{asset1_symbol}_{int(time.time())}"
                )
                order2 = await self.exchange.place_market_order(
                    asset2_symbol, asset2_side, asset2_amount,
                    client_order_id=f"{pair_id}_{asset2_symbol}_{int(time.time())}"
                )
            else:
                # For limit orders, use current prices (could be improved with better pricing logic)
                order1 = await self.exchange.place_limit_order(
                    asset1_symbol, asset1_side, asset1_amount, price1,
                    client_order_id=f"{pair_id}_{asset1_symbol}_{int(time.time())}"
                )
                order2 = await self.exchange.place_limit_order(
                    asset2_symbol, asset2_side, asset2_amount, price2,
                    client_order_id=f"{pair_id}_{asset2_symbol}_{int(time.time())}"
                )
            
            logger.info(
                "Pairs trade executed",
                pair_id=pair_id,
                order1_id=order1.order_id if order1 else None,
                order2_id=order2.order_id if order2 else None
            )
            
            return order1, order2
            
        except Exception as e:
            logger.error(
                "Failed to execute pairs trade",
                pair_id=pair_id,
                error=str(e),
                exc_info=True
            )
            return None, None
    
    async def update_positions(self) -> None:
        """Update position information."""
        try:
            balance = await self.exchange.get_balance()
            if not balance:
                return
            
            # Update positions based on non-zero balances
            self.positions.clear()
            
            for currency, balance_info in balance.items():
                if balance_info.total != 0:
                    # Get current price for PnL calculation
                    if currency != 'USDT':  # Skip USDT as it's the quote currency
                        symbol = f"{currency}/USDT"
                        ticker = await self.exchange.get_ticker(symbol)
                        current_price = ticker['last'] if ticker else 0
                        
                        position = PositionInfo(
                            symbol=symbol,
                            side='long' if balance_info.total > 0 else 'short',
                            size=abs(balance_info.total),
                            entry_price=None,  # Would need to track from orders
                            current_price=current_price,
                            unrealized_pnl=0,  # Would need entry price to calculate
                            percentage=0,
                            timestamp=datetime.utcnow()
                        )
                        
                        self.positions[symbol] = position
            
            self.last_position_update = datetime.utcnow()
            
        except Exception as e:
            logger.error("Failed to update positions", error=str(e), exc_info=True)
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a specific symbol."""
        normalized_symbol = self.exchange._normalize_symbol(symbol)
        return self.positions.get(normalized_symbol)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """Get all current positions."""
        return self.positions.copy()
