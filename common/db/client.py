"""SQLAlchemy database client with session management."""

import os
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import structlog

from .models import Base, Order, Trade, TradeOrder, PortfolioPnL, SystemEvent

logger = structlog.get_logger(__name__)


class DatabaseClient:
    """Database client for managing trading data persistence."""
    
    def __init__(
        self,
        database_url: str = "sqlite:///data/trading.db",
        echo: bool = False,
        pool_pre_ping: bool = True
    ):
        """
        Initialize database client.
        
        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to echo SQL statements
            pool_pre_ping: Whether to enable connection pool pre-ping
        """
        self.database_url = database_url
        
        # Ensure data directory exists for SQLite
        if database_url.startswith('sqlite:///'):
            db_path = database_url.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create engine
        self.engine = create_engine(
            database_url,
            echo=echo,
            pool_pre_ping=pool_pre_ping,
            # SQLite specific settings
            connect_args={'check_same_thread': False} if 'sqlite' in database_url else {}
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database client initialized", database_url=database_url)
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e), exc_info=True)
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables (use with caution)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e), exc_info=True)
            raise
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Database session error", error=str(e), exc_info=True)
            raise
        finally:
            session.close()
    
    def write_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        status: str,
        price: Optional[float] = None,
        filled: float = 0.0,
        remaining: Optional[float] = None,
        cost: float = 0.0,
        fee: Optional[float] = None,
        fee_currency: Optional[str] = None,
        client_order_id: Optional[str] = None,
        exchange: str = 'binance_testnet',
        exchange_timestamp: Optional[datetime] = None,
        strategy_id: Optional[str] = None,
        pair_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """
        Write order to database.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            amount: Order amount
            status: Order status
            price: Order price (optional for market orders)
            filled: Filled amount
            remaining: Remaining amount
            cost: Total cost
            fee: Trading fee
            fee_currency: Fee currency
            client_order_id: Client order ID
            exchange: Exchange name
            exchange_timestamp: Exchange timestamp
            strategy_id: Strategy identifier
            pair_id: Trading pair identifier
            metadata: Additional metadata
            
        Returns:
            Created Order object or None if failed
        """
        try:
            with self.get_session() as session:
                # Check if order already exists
                existing_order = session.query(Order).filter_by(order_id=order_id).first()
                
                if existing_order:
                    # Update existing order
                    existing_order.status = status
                    existing_order.filled = filled
                    existing_order.remaining = remaining if remaining is not None else amount - filled
                    existing_order.cost = cost
                    existing_order.fee = fee
                    existing_order.fee_currency = fee_currency
                    existing_order.updated_at = datetime.utcnow()
                    
                    if metadata:
                        existing_order.extra_data = json.dumps(metadata)
                    
                    logger.debug("Order updated", order_id=order_id, status=status)
                    return existing_order
                
                else:
                    # Create new order
                    order = Order(
                        order_id=order_id,
                        client_order_id=client_order_id,
                        symbol=symbol,
                        exchange=exchange,
                        side=side,
                        order_type=order_type,
                        amount=amount,
                        price=price,
                        status=status,
                        filled=filled,
                        remaining=remaining if remaining is not None else amount - filled,
                        cost=cost,
                        fee=fee,
                        fee_currency=fee_currency,
                        exchange_timestamp=exchange_timestamp,
                        strategy_id=strategy_id,
                        pair_id=pair_id,
                        extra_data=json.dumps(metadata) if metadata else None
                    )
                    
                    session.add(order)
                    session.flush()  # Get the ID
                    
                    logger.info(
                        "Order written to database",
                        order_id=order_id,
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        status=status
                    )
                    
                    return order
                    
        except Exception as e:
            logger.error(
                "Failed to write order to database",
                order_id=order_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def write_trade(
        self,
        trade_id: str,
        pair_id: str,
        asset1_symbol: str,
        asset2_symbol: str,
        side: str,
        strategy_id: str,
        entry_time: datetime,
        entry_price_asset1: float,
        entry_price_asset2: float,
        entry_spread: float,
        hedge_ratio: float,
        quantity_asset1: float,
        quantity_asset2: float,
        notional_value: float,
        entry_z_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Trade]:
        """
        Write trade to database.
        
        Args:
            trade_id: Unique trade identifier
            pair_id: Trading pair identifier
            asset1_symbol: First asset symbol
            asset2_symbol: Second asset symbol
            side: Trade side ('long' or 'short')
            strategy_id: Strategy identifier
            entry_time: Trade entry time
            entry_price_asset1: Entry price for asset 1
            entry_price_asset2: Entry price for asset 2
            entry_spread: Entry spread value
            hedge_ratio: Hedge ratio used
            quantity_asset1: Quantity of asset 1
            quantity_asset2: Quantity of asset 2
            notional_value: Total notional value
            entry_z_score: Entry z-score
            metadata: Additional metadata
            
        Returns:
            Created Trade object or None if failed
        """
        try:
            with self.get_session() as session:
                trade = Trade(
                    trade_id=trade_id,
                    pair_id=pair_id,
                    asset1_symbol=asset1_symbol,
                    asset2_symbol=asset2_symbol,
                    side=side,
                    strategy_id=strategy_id,
                    entry_time=entry_time,
                    entry_price_asset1=entry_price_asset1,
                    entry_price_asset2=entry_price_asset2,
                    entry_spread=entry_spread,
                    entry_z_score=entry_z_score,
                    hedge_ratio=hedge_ratio,
                    quantity_asset1=quantity_asset1,
                    quantity_asset2=quantity_asset2,
                    notional_value=notional_value,
                    status='open',
                    metadata=json.dumps(metadata) if metadata else None
                )
                
                session.add(trade)
                session.flush()  # Get the ID
                
                logger.info(
                    "Trade written to database",
                    trade_id=trade_id,
                    pair_id=pair_id,
                    side=side,
                    notional_value=notional_value
                )
                
                return trade
                
        except Exception as e:
            logger.error(
                "Failed to write trade to database",
                trade_id=trade_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def close_trade(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price_asset1: float,
        exit_price_asset2: float,
        exit_spread: float,
        realized_pnl: float,
        close_reason: str = 'manual',
        exit_z_score: Optional[float] = None,
        total_fees: float = 0.0
    ) -> Optional[Trade]:
        """
        Close an existing trade.
        
        Args:
            trade_id: Trade identifier
            exit_time: Trade exit time
            exit_price_asset1: Exit price for asset 1
            exit_price_asset2: Exit price for asset 2
            exit_spread: Exit spread value
            realized_pnl: Realized P&L
            close_reason: Reason for closing
            exit_z_score: Exit z-score
            total_fees: Total fees paid
            
        Returns:
            Updated Trade object or None if failed
        """
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter_by(trade_id=trade_id).first()
                
                if not trade:
                    logger.error("Trade not found for closing", trade_id=trade_id)
                    return None
                
                # Update trade with exit information
                trade.exit_time = exit_time
                trade.exit_price_asset1 = exit_price_asset1
                trade.exit_price_asset2 = exit_price_asset2
                trade.exit_spread = exit_spread
                trade.exit_z_score = exit_z_score
                trade.realized_pnl = realized_pnl
                trade.status = 'closed'
                trade.close_reason = close_reason
                trade.total_fees = total_fees
                trade.updated_at = datetime.utcnow()
                
                logger.info(
                    "Trade closed in database",
                    trade_id=trade_id,
                    realized_pnl=realized_pnl,
                    close_reason=close_reason
                )
                
                return trade
                
        except Exception as e:
            logger.error(
                "Failed to close trade in database",
                trade_id=trade_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def update_trade_pnl(
        self,
        trade_id: str,
        unrealized_pnl: float,
        max_favorable_excursion: Optional[float] = None,
        max_adverse_excursion: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Update trade P&L metrics.
        
        Args:
            trade_id: Trade identifier
            unrealized_pnl: Current unrealized P&L
            max_favorable_excursion: Maximum favorable excursion
            max_adverse_excursion: Maximum adverse excursion
            
        Returns:
            Updated Trade object or None if failed
        """
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter_by(trade_id=trade_id).first()
                
                if not trade:
                    logger.error("Trade not found for P&L update", trade_id=trade_id)
                    return None
                
                trade.unrealized_pnl = unrealized_pnl
                
                if max_favorable_excursion is not None:
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion, max_favorable_excursion
                    )
                
                if max_adverse_excursion is not None:
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion, max_adverse_excursion
                    )
                
                trade.updated_at = datetime.utcnow()
                
                return trade
                
        except Exception as e:
            logger.error(
                "Failed to update trade P&L",
                trade_id=trade_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def link_trade_order(
        self,
        trade_id: str,
        order_id: str,
        order_role: str
    ) -> Optional[TradeOrder]:
        """
        Link an order to a trade.
        
        Args:
            trade_id: Trade identifier
            order_id: Order identifier
            order_role: Role of order in trade
            
        Returns:
            Created TradeOrder object or None if failed
        """
        try:
            with self.get_session() as session:
                # Get trade and order objects
                trade = session.query(Trade).filter_by(trade_id=trade_id).first()
                order = session.query(Order).filter_by(order_id=order_id).first()
                
                if not trade or not order:
                    logger.error(
                        "Trade or order not found for linking",
                        trade_id=trade_id,
                        order_id=order_id
                    )
                    return None
                
                # Create link
                trade_order = TradeOrder(
                    trade_id=trade.id,
                    order_id=order.id,
                    order_role=order_role
                )
                
                session.add(trade_order)
                
                logger.debug(
                    "Trade-order link created",
                    trade_id=trade_id,
                    order_id=order_id,
                    role=order_role
                )
                
                return trade_order
                
        except Exception as e:
            logger.error(
                "Failed to link trade and order",
                trade_id=trade_id,
                order_id=order_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def write_portfolio_pnl(
        self,
        timestamp: datetime,
        total_value: float,
        cash_balance: float,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        open_positions: int = 0,
        daily_pnl: Optional[float] = None,
        drawdown: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        strategy_pnl: Optional[Dict[str, float]] = None,
        pair_pnl: Optional[Dict[str, float]] = None
    ) -> Optional[PortfolioPnL]:
        """
        Write portfolio P&L snapshot to database.
        
        Args:
            timestamp: Snapshot timestamp
            total_value: Total portfolio value
            cash_balance: Available cash balance
            unrealized_pnl: Unrealized P&L from open positions
            realized_pnl: Realized P&L from closed trades
            open_positions: Number of open positions
            daily_pnl: P&L for the day
            drawdown: Current drawdown
            max_drawdown: Maximum drawdown
            strategy_pnl: P&L breakdown by strategy
            pair_pnl: P&L breakdown by trading pair
            
        Returns:
            Created PortfolioPnL object or None if failed
        """
        try:
            with self.get_session() as session:
                portfolio_pnl = PortfolioPnL(
                    timestamp=timestamp,
                    total_value=total_value,
                    cash_balance=cash_balance,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl,
                    open_positions=open_positions,
                    daily_pnl=daily_pnl,
                    drawdown=drawdown,
                    max_drawdown=max_drawdown,
                    strategy_pnl=json.dumps(strategy_pnl) if strategy_pnl else None,
                    pair_pnl=json.dumps(pair_pnl) if pair_pnl else None
                )
                
                session.add(portfolio_pnl)
                
                logger.debug(
                    "Portfolio P&L written to database",
                    timestamp=timestamp.isoformat(),
                    total_value=total_value,
                    unrealized_pnl=unrealized_pnl
                )
                
                return portfolio_pnl
                
        except Exception as e:
            logger.error(
                "Failed to write portfolio P&L",
                error=str(e),
                exc_info=True
            )
            return None
    
    def write_system_event(
        self,
        event_type: str,
        service_name: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        pair_id: Optional[str] = None,
        trade_id: Optional[str] = None,
        order_id: Optional[str] = None
    ) -> Optional[SystemEvent]:
        """
        Write system event to database.
        
        Args:
            event_type: Type of event
            service_name: Name of service generating event
            severity: Event severity level
            message: Event message
            details: Additional event details
            pair_id: Related trading pair
            trade_id: Related trade ID
            order_id: Related order ID
            
        Returns:
            Created SystemEvent object or None if failed
        """
        try:
            with self.get_session() as session:
                event = SystemEvent(
                    event_type=event_type,
                    service_name=service_name,
                    severity=severity,
                    message=message,
                    details=json.dumps(details) if details else None,
                    pair_id=pair_id,
                    trade_id=trade_id,
                    order_id=order_id
                )
                
                session.add(event)
                
                return event
                
        except Exception as e:
            logger.error(
                "Failed to write system event",
                event_type=event_type,
                error=str(e),
                exc_info=True
            )
            return None
    
    def get_open_trades(self, pair_id: Optional[str] = None) -> List[Trade]:
        """Get all open trades, optionally filtered by pair."""
        try:
            with self.get_session() as session:
                query = session.query(Trade).filter_by(status='open')
                
                if pair_id:
                    query = query.filter_by(pair_id=pair_id)
                
                return query.all()
                
        except Exception as e:
            logger.error("Failed to get open trades", error=str(e), exc_info=True)
            return []
    
    def get_recent_portfolio_pnl(self, hours: int = 24) -> List[PortfolioPnL]:
        """Get recent portfolio P&L data."""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                return session.query(PortfolioPnL)\
                    .filter(PortfolioPnL.timestamp >= cutoff_time)\
                    .order_by(PortfolioPnL.timestamp.desc())\
                    .all()
                    
        except Exception as e:
            logger.error("Failed to get recent portfolio P&L", error=str(e), exc_info=True)
            return []
    
    def get_trade_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get trade performance summary for the last N days."""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                
                # Get closed trades in the period
                closed_trades = session.query(Trade)\
                    .filter(Trade.status == 'closed')\
                    .filter(Trade.exit_time >= cutoff_time)\
                    .all()
                
                if not closed_trades:
                    return {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'average_pnl': 0.0,
                        'max_win': 0.0,
                        'max_loss': 0.0
                    }
                
                # Calculate metrics
                total_trades = len(closed_trades)
                winning_trades = sum(1 for t in closed_trades if t.realized_pnl > 0)
                losing_trades = total_trades - winning_trades
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                
                pnls = [t.realized_pnl for t in closed_trades if t.realized_pnl is not None]
                total_pnl = sum(pnls)
                average_pnl = total_pnl / len(pnls) if pnls else 0.0
                max_win = max(pnls) if pnls else 0.0
                max_loss = min(pnls) if pnls else 0.0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'average_pnl': average_pnl,
                    'max_win': max_win,
                    'max_loss': max_loss
                }
                
        except Exception as e:
            logger.error("Failed to get trade performance summary", error=str(e), exc_info=True)
            return {}
    
    def cleanup_old_data(self, days: int = 90) -> None:
        """Clean up old data to prevent database bloat."""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                
                # Clean up old system events
                deleted_events = session.query(SystemEvent)\
                    .filter(SystemEvent.timestamp < cutoff_time)\
                    .delete()
                
                # Clean up old portfolio P&L (keep daily snapshots)
                deleted_pnl = session.query(PortfolioPnL)\
                    .filter(PortfolioPnL.timestamp < cutoff_time)\
                    .delete()
                
                logger.info(
                    "Database cleanup completed",
                    deleted_events=deleted_events,
                    deleted_pnl_records=deleted_pnl,
                    cutoff_days=days
                )
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e), exc_info=True)
