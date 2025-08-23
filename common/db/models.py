"""SQLAlchemy models for trades, orders, and portfolio tracking."""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any
import json

Base = declarative_base()


class Order(Base):
    """Model for individual orders placed on the exchange."""
    
    __tablename__ = 'orders'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Order identification
    order_id = Column(String(100), nullable=False, unique=True, index=True)
    client_order_id = Column(String(100), nullable=True, index=True)
    
    # Trading pair and exchange info
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20), nullable=False, default='binance_testnet')
    
    # Order details
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    order_type = Column(String(20), nullable=False)  # 'market', 'limit', etc.
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=True)  # None for market orders
    
    # Order status and execution
    status = Column(String(20), nullable=False, index=True)  # 'open', 'filled', 'cancelled', etc.
    filled = Column(Float, nullable=False, default=0.0)
    remaining = Column(Float, nullable=False, default=0.0)
    average_price = Column(Float, nullable=True)
    
    # Financial details
    cost = Column(Float, nullable=False, default=0.0)  # Total cost in quote currency
    fee = Column(Float, nullable=True)
    fee_currency = Column(String(10), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    exchange_timestamp = Column(DateTime, nullable=True)
    
    # Strategy context
    strategy_id = Column(String(50), nullable=True, index=True)
    pair_id = Column(String(20), nullable=True, index=True)  # Trading pair like 'BTCETH'
    
    # Additional metadata
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationships
    trade_orders = relationship("TradeOrder", back_populates="order")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='positive_amount'),
        CheckConstraint('filled >= 0', name='non_negative_filled'),
        CheckConstraint('remaining >= 0', name='non_negative_remaining'),
        CheckConstraint('cost >= 0', name='non_negative_cost'),
        Index('idx_orders_symbol_status', 'symbol', 'status'),
        Index('idx_orders_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Order(id={self.id}, order_id='{self.order_id}', symbol='{self.symbol}', side='{self.side}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'id': self.id,
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side,
            'order_type': self.order_type,
            'amount': self.amount,
            'price': self.price,
            'status': self.status,
            'filled': self.filled,
            'remaining': self.remaining,
            'average_price': self.average_price,
            'cost': self.cost,
            'fee': self.fee,
            'fee_currency': self.fee_currency,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'exchange_timestamp': self.exchange_timestamp.isoformat() if self.exchange_timestamp else None,
            'strategy_id': self.strategy_id,
            'pair_id': self.pair_id,
            'metadata': json.loads(self.extra_data) if self.extra_data else None
        }


class Trade(Base):
    """Model for completed trades (pairs of orders)."""
    
    __tablename__ = 'trades'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Trade identification
    trade_id = Column(String(100), nullable=False, unique=True, index=True)
    
    # Trading pair info
    pair_id = Column(String(20), nullable=False, index=True)  # e.g., 'BTCETH'
    asset1_symbol = Column(String(10), nullable=False)  # e.g., 'BTC'
    asset2_symbol = Column(String(10), nullable=False)  # e.g., 'ETH'
    
    # Trade direction and strategy
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    strategy_id = Column(String(50), nullable=False, index=True)
    
    # Entry details
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price_asset1 = Column(Float, nullable=False)
    entry_price_asset2 = Column(Float, nullable=False)
    entry_spread = Column(Float, nullable=False)
    entry_z_score = Column(Float, nullable=True)
    hedge_ratio = Column(Float, nullable=False)
    
    # Position sizing
    quantity_asset1 = Column(Float, nullable=False)
    quantity_asset2 = Column(Float, nullable=False)
    notional_value = Column(Float, nullable=False)  # Total position value in USD
    
    # Exit details (nullable until trade is closed)
    exit_time = Column(DateTime, nullable=True, index=True)
    exit_price_asset1 = Column(Float, nullable=True)
    exit_price_asset2 = Column(Float, nullable=True)
    exit_spread = Column(Float, nullable=True)
    exit_z_score = Column(Float, nullable=True)
    
    # P&L and performance
    realized_pnl = Column(Float, nullable=True)  # Final P&L when closed
    unrealized_pnl = Column(Float, nullable=False, default=0.0)  # Current P&L
    max_favorable_excursion = Column(Float, nullable=False, default=0.0)
    max_adverse_excursion = Column(Float, nullable=False, default=0.0)
    
    # Trade status and metadata
    status = Column(String(20), nullable=False, default='open', index=True)  # 'open', 'closed', 'error'
    close_reason = Column(String(50), nullable=True)  # 'profit_target', 'stop_loss', 'timeout', etc.
    
    # Fees and costs
    total_fees = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    
    # Additional metadata
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationships
    trade_orders = relationship("TradeOrder", back_populates="trade")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('quantity_asset1 > 0', name='positive_quantity_asset1'),
        CheckConstraint('quantity_asset2 > 0', name='positive_quantity_asset2'),
        CheckConstraint('notional_value > 0', name='positive_notional_value'),
        CheckConstraint('hedge_ratio > 0', name='positive_hedge_ratio'),
        Index('idx_trades_pair_status', 'pair_id', 'status'),
        Index('idx_trades_entry_time', 'entry_time'),
        Index('idx_trades_strategy', 'strategy_id'),
    )
    
    def __repr__(self):
        return f"<Trade(id={self.id}, trade_id='{self.trade_id}', pair_id='{self.pair_id}', side='{self.side}', status='{self.status}')>"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate trade duration in seconds."""
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.status == 'open'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'pair_id': self.pair_id,
            'asset1_symbol': self.asset1_symbol,
            'asset2_symbol': self.asset2_symbol,
            'side': self.side,
            'strategy_id': self.strategy_id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_price_asset1': self.entry_price_asset1,
            'entry_price_asset2': self.entry_price_asset2,
            'entry_spread': self.entry_spread,
            'entry_z_score': self.entry_z_score,
            'hedge_ratio': self.hedge_ratio,
            'quantity_asset1': self.quantity_asset1,
            'quantity_asset2': self.quantity_asset2,
            'notional_value': self.notional_value,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price_asset1': self.exit_price_asset1,
            'exit_price_asset2': self.exit_price_asset2,
            'exit_spread': self.exit_spread,
            'exit_z_score': self.exit_z_score,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'status': self.status,
            'close_reason': self.close_reason,
            'total_fees': self.total_fees,
            'duration_seconds': self.duration_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': json.loads(self.extra_data) if self.extra_data else None
        }


class TradeOrder(Base):
    """Association table linking trades to their constituent orders."""
    
    __tablename__ = 'trade_orders'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False, index=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False, index=True)
    
    # Order role in the trade
    order_role = Column(String(20), nullable=False)  # 'entry_asset1', 'entry_asset2', 'exit_asset1', 'exit_asset2'
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    trade = relationship("Trade", back_populates="trade_orders")
    order = relationship("Order", back_populates="trade_orders")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('trade_id', 'order_id', name='unique_trade_order'),
        Index('idx_trade_orders_trade', 'trade_id'),
        Index('idx_trade_orders_order', 'order_id'),
    )
    
    def __repr__(self):
        return f"<TradeOrder(trade_id={self.trade_id}, order_id={self.order_id}, role='{self.order_role}')>"


class PortfolioPnL(Base):
    """Time series of portfolio P&L for performance tracking."""
    
    __tablename__ = 'portfolio_pnl'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Portfolio values
    total_value = Column(Float, nullable=False)  # Total portfolio value in USD
    cash_balance = Column(Float, nullable=False)  # Available cash
    unrealized_pnl = Column(Float, nullable=False, default=0.0)  # Open positions P&L
    realized_pnl = Column(Float, nullable=False, default=0.0)  # Closed trades P&L
    
    # Position details
    open_positions = Column(Integer, nullable=False, default=0)  # Number of open positions
    daily_pnl = Column(Float, nullable=True)  # P&L for the day
    
    # Risk metrics
    drawdown = Column(Float, nullable=True)  # Current drawdown from peak
    max_drawdown = Column(Float, nullable=True)  # Maximum drawdown to date
    
    # Strategy breakdown (JSON)
    strategy_pnl = Column(Text, nullable=True)  # JSON with P&L by strategy
    pair_pnl = Column(Text, nullable=True)  # JSON with P&L by trading pair
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('total_value >= 0', name='non_negative_total_value'),
        CheckConstraint('cash_balance >= 0', name='non_negative_cash_balance'),
        CheckConstraint('open_positions >= 0', name='non_negative_open_positions'),
        Index('idx_portfolio_pnl_timestamp', 'timestamp'),
        Index('idx_portfolio_pnl_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<PortfolioPnL(timestamp={self.timestamp}, total_value={self.total_value}, unrealized_pnl={self.unrealized_pnl})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio PnL to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'open_positions': self.open_positions,
            'daily_pnl': self.daily_pnl,
            'drawdown': self.drawdown,
            'max_drawdown': self.max_drawdown,
            'strategy_pnl': json.loads(self.strategy_pnl) if self.strategy_pnl else None,
            'pair_pnl': json.loads(self.pair_pnl) if self.pair_pnl else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SystemEvent(Base):
    """Log of system events for debugging and monitoring."""
    
    __tablename__ = 'system_events'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # 'service_start', 'error', 'signal', etc.
    service_name = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)  # 'info', 'warning', 'error', 'critical'
    
    # Event content
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON string with additional details
    
    # Context
    pair_id = Column(String(20), nullable=True, index=True)
    trade_id = Column(String(100), nullable=True, index=True)
    order_id = Column(String(100), nullable=True, index=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=func.now(), index=True)
    
    # Constraints
    __table_args__ = (
        Index('idx_system_events_type_timestamp', 'event_type', 'timestamp'),
        Index('idx_system_events_service_timestamp', 'service_name', 'timestamp'),
        Index('idx_system_events_severity', 'severity'),
    )
    
    def __repr__(self):
        return f"<SystemEvent(id={self.id}, type='{self.event_type}', service='{self.service_name}', severity='{self.severity}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system event to dictionary."""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'service_name': self.service_name,
            'severity': self.severity,
            'message': self.message,
            'details': json.loads(self.details) if self.details else None,
            'pair_id': self.pair_id,
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
