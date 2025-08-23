"""Data loader for historical market data in backtesting engine."""

import pandas as pd
import numpy as np
from typing import Iterator, List, Optional, Dict, Union, Tuple
from datetime import datetime, timedelta
import json
import gzip
import pickle
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

from .models import (
    OrderBook, Trade, PriceLevel, OrderSide, MarketDataEvent,
    BacktestEvent, EventType
)
from common.logger import get_logger

logger = get_logger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def load_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Iterator[BacktestEvent]:
        """Load data for the specified symbol and time range."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @abstractmethod
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available data range for symbol."""
        pass


class CSVDataSource(DataSource):
    """CSV data source for historical data."""
    
    def __init__(self, data_directory: str):
        """
        Initialize CSV data source.
        
        Args:
            data_directory: Directory containing CSV files
        """
        self.data_directory = Path(data_directory)
        self._symbol_cache = {}
        self._scan_data_files()
    
    def _scan_data_files(self) -> None:
        """Scan data directory for available files."""
        logger.info("Scanning data directory", directory=str(self.data_directory))
        
        if not self.data_directory.exists():
            logger.warning("Data directory does not exist", directory=str(self.data_directory))
            return
        
        for file_path in self.data_directory.glob("*.csv"):
            symbol = file_path.stem.upper()
            self._symbol_cache[symbol] = file_path
            logger.debug("Found data file", symbol=symbol, file=str(file_path))
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self._symbol_cache.keys())
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available data range for symbol."""
        if symbol not in self._symbol_cache:
            raise ValueError(f"Symbol {symbol} not found in data directory")
        
        file_path = self._symbol_cache[symbol]
        
        # Read first and last rows to get date range
        df_first = pd.read_csv(file_path, nrows=1)
        df_last = pd.read_csv(file_path).tail(1)
        
        start_time = pd.to_datetime(df_first.iloc[0]['timestamp'])
        end_time = pd.to_datetime(df_last.iloc[0]['timestamp'])
        
        return start_time, end_time
    
    def load_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Iterator[BacktestEvent]:
        """Load data for the specified symbol and time range."""
        if symbol not in self._symbol_cache:
            raise ValueError(f"Symbol {symbol} not found in data directory")
        
        file_path = self._symbol_cache[symbol]
        logger.info("Loading data", symbol=symbol, file=str(file_path))
        
        # Read CSV in chunks for memory efficiency
        chunk_size = 10000
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            
            # Filter by time range
            mask = (chunk['timestamp'] >= start_time) & (chunk['timestamp'] <= end_time)
            filtered_chunk = chunk[mask]
            
            for _, row in filtered_chunk.iterrows():
                event = self._parse_csv_row(symbol, row)
                if event:
                    yield event
    
    def _parse_csv_row(self, symbol: str, row: pd.Series) -> Optional[BacktestEvent]:
        """Parse CSV row into BacktestEvent."""
        try:
            timestamp = row['timestamp']
            
            # Check if this is order book data or trade data
            if 'bids' in row and 'asks' in row:
                # Order book data
                bids = self._parse_price_levels(row['bids'])
                asks = self._parse_price_levels(row['asks'])
                
                order_book = OrderBook(
                    symbol=symbol,
                    timestamp=timestamp,
                    bids=bids,
                    asks=asks,
                    sequence_id=row.get('sequence_id', 0)
                )
                
                return MarketDataEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    order_book=order_book
                )
            
            elif 'price' in row and 'quantity' in row:
                # Trade data
                side = OrderSide.BUY if row.get('side', 'buy').lower() == 'buy' else OrderSide.SELL
                
                trade = Trade(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=float(row['price']),
                    quantity=float(row['quantity']),
                    side=side,
                    trade_id=row.get('trade_id')
                )
                
                return MarketDataEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    trade=trade
                )
            
            else:
                logger.warning("Unknown CSV row format", row=row.to_dict())
                return None
                
        except Exception as e:
            logger.error("Error parsing CSV row", error=str(e), row=row.to_dict())
            return None
    
    def _parse_price_levels(self, price_levels_str: str) -> List[PriceLevel]:
        """Parse price levels from string representation."""
        try:
            # Expect format: "[[price1, qty1], [price2, qty2], ...]"
            levels_data = json.loads(price_levels_str)
            return [PriceLevel(price=float(level[0]), quantity=float(level[1])) 
                   for level in levels_data]
        except Exception as e:
            logger.error("Error parsing price levels", error=str(e), data=price_levels_str)
            return []


class ParquetDataSource(DataSource):
    """Parquet data source for historical data (more efficient for large datasets)."""
    
    def __init__(self, data_directory: str):
        """
        Initialize Parquet data source.
        
        Args:
            data_directory: Directory containing Parquet files
        """
        self.data_directory = Path(data_directory)
        self._symbol_cache = {}
        self._scan_data_files()
    
    def _scan_data_files(self) -> None:
        """Scan data directory for available files."""
        logger.info("Scanning Parquet data directory", directory=str(self.data_directory))
        
        if not self.data_directory.exists():
            logger.warning("Data directory does not exist", directory=str(self.data_directory))
            return
        
        for file_path in self.data_directory.glob("*.parquet"):
            symbol = file_path.stem.upper()
            self._symbol_cache[symbol] = file_path
            logger.debug("Found Parquet file", symbol=symbol, file=str(file_path))
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self._symbol_cache.keys())
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available data range for symbol."""
        if symbol not in self._symbol_cache:
            raise ValueError(f"Symbol {symbol} not found in data directory")
        
        file_path = self._symbol_cache[symbol]
        
        # Read metadata to get date range efficiently
        df = pd.read_parquet(file_path, columns=['timestamp'])
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        
        return start_time, end_time
    
    def load_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Iterator[BacktestEvent]:
        """Load data for the specified symbol and time range."""
        if symbol not in self._symbol_cache:
            raise ValueError(f"Symbol {symbol} not found in data directory")
        
        file_path = self._symbol_cache[symbol]
        logger.info("Loading Parquet data", symbol=symbol, file=str(file_path))
        
        # Read Parquet file with time filtering
        df = pd.read_parquet(
            file_path,
            filters=[
                ('timestamp', '>=', start_time),
                ('timestamp', '<=', end_time)
            ]
        )
        
        for _, row in df.iterrows():
            event = self._parse_parquet_row(symbol, row)
            if event:
                yield event
    
    def _parse_parquet_row(self, symbol: str, row: pd.Series) -> Optional[BacktestEvent]:
        """Parse Parquet row into BacktestEvent."""
        # Similar to CSV parsing but optimized for Parquet format
        return self._parse_csv_row(symbol, row)  # Reuse CSV parsing logic


class KafkaRecordingDataSource(DataSource):
    """Data source for Kafka recorded data."""
    
    def __init__(self, recording_directory: str):
        """
        Initialize Kafka recording data source.
        
        Args:
            recording_directory: Directory containing Kafka recordings
        """
        self.recording_directory = Path(recording_directory)
        self._recording_cache = {}
        self._scan_recordings()
    
    def _scan_recordings(self) -> None:
        """Scan recording directory for available recordings."""
        logger.info("Scanning Kafka recordings", directory=str(self.recording_directory))
        
        if not self.recording_directory.exists():
            logger.warning("Recording directory does not exist", directory=str(self.recording_directory))
            return
        
        for recording_file in self.recording_directory.glob("*.json.gz"):
            # Extract metadata from filename
            parts = recording_file.stem.replace('.json', '').split('_')
            if len(parts) >= 3:
                symbol = parts[0].upper()
                start_date = parts[1]
                end_date = parts[2]
                
                if symbol not in self._recording_cache:
                    self._recording_cache[symbol] = []
                
                self._recording_cache[symbol].append({
                    'file': recording_file,
                    'start_date': start_date,
                    'end_date': end_date
                })
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self._recording_cache.keys())
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available data range for symbol."""
        if symbol not in self._recording_cache:
            raise ValueError(f"Symbol {symbol} not found in recordings")
        
        recordings = self._recording_cache[symbol]
        start_dates = [r['start_date'] for r in recordings]
        end_dates = [r['end_date'] for r in recordings]
        
        start_time = pd.to_datetime(min(start_dates))
        end_time = pd.to_datetime(max(end_dates))
        
        return start_time, end_time
    
    def load_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Iterator[BacktestEvent]:
        """Load data for the specified symbol and time range."""
        if symbol not in self._recording_cache:
            raise ValueError(f"Symbol {symbol} not found in recordings")
        
        recordings = self._recording_cache[symbol]
        
        for recording in recordings:
            recording_start = pd.to_datetime(recording['start_date'])
            recording_end = pd.to_datetime(recording['end_date'])
            
            # Check if recording overlaps with requested range
            if recording_end < start_time or recording_start > end_time:
                continue
            
            logger.info("Loading Kafka recording", 
                       symbol=symbol, 
                       file=str(recording['file']))
            
            with gzip.open(recording['file'], 'rt') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        event = self._parse_kafka_message(symbol, data)
                        
                        if event and start_time <= event.timestamp <= end_time:
                            yield event
                            
                    except Exception as e:
                        logger.error("Error parsing Kafka message", error=str(e))
    
    def _parse_kafka_message(self, symbol: str, data: Dict) -> Optional[BacktestEvent]:
        """Parse Kafka message into BacktestEvent."""
        try:
            timestamp = pd.to_datetime(data['timestamp'])
            message_type = data.get('type', 'unknown')
            
            if message_type == 'order_book_update':
                bids = [PriceLevel(price=float(bid[0]), quantity=float(bid[1])) 
                       for bid in data.get('bids', [])]
                asks = [PriceLevel(price=float(ask[0]), quantity=float(ask[1])) 
                       for ask in data.get('asks', [])]
                
                order_book = OrderBook(
                    symbol=symbol,
                    timestamp=timestamp,
                    bids=bids,
                    asks=asks,
                    sequence_id=data.get('final_update_id', 0)
                )
                
                return MarketDataEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    order_book=order_book
                )
            
            elif message_type == 'trade_tick':
                trade = Trade(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=float(data['price']),
                    quantity=float(data['quantity']),
                    side=OrderSide.BUY  # Default for synthetic ticks
                )
                
                return MarketDataEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    trade=trade
                )
            
            else:
                logger.debug("Unknown Kafka message type", type=message_type)
                return None
                
        except Exception as e:
            logger.error("Error parsing Kafka message", error=str(e), data=data)
            return None


class SyntheticDataSource(DataSource):
    """Synthetic data source for testing."""
    
    def __init__(
        self,
        symbols: List[str],
        start_price: float = 50000.0,
        volatility: float = 0.02,
        tick_frequency_ms: int = 100
    ):
        """
        Initialize synthetic data source.
        
        Args:
            symbols: List of symbols to generate data for
            start_price: Starting price for synthetic data
            volatility: Daily volatility
            tick_frequency_ms: Frequency of ticks in milliseconds
        """
        self.symbols = symbols
        self.start_price = start_price
        self.volatility = volatility
        self.tick_frequency = timedelta(milliseconds=tick_frequency_ms)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.symbols
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available data range for symbol."""
        # Synthetic data can generate any range
        return datetime(2020, 1, 1), datetime(2024, 12, 31)
    
    def load_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> Iterator[BacktestEvent]:
        """Load synthetic data for the specified symbol and time range."""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not available in synthetic data")
        
        logger.info("Generating synthetic data", 
                   symbol=symbol, 
                   start=start_time, 
                   end=end_time)
        
        current_time = start_time
        current_price = self.start_price
        
        # Calculate tick volatility
        ticks_per_day = 24 * 60 * 60 * 1000 / self.tick_frequency.total_seconds() / 1000
        tick_volatility = self.volatility / np.sqrt(ticks_per_day)
        
        while current_time <= end_time:
            # Generate price movement
            price_change = np.random.normal(0, tick_volatility * current_price)
            current_price = max(current_price + price_change, 0.01)  # Prevent negative prices
            
            # Generate order book
            spread_bps = np.random.uniform(1, 10)  # 1-10 bps spread
            spread = current_price * spread_bps / 10000
            
            best_bid = current_price - spread / 2
            best_ask = current_price + spread / 2
            
            # Generate multiple levels
            bids = []
            asks = []
            
            for i in range(5):  # 5 levels each side
                bid_price = best_bid - i * spread * 0.1
                ask_price = best_ask + i * spread * 0.1
                
                bid_qty = np.random.uniform(0.1, 10.0)
                ask_qty = np.random.uniform(0.1, 10.0)
                
                bids.append(PriceLevel(price=bid_price, quantity=bid_qty))
                asks.append(PriceLevel(price=ask_price, quantity=ask_qty))
            
            order_book = OrderBook(
                symbol=symbol,
                timestamp=current_time,
                bids=bids,
                asks=asks
            )
            
            yield MarketDataEvent(
                timestamp=current_time,
                symbol=symbol,
                order_book=order_book
            )
            
            current_time += self.tick_frequency


class DataLoader:
    """Main data loader class that manages multiple data sources."""
    
    def __init__(self):
        """Initialize data loader."""
        self.data_sources: Dict[str, DataSource] = {}
        self.default_source: Optional[str] = None
    
    def add_data_source(self, name: str, data_source: DataSource, is_default: bool = False) -> None:
        """Add a data source."""
        self.data_sources[name] = data_source
        
        if is_default or not self.default_source:
            self.default_source = name
        
        logger.info("Added data source", 
                   name=name, 
                   type=type(data_source).__name__,
                   is_default=is_default)
    
    def get_available_symbols(self, source_name: Optional[str] = None) -> List[str]:
        """Get available symbols from a data source."""
        source_name = source_name or self.default_source
        
        if source_name not in self.data_sources:
            raise ValueError(f"Data source {source_name} not found")
        
        return self.data_sources[source_name].get_available_symbols()
    
    def get_data_range(self, symbol: str, source_name: Optional[str] = None) -> Tuple[datetime, datetime]:
        """Get data range for a symbol."""
        source_name = source_name or self.default_source
        
        if source_name not in self.data_sources:
            raise ValueError(f"Data source {source_name} not found")
        
        return self.data_sources[source_name].get_data_range(symbol)
    
    def load_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        source_name: Optional[str] = None
    ) -> Iterator[BacktestEvent]:
        """Load data for backtesting."""
        source_name = source_name or self.default_source
        
        if source_name not in self.data_sources:
            raise ValueError(f"Data source {source_name} not found")
        
        logger.info("Loading data for backtesting",
                   symbol=symbol,
                   start=start_time,
                   end=end_time,
                   source=source_name)
        
        return self.data_sources[source_name].load_data(symbol, start_time, end_time)
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        source_name: Optional[str] = None
    ) -> Iterator[BacktestEvent]:
        """Load data for multiple symbols and merge by timestamp."""
        # Load data from all symbols
        all_events = []
        
        for symbol in symbols:
            events = list(self.load_data(symbol, start_time, end_time, source_name))
            all_events.extend(events)
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x.timestamp)
        
        logger.info("Loaded multi-symbol data",
                   symbols=symbols,
                   total_events=len(all_events),
                   start=start_time,
                   end=end_time)
        
        for event in all_events:
            yield event
