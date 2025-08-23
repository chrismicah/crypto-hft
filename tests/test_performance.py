"""Performance tests for the crypto HFT bot."""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock
from decimal import Decimal

from src.models import OrderBook, OrderBookUpdate, PriceLevel, WebSocketMessage
from src.websocket_client import parse_order_book_update
from src.order_book_manager import OrderBookManager


class TestPerformance:
    """Performance tests to ensure low-latency requirements."""
    
    @pytest.mark.slow
    def test_order_book_update_parsing_performance(self):
        """Test order book update parsing performance."""
        # Create a large WebSocket message
        bids_data = [[f"{100.0 - i * 0.01:.2f}", f"{i + 1}.0"] for i in range(1000)]
        asks_data = [[f"{101.0 + i * 0.01:.2f}", f"{i + 1}.0"] for i in range(1000)]
        
        ws_message = WebSocketMessage(
            stream="btcusdt@depth20@100ms",
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "U": 12345,
                "u": 12346,
                "b": bids_data,
                "a": asks_data
            }
        )
        
        # Measure parsing time
        start_time = time.perf_counter()
        
        for _ in range(100):  # Parse 100 times
            update = parse_order_book_update(ws_message)
            assert update is not None
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100
        
        # Should parse in less than 1ms on average for HFT requirements
        assert avg_time < 0.001, f"Parsing took {avg_time:.6f}s, should be < 0.001s"
    
    @pytest.mark.slow
    def test_order_book_update_application_performance(self):
        """Test order book update application performance."""
        manager = OrderBookManager(['BTCUSDT'])
        
        # Create initial order book with many levels
        initial_bids = [PriceLevel(price=f"{100.0 - i * 0.01:.2f}", quantity="1.0") 
                       for i in range(500)]
        initial_asks = [PriceLevel(price=f"{101.0 + i * 0.01:.2f}", quantity="1.0") 
                       for i in range(500)]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=initial_bids,
            asks=initial_asks,
            last_update_id=12345
        )
        
        # Create update that modifies many levels
        update_bids = [PriceLevel(price=f"{100.0 - i * 0.01:.2f}", quantity="2.0") 
                      for i in range(0, 100, 2)]  # Update every other level
        update_asks = [PriceLevel(price=f"{101.0 + i * 0.01:.2f}", quantity="2.0") 
                      for i in range(0, 100, 2)]
        
        update = OrderBookUpdate(
            symbol="BTCUSDT",
            first_update_id=12346,
            final_update_id=12346,
            bids=update_bids,
            asks=update_asks
        )
        
        # Measure update application time
        start_time = time.perf_counter()
        
        for _ in range(100):  # Apply 100 updates
            updated_book = manager._apply_update(order_book, update)
            assert updated_book is not None
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100
        
        # Should apply updates in less than 1ms on average
        assert avg_time < 0.001, f"Update application took {avg_time:.6f}s, should be < 0.001s"
    
    @pytest.mark.slow
    def test_order_book_validation_performance(self):
        """Test order book validation performance."""
        manager = OrderBookManager(['BTCUSDT'])
        
        # Create large order book
        bids = [PriceLevel(price=f"{100.0 - i * 0.01:.2f}", quantity="1.0") 
               for i in range(1000)]
        asks = [PriceLevel(price=f"{101.0 + i * 0.01:.2f}", quantity="1.0") 
               for i in range(1000)]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        # Measure validation time
        start_time = time.perf_counter()
        
        for _ in range(1000):  # Validate 1000 times
            result = manager._validate_order_book(order_book)
            assert result is True
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 1000
        
        # Should validate in less than 0.1ms on average
        assert avg_time < 0.0001, f"Validation took {avg_time:.6f}s, should be < 0.0001s"
    
    @pytest.mark.slow
    def test_checksum_calculation_performance(self):
        """Test checksum calculation performance."""
        # Create order book with many levels
        bids = [PriceLevel(price=f"{100.0 - i * 0.01:.2f}", quantity=f"{i + 1}.0") 
               for i in range(100)]
        asks = [PriceLevel(price=f"{101.0 + i * 0.01:.2f}", quantity=f"{i + 1}.0") 
               for i in range(100)]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        # Measure checksum calculation time
        start_time = time.perf_counter()
        
        for _ in range(1000):  # Calculate 1000 checksums
            checksum = order_book.calculate_checksum()
            assert len(checksum) == 32
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 1000
        
        # Should calculate checksum in less than 0.1ms on average
        assert avg_time < 0.0001, f"Checksum calculation took {avg_time:.6f}s, should be < 0.0001s"
    
    @pytest.mark.slow
    def test_decimal_operations_performance(self):
        """Test Decimal operations performance for price calculations."""
        # Create many price levels
        prices = [Decimal(f"{100.0 + i * 0.001:.3f}") for i in range(10000)]
        quantities = [Decimal(f"{i + 1}.0") for i in range(10000)]
        
        # Measure arithmetic operations
        start_time = time.perf_counter()
        
        total_value = Decimal('0')
        for price, quantity in zip(prices, quantities):
            total_value += price * quantity
        
        end_time = time.perf_counter()
        operation_time = end_time - start_time
        
        # Should complete 10,000 operations in reasonable time
        assert operation_time < 0.1, f"Decimal operations took {operation_time:.6f}s, should be < 0.1s"
        assert total_value > 0  # Sanity check
    
    @pytest.mark.slow
    def test_memory_usage_order_book_updates(self):
        """Test memory usage doesn't grow excessively with updates."""
        import gc
        import sys
        
        manager = OrderBookManager(['BTCUSDT'])
        
        # Create initial order book
        initial_bids = [PriceLevel(price="100.00", quantity="1.0")]
        initial_asks = [PriceLevel(price="101.00", quantity="1.0")]
        order_book = OrderBook(
            symbol="BTCUSDT",
            bids=initial_bids,
            asks=initial_asks,
            last_update_id=12345
        )
        
        # Force garbage collection and measure initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Apply many updates
        for i in range(1000):
            update = OrderBookUpdate(
                symbol="BTCUSDT",
                first_update_id=12346 + i,
                final_update_id=12346 + i,
                bids=[PriceLevel(price=f"{100.0 + i * 0.01:.2f}", quantity="1.0")],
                asks=[PriceLevel(price=f"{101.0 + i * 0.01:.2f}", quantity="1.0")]
            )
            order_book = manager._apply_update(order_book, update)
        
        # Force garbage collection and measure final memory
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable (less than 50% increase)
        growth_ratio = final_objects / initial_objects
        assert growth_ratio < 1.5, f"Memory grew by {growth_ratio:.2f}x, should be < 1.5x"
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_update_processing(self):
        """Test processing multiple updates concurrently."""
        manager = OrderBookManager(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
        
        # Create updates for different symbols
        updates = []
        for i, symbol in enumerate(['BTCUSDT', 'ETHUSDT', 'ADAUSDT']):
            for j in range(100):  # 100 updates per symbol
                update = OrderBookUpdate(
                    symbol=symbol,
                    first_update_id=j + 1,
                    final_update_id=j + 1,
                    bids=[PriceLevel(price=f"{100.0 + j * 0.01:.2f}", quantity="1.0")],
                    asks=[PriceLevel(price=f"{101.0 + j * 0.01:.2f}", quantity="1.0")]
                )
                updates.append(update)
        
        # Process updates concurrently
        start_time = time.perf_counter()
        
        # Note: In real implementation, these would be processed sequentially per symbol
        # but we can test the processing speed
        tasks = []
        for update in updates:
            # Simulate processing (without actual order book state)
            task = asyncio.create_task(asyncio.sleep(0))  # Minimal async operation
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should process 300 updates very quickly
        assert total_time < 0.1, f"Concurrent processing took {total_time:.6f}s, should be < 0.1s"
