"""End-to-end system tests for execution service."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

from services.execution.main import ExecutionService, DataAggregator
from services.execution.exchange import BinanceTestnetExchange, OrderResult
from services.execution.signals import MarketData, TradingSignal, SignalType, SignalStrength
from services.execution.state_machine import TradingState, PositionSide


class MockKafkaMessage:
    """Mock Kafka message for testing."""
    
    def __init__(self, value: Dict[str, Any]):
        self.value = value


class TestDataAggregator:
    """Test cases for DataAggregator class."""
    
    def test_initialization(self):
        """Test data aggregator initialization."""
        aggregator = DataAggregator()
        
        assert len(aggregator.order_book_data) == 0
        assert len(aggregator.hedge_ratios) == 0
        assert len(aggregator.thresholds) == 0
        assert aggregator.data_ttl == timedelta(seconds=30)
    
    def test_update_data_methods(self):
        """Test data update methods."""
        aggregator = DataAggregator()
        
        # Update order book
        order_book_data = {
            'symbol': 'BTCUSDT',
            'bids': [[49000.0, 1.5], [48900.0, 2.0]],
            'asks': [[49100.0, 1.2], [49200.0, 1.8]]
        }
        aggregator.update_order_book('BTCETH', order_book_data)
        
        assert 'BTCETH' in aggregator.order_book_data
        assert aggregator.order_book_data['BTCETH'] == order_book_data
        assert 'BTCETH' in aggregator.last_update_times['order_book']
        
        # Update hedge ratio
        hedge_ratio_data = {
            'pair_id': 'BTCETH',
            'hedge_ratio': 15.2,
            'confidence_interval_95': 0.85
        }
        aggregator.update_hedge_ratio('BTCETH', hedge_ratio_data)
        
        assert 'BTCETH' in aggregator.hedge_ratios
        assert aggregator.hedge_ratios['BTCETH'] == hedge_ratio_data
        
        # Update thresholds
        threshold_data = {
            'pair_id': 'BTCETH',
            'volatility_forecast': 0.12,
            'entry_threshold_long': -2.0,
            'entry_threshold_short': 2.0,
            'exit_threshold': 0.5,
            'volatility_regime': 'normal'
        }
        aggregator.update_thresholds('BTCETH', threshold_data)
        
        assert 'BTCETH' in aggregator.thresholds
        assert aggregator.thresholds['BTCETH'] == threshold_data
    
    def test_get_market_data_success(self):
        """Test successful market data aggregation."""
        aggregator = DataAggregator()
        
        # Add fresh data
        aggregator.update_order_book('BTCETH', {
            'symbol': 'BTCUSDT',
            'bids': [[49000.0, 1.5]],
            'asks': [[49100.0, 1.2]]
        })
        
        aggregator.update_hedge_ratio('BTCETH', {
            'hedge_ratio': 15.2,
            'confidence_interval_95': 0.85
        })
        
        aggregator.update_thresholds('BTCETH', {
            'volatility_forecast': 0.12,
            'entry_threshold_long': -2.0,
            'entry_threshold_short': 2.0,
            'exit_threshold': 0.5,
            'volatility_regime': 'normal'
        })
        
        # Get aggregated data
        market_data = aggregator.get_market_data('BTCETH')
        
        assert market_data is not None
        assert isinstance(market_data, MarketData)
        assert market_data.hedge_ratio == 15.2
        assert market_data.volatility_forecast == 0.12
        assert market_data.entry_threshold_long == -2.0
        assert market_data.volatility_regime == 'normal'
    
    def test_get_market_data_stale_data(self):
        """Test market data aggregation with stale data."""
        aggregator = DataAggregator(data_ttl=timedelta(seconds=1))
        
        # Add data
        aggregator.update_order_book('BTCETH', {
            'bids': [[49000.0, 1.5]],
            'asks': [[49100.0, 1.2]]
        })
        
        # Make data stale
        aggregator.last_update_times['order_book']['BTCETH'] = (
            datetime.utcnow() - timedelta(seconds=2)
        )
        
        # Should return None for stale data
        market_data = aggregator.get_market_data('BTCETH')
        assert market_data is None
    
    def test_get_data_status(self):
        """Test data status reporting."""
        aggregator = DataAggregator()
        
        # Add some data
        aggregator.update_order_book('BTCETH', {'test': 'data'})
        aggregator.update_hedge_ratio('BTCADA', {'test': 'data'})
        
        status = aggregator.get_data_status()
        
        assert 'BTCETH' in status
        assert 'BTCADA' in status
        
        # Check BTCETH status
        btceth_status = status['BTCETH']
        assert 'order_book' in btceth_status
        assert 'hedge_ratios' in btceth_status
        assert 'thresholds' in btceth_status
        
        assert btceth_status['order_book']['is_fresh'] is True
        assert btceth_status['hedge_ratios']['is_fresh'] is False  # No data
        assert btceth_status['thresholds']['is_fresh'] is False   # No data


class TestExecutionServiceIntegration:
    """Integration tests for ExecutionService."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange for testing."""
        exchange = Mock(spec=BinanceTestnetExchange)
        exchange.initialize = AsyncMock(return_value=True)
        exchange.get_balance = AsyncMock(return_value={
            'USDT': Mock(free=10000.0, used=0.0, total=10000.0)
        })
        exchange.close = AsyncMock()
        return exchange
    
    @pytest.fixture
    def mock_exchange_manager(self, mock_exchange):
        """Mock exchange manager for testing."""
        manager = Mock()
        manager.initialize = AsyncMock(return_value=True)
        manager.update_positions = AsyncMock()
        manager.execute_pair_trade = AsyncMock(return_value=(
            OrderResult(
                order_id='order1',
                symbol='BTCUSDT',
                side='buy',
                amount=0.1,
                price=49000.0,
                order_type='market',
                status='filled',
                filled=0.1,
                remaining=0.0,
                cost=4900.0,
                fee=None,
                timestamp=datetime.utcnow(),
                info={}
            ),
            OrderResult(
                order_id='order2',
                symbol='ETHUSDT',
                side='sell',
                amount=1.5,
                price=3000.0,
                order_type='market',
                status='filled',
                filled=1.5,
                remaining=0.0,
                cost=4500.0,
                fee=None,
                timestamp=datetime.utcnow(),
                info={}
            )
        ))
        return manager
    
    @patch('services.execution.main.KafkaConsumer')
    async def test_service_initialization(self, mock_kafka_consumer, mock_exchange):
        """Test service initialization."""
        # Mock Kafka consumer
        mock_consumer_instance = Mock()
        mock_kafka_consumer.return_value = mock_consumer_instance
        
        service = ExecutionService()
        
        # Mock exchange initialization
        with patch.object(service, 'exchange', mock_exchange):
            with patch('services.execution.main.ExchangeManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager.initialize = AsyncMock(return_value=True)
                mock_manager_class.return_value = mock_manager
                
                await service._initialize_components()
        
        # Check that state machines were created
        assert len(service.state_machines) > 0
        assert 'BTCETH' in service.state_machines
        assert 'BTCADA' in service.state_machines
        
        # Check initial states
        for sm in service.state_machines.values():
            assert sm.get_current_state() == TradingState.SEARCHING
    
    async def test_complete_trading_workflow(self, mock_exchange, mock_exchange_manager):
        """Test complete trading workflow from signal to execution."""
        service = ExecutionService()
        service.exchange = mock_exchange
        service.exchange_manager = mock_exchange_manager
        
        # Initialize state machine
        service.state_machines['BTCETH'] = Mock()
        service.state_machines['BTCETH'].get_current_state.return_value = TradingState.ENTERING
        service.state_machines['BTCETH'].has_position.return_value = True
        service.state_machines['BTCETH'].current_position = Mock()
        service.state_machines['BTCETH'].current_position.asset1_order_id = None
        service.state_machines['BTCETH'].current_position.asset2_order_id = None
        service.state_machines['BTCETH'].position_entered = Mock()
        service.state_machines['BTCETH'].entry_failed = Mock()
        service.state_machines['BTCETH'].order_sizer = Mock()
        service.state_machines['BTCETH'].order_sizer.calculate_position_size.return_value = 1000.0
        
        # Create entry signal
        entry_signal = TradingSignal(
            pair_id='BTCETH',
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long',
            hedge_ratio=15.0,
            volatility_forecast=0.12
        )
        
        # Test entry order placement
        pair_config = {
            'pair_id': 'BTCETH',
            'asset1_symbol': 'BTCUSDT',
            'asset2_symbol': 'ETHUSDT',
            'enabled': True
        }
        
        success = await service._place_entry_orders(
            'BTCETH',
            pair_config,
            service.state_machines['BTCETH'],
            entry_signal
        )
        
        assert success is True
        mock_exchange_manager.execute_pair_trade.assert_called_once()
        
        # Check that order info was stored
        assert service.stats['orders_placed'] == 2
    
    def test_kafka_message_processing(self):
        """Test Kafka message processing."""
        service = ExecutionService()
        
        # Test order book message processing
        order_book_message = MockKafkaMessage({
            'symbol': 'BTCUSDT',
            'bids': [[49000.0, 1.5], [48900.0, 2.0]],
            'asks': [[49100.0, 1.2], [49200.0, 1.8]],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Mock the consumer to avoid actual Kafka
        service.consumers['order_book'] = [order_book_message]
        service.running = False  # Stop after one message
        
        # Process message (simplified)
        data = order_book_message.value
        symbol = data.get('symbol', '')
        
        if symbol == 'BTCUSDT':
            service.data_aggregator.update_order_book('BTCETH', data)
        
        # Check that data was aggregated
        assert 'BTCETH' in service.data_aggregator.order_book_data
        assert service.data_aggregator.order_book_data['BTCETH']['symbol'] == 'BTCUSDT'
    
    async def test_signal_processing_and_state_transitions(self):
        """Test signal processing and state machine transitions."""
        service = ExecutionService()
        
        # Mock components
        service.exchange = Mock()
        service.exchange_manager = Mock()
        service.exchange_manager.execute_pair_trade = AsyncMock(return_value=(Mock(), Mock()))
        
        # Create state machine
        from services.execution.state_machine import StrategyStateMachine
        state_machine = StrategyStateMachine('BTCETH')
        service.state_machines['BTCETH'] = state_machine
        
        # Create market data that should trigger entry
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5,
            volatility_forecast=0.12
        )
        
        # Mock data aggregator to return this data
        service.data_aggregator.get_market_data = Mock(return_value=market_data)
        
        # Mock signal generator to return entry signal
        with patch.object(service.signal_generator, 'generate_signal') as mock_generate:
            entry_signal = TradingSignal(
                pair_id='BTCETH',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                confidence=0.8,
                side='long',
                hedge_ratio=15.0
            )
            mock_generate.return_value = entry_signal
            
            # Process trading pair
            pair_config = {
                'pair_id': 'BTCETH',
                'asset1_symbol': 'BTCUSDT',
                'asset2_symbol': 'ETHUSDT',
                'enabled': True
            }
            
            await service._process_trading_pair('BTCETH', pair_config)
        
        # Check that signal was processed and state changed
        assert state_machine.get_current_state() == TradingState.ENTERING
        assert state_machine.has_position()
        assert state_machine.current_position.side == PositionSide.LONG
    
    def test_health_status_reporting(self):
        """Test health status reporting."""
        service = ExecutionService()
        service.running = True
        service.stats['start_time'] = datetime.utcnow() - timedelta(minutes=5)
        service.stats['messages_processed'] = 100
        service.stats['orders_placed'] = 10
        
        # Add mock state machine
        service.state_machines['BTCETH'] = Mock()
        service.state_machines['BTCETH'].get_state_info.return_value = {
            'pair_id': 'BTCETH',
            'current_state': 'searching',
            'time_in_state_seconds': 300
        }
        
        # Mock exchange
        service.exchange = Mock()
        
        health_status = service.get_health_status()
        
        required_keys = [
            'status', 'uptime_seconds', 'stats', 'state_machines',
            'exchange_connected', 'data_status', 'active_consumers'
        ]
        
        for key in required_keys:
            assert key in health_status
        
        assert health_status['status'] == 'healthy'
        assert health_status['uptime_seconds'] > 0
        assert health_status['exchange_connected'] is True
        assert 'BTCETH' in health_status['state_machines']


class TestSyntheticDataScenarios:
    """Test execution service with synthetic market data scenarios."""
    
    def create_synthetic_order_book_update(
        self,
        symbol: str,
        base_price: float,
        spread_bps: int = 10
    ) -> Dict[str, Any]:
        """Create synthetic order book update."""
        spread = base_price * (spread_bps / 10000.0)
        
        return {
            'symbol': symbol,
            'bids': [
                [base_price - spread/2, 1.5],
                [base_price - spread, 2.0],
                [base_price - spread*1.5, 1.0]
            ],
            'asks': [
                [base_price + spread/2, 1.2],
                [base_price + spread, 1.8],
                [base_price + spread*1.5, 0.8]
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def create_synthetic_hedge_ratio_signal(
        self,
        pair_id: str,
        hedge_ratio: float,
        confidence: float = 0.85
    ) -> Dict[str, Any]:
        """Create synthetic hedge ratio signal."""
        return {
            'pair_id': pair_id,
            'hedge_ratio': hedge_ratio,
            'confidence_interval_95': confidence,
            'kalman_gain': 0.1,
            'prediction_error': 0.05,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def create_synthetic_threshold_signal(
        self,
        pair_id: str,
        volatility_forecast: float,
        regime: str = 'normal'
    ) -> Dict[str, Any]:
        """Create synthetic threshold signal."""
        # Adjust thresholds based on volatility
        base_threshold = 2.0
        vol_multiplier = 1.0 + (volatility_forecast - 0.1) * 2.0
        
        return {
            'pair_id': pair_id,
            'volatility_forecast': volatility_forecast,
            'entry_threshold_long': -base_threshold * vol_multiplier,
            'entry_threshold_short': base_threshold * vol_multiplier,
            'exit_threshold': 0.5 * vol_multiplier,
            'volatility_regime': regime,
            'z_score': 0.0,  # Will be calculated by signal generator
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def test_mean_reversion_scenario(self):
        """Test mean reversion trading scenario."""
        service = ExecutionService()
        
        # Mock exchange components
        service.exchange = Mock()
        service.exchange_manager = Mock()
        service.exchange_manager.execute_pair_trade = AsyncMock(return_value=(Mock(), Mock()))
        
        # Initialize state machine
        from services.execution.state_machine import StrategyStateMachine
        state_machine = StrategyStateMachine('BTCETH')
        service.state_machines['BTCETH'] = state_machine
        
        # Scenario: BTC/ETH spread widens, then reverts
        scenarios = [
            # 1. Normal spread
            {
                'btc_price': 50000.0,
                'eth_price': 3000.0,
                'hedge_ratio': 16.67,  # 50000/3000
                'volatility': 0.08,
                'expected_signal': SignalType.HOLD
            },
            # 2. Spread widens (BTC outperforms)
            {
                'btc_price': 52000.0,
                'eth_price': 3000.0,
                'hedge_ratio': 16.67,
                'volatility': 0.12,
                'expected_signal': SignalType.ENTRY_SHORT  # Short BTC, Long ETH
            },
            # 3. Spread reverts
            {
                'btc_price': 50500.0,
                'eth_price': 3030.0,
                'hedge_ratio': 16.67,
                'volatility': 0.10,
                'expected_signal': SignalType.EXIT
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            # Create synthetic data
            order_book_btc = self.create_synthetic_order_book_update(
                'BTCUSDT', scenario['btc_price']
            )
            order_book_eth = self.create_synthetic_order_book_update(
                'ETHUSDT', scenario['eth_price']
            )
            hedge_ratio_signal = self.create_synthetic_hedge_ratio_signal(
                'BTCETH', scenario['hedge_ratio']
            )
            threshold_signal = self.create_synthetic_threshold_signal(
                'BTCETH', scenario['volatility']
            )
            
            # Update data aggregator
            service.data_aggregator.update_order_book('BTCETH', order_book_btc)
            service.data_aggregator.update_hedge_ratio('BTCETH', hedge_ratio_signal)
            service.data_aggregator.update_thresholds('BTCETH', threshold_signal)
            
            # Get market data
            market_data = service.data_aggregator.get_market_data('BTCETH')
            assert market_data is not None
            
            # Generate signal
            current_position = None
            if state_machine.has_position():
                position_side = state_machine.get_position_side()
                current_position = position_side.value if position_side else None
            
            signal = service.signal_generator.generate_signal(
                'BTCETH', market_data, current_position
            )
            
            # Process signal
            state_machine.process_signal(signal)
            
            # Simulate order execution for entry/exit signals
            if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                state_machine.position_entered()
            elif signal.signal_type in [SignalType.EXIT, SignalType.STOP_LOSS]:
                state_machine.position_exited()
            
            print(f"Scenario {i+1}: {signal.signal_type.value} (expected: {scenario['expected_signal'].value})")
    
    async def test_high_volatility_scenario(self):
        """Test trading during high volatility periods."""
        service = ExecutionService()
        
        # Mock components
        service.exchange = Mock()
        service.exchange_manager = Mock()
        
        # Initialize state machine
        from services.execution.state_machine import StrategyStateMachine
        state_machine = StrategyStateMachine('BTCETH')
        service.state_machines['BTCETH'] = state_machine
        
        # High volatility scenario - thresholds should widen
        high_vol_data = {
            'order_book': self.create_synthetic_order_book_update('BTCUSDT', 48000.0),
            'hedge_ratio': self.create_synthetic_hedge_ratio_signal('BTCETH', 16.0, 0.7),
            'thresholds': self.create_synthetic_threshold_signal('BTCETH', 0.25, 'high')
        }
        
        # Update aggregator
        service.data_aggregator.update_order_book('BTCETH', high_vol_data['order_book'])
        service.data_aggregator.update_hedge_ratio('BTCETH', high_vol_data['hedge_ratio'])
        service.data_aggregator.update_thresholds('BTCETH', high_vol_data['thresholds'])
        
        # Get market data
        market_data = service.data_aggregator.get_market_data('BTCETH')
        assert market_data is not None
        assert market_data.volatility_regime == 'high'
        
        # Generate signal
        signal = service.signal_generator.generate_signal('BTCETH', market_data)
        
        # In high volatility, confidence should be reduced
        if signal.signal_type != SignalType.HOLD:
            assert signal.confidence < 0.9  # Should be reduced due to high volatility
        
        # Thresholds should be wider
        assert abs(market_data.entry_threshold_long) > 2.0
        assert market_data.entry_threshold_short > 2.0
    
    async def test_multiple_pairs_coordination(self):
        """Test coordination across multiple trading pairs."""
        service = ExecutionService()
        
        # Mock components
        service.exchange = Mock()
        service.exchange_manager = Mock()
        
        # Initialize state machines for multiple pairs
        pairs = ['BTCETH', 'BTCADA']
        for pair_id in pairs:
            from services.execution.state_machine import StrategyStateMachine
            service.state_machines[pair_id] = StrategyStateMachine(pair_id)
        
        # Create different scenarios for each pair
        pair_scenarios = {
            'BTCETH': {
                'btc_price': 50000.0,
                'alt_price': 3000.0,
                'hedge_ratio': 16.67,
                'volatility': 0.10
            },
            'BTCADA': {
                'btc_price': 50000.0,
                'alt_price': 0.5,
                'hedge_ratio': 100000.0,
                'volatility': 0.15
            }
        }
        
        signals = {}
        
        for pair_id, scenario in pair_scenarios.items():
            # Create synthetic data
            order_book = self.create_synthetic_order_book_update(
                'BTCUSDT', scenario['btc_price']
            )
            hedge_ratio_signal = self.create_synthetic_hedge_ratio_signal(
                pair_id, scenario['hedge_ratio']
            )
            threshold_signal = self.create_synthetic_threshold_signal(
                pair_id, scenario['volatility']
            )
            
            # Update data
            service.data_aggregator.update_order_book(pair_id, order_book)
            service.data_aggregator.update_hedge_ratio(pair_id, hedge_ratio_signal)
            service.data_aggregator.update_thresholds(pair_id, threshold_signal)
            
            # Generate signal
            market_data = service.data_aggregator.get_market_data(pair_id)
            if market_data:
                signal = service.signal_generator.generate_signal(pair_id, market_data)
                signals[pair_id] = signal
        
        # Check that each pair generated independent signals
        assert len(signals) == 2
        assert 'BTCETH' in signals
        assert 'BTCADA' in signals
        
        # Signals should be independent
        for pair_id, signal in signals.items():
            assert signal.pair_id == pair_id
            assert isinstance(signal.signal_type, SignalType)


@pytest.fixture
def sample_execution_service():
    """Fixture providing a sample execution service."""
    service = ExecutionService()
    service.running = False  # Prevent actual startup
    return service


class TestExecutionServiceE2E:
    """End-to-end tests for execution service."""
    
    @patch('services.execution.main.KafkaConsumer')
    async def test_full_service_lifecycle(self, mock_kafka_consumer, sample_execution_service):
        """Test full service lifecycle from startup to shutdown."""
        service = sample_execution_service
        
        # Mock Kafka consumers
        mock_consumer = Mock()
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.initialize = AsyncMock(return_value=True)
        mock_exchange.get_balance = AsyncMock(return_value={
            'USDT': Mock(free=10000.0)
        })
        mock_exchange.close = AsyncMock()
        
        with patch('services.execution.main.BinanceTestnetExchange', return_value=mock_exchange):
            with patch('services.execution.main.ExchangeManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager.initialize = AsyncMock(return_value=True)
                mock_manager_class.return_value = mock_manager
                
                # Initialize components
                await service._initialize_components()
        
        # Check initialization
        assert service.exchange == mock_exchange
        assert len(service.state_machines) > 0
        
        # Test health status
        health = service.get_health_status()
        assert 'status' in health
        assert 'stats' in health
        assert 'state_machines' in health
        
        # Cleanup
        await service._cleanup()
        mock_exchange.close.assert_called_once()
    
    async def test_error_handling_and_recovery(self, sample_execution_service):
        """Test error handling and recovery mechanisms."""
        service = sample_execution_service
        
        # Initialize state machine
        from services.execution.state_machine import StrategyStateMachine
        state_machine = StrategyStateMachine('BTCETH')
        service.state_machines['BTCETH'] = state_machine
        
        # Simulate error in signal processing
        with patch.object(service.signal_generator, 'generate_signal', side_effect=Exception("Test error")):
            market_data = MarketData(
                asset1_bid=49000.0,
                asset1_ask=49100.0,
                hedge_ratio=15.0
            )
            
            # Should handle error gracefully
            pair_config = {'pair_id': 'BTCETH', 'enabled': True}
            await service._process_trading_pair('BTCETH', pair_config)
            
            # Error count should increase
            assert service.stats['errors'] > 0
            
            # State machine should still be functional
            assert state_machine.get_current_state() == TradingState.SEARCHING
    
    def test_performance_metrics_tracking(self, sample_execution_service):
        """Test performance metrics tracking."""
        service = sample_execution_service
        
        # Initialize stats
        service.stats['start_time'] = datetime.utcnow() - timedelta(minutes=10)
        service.stats['messages_processed'] = 1000
        service.stats['signals_generated'] = 800
        service.stats['orders_placed'] = 50
        service.stats['positions_opened'] = 25
        service.stats['positions_closed'] = 20
        service.stats['errors'] = 5
        
        # Log stats (should not raise exception)
        service._log_stats()
        
        # Check health status includes performance metrics
        health = service.get_health_status()
        
        assert health['stats']['messages_processed'] == 1000
        assert health['stats']['signals_generated'] == 800
        assert health['stats']['orders_placed'] == 50
        assert health['uptime_seconds'] > 0
