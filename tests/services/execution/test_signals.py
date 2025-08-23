"""Unit tests for signal generation logic."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.execution.signals import (
    SignalGenerator,
    SpreadCalculator,
    MarketData,
    TradingSignal,
    SignalType,
    SignalStrength
)


class TestSpreadCalculator:
    """Test cases for SpreadCalculator class."""
    
    def test_initialization(self):
        """Test spread calculator initialization."""
        calculator = SpreadCalculator()
        
        assert calculator.lookback_window == 100
        assert len(calculator.spread_history) == 0
        assert len(calculator.timestamp_history) == 0
    
    def test_initialization_custom_window(self):
        """Test spread calculator with custom window size."""
        calculator = SpreadCalculator(lookback_window=50)
        
        assert calculator.lookback_window == 50
    
    def test_calculate_spread_ratio_method(self):
        """Test spread calculation using ratio method."""
        calculator = SpreadCalculator()
        
        # Test ratio method
        spread = calculator.calculate_spread(
            pair_id="BTCETH",
            asset1_price=50000.0,  # BTC
            asset2_price=3000.0,   # ETH
            hedge_ratio=15.0,
            method='ratio'
        )
        
        expected_spread = 50000.0 / 3000.0 - 15.0  # ~16.67 - 15.0 = 1.67
        assert abs(spread - expected_spread) < 0.01
        
        # Check history was updated
        assert len(calculator.spread_history["BTCETH"]) == 1
        assert len(calculator.timestamp_history["BTCETH"]) == 1
        assert calculator.spread_history["BTCETH"][0] == spread
    
    def test_calculate_spread_difference_method(self):
        """Test spread calculation using difference method."""
        calculator = SpreadCalculator()
        
        spread = calculator.calculate_spread(
            pair_id="BTCETH",
            asset1_price=50000.0,
            asset2_price=3000.0,
            hedge_ratio=15.0,
            method='difference'
        )
        
        expected_spread = 50000.0 - (15.0 * 3000.0)  # 50000 - 45000 = 5000
        assert spread == expected_spread
    
    def test_calculate_spread_log_ratio_method(self):
        """Test spread calculation using log ratio method."""
        calculator = SpreadCalculator()
        
        spread = calculator.calculate_spread(
            pair_id="BTCETH",
            asset1_price=50000.0,
            asset2_price=3000.0,
            hedge_ratio=15.0,
            method='log_ratio'
        )
        
        expected_spread = np.log(50000.0 / 3000.0) - np.log(15.0)
        assert abs(spread - expected_spread) < 0.001
    
    def test_calculate_spread_invalid_method(self):
        """Test spread calculation with invalid method."""
        calculator = SpreadCalculator()
        
        # Should return 0.0 on error
        spread = calculator.calculate_spread(
            pair_id="BTCETH",
            asset1_price=50000.0,
            asset2_price=3000.0,
            hedge_ratio=15.0,
            method='invalid_method'
        )
        
        assert spread == 0.0
    
    def test_rolling_window_behavior(self):
        """Test that rolling window maintains correct size."""
        window_size = 5
        calculator = SpreadCalculator(lookback_window=window_size)
        
        # Add more spreads than window size
        for i in range(10):
            calculator.calculate_spread(
                pair_id="BTCETH",
                asset1_price=50000.0 + i * 100,
                asset2_price=3000.0,
                hedge_ratio=15.0
            )
        
        # Should only keep last window_size spreads
        assert len(calculator.spread_history["BTCETH"]) == window_size
        assert len(calculator.timestamp_history["BTCETH"]) == window_size
    
    def test_calculate_z_score(self):
        """Test z-score calculation."""
        calculator = SpreadCalculator()
        
        # Add known spread values
        spreads = [1.0, 2.0, 3.0, 4.0, 5.0]
        for spread in spreads:
            calculator.spread_history["BTCETH"] = spreads
        
        # Calculate z-score for value of 6.0
        z_score = calculator.calculate_z_score("BTCETH", 6.0)
        
        # Expected: mean = 3.0, std = sqrt(2.5), z = (6-3)/sqrt(2.5)
        expected_mean = 3.0
        expected_std = np.sqrt(2.5)
        expected_z_score = (6.0 - expected_mean) / expected_std
        
        assert abs(z_score - expected_z_score) < 0.001
    
    def test_calculate_z_score_insufficient_data(self):
        """Test z-score calculation with insufficient data."""
        calculator = SpreadCalculator()
        
        # No data
        z_score = calculator.calculate_z_score("BTCETH")
        assert z_score is None
        
        # Only one data point
        calculator.spread_history["BTCETH"] = [1.0]
        z_score = calculator.calculate_z_score("BTCETH")
        assert z_score is None
    
    def test_calculate_z_score_zero_std(self):
        """Test z-score calculation with zero standard deviation."""
        calculator = SpreadCalculator()
        
        # All same values (zero std)
        calculator.spread_history["BTCETH"] = [2.0, 2.0, 2.0, 2.0]
        
        z_score = calculator.calculate_z_score("BTCETH", 2.0)
        assert z_score == 0.0
    
    def test_get_spread_statistics(self):
        """Test spread statistics calculation."""
        calculator = SpreadCalculator()
        
        # Empty statistics
        stats = calculator.get_spread_statistics("BTCETH")
        assert stats == {}
        
        # Add data
        spreads = [1.0, 2.0, 3.0, 4.0, 5.0]
        calculator.spread_history["BTCETH"] = spreads
        
        stats = calculator.get_spread_statistics("BTCETH")
        
        required_keys = ['mean', 'std', 'min', 'max', 'count', 'current']
        for key in required_keys:
            assert key in stats
        
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['count'] == 5
        assert stats['current'] == 5.0


class TestSignalGenerator:
    """Test cases for SignalGenerator class."""
    
    def test_initialization(self):
        """Test signal generator initialization."""
        generator = SignalGenerator()
        
        assert generator.min_confidence_threshold == 0.6
        assert generator.volatility_adjustment is True
        assert generator.risk_adjustment is True
        assert isinstance(generator.spread_calculator, SpreadCalculator)
        assert len(generator.signal_history) == 0
    
    def test_initialization_custom_params(self):
        """Test signal generator with custom parameters."""
        generator = SignalGenerator(
            min_confidence_threshold=0.8,
            volatility_adjustment=False,
            risk_adjustment=False
        )
        
        assert generator.min_confidence_threshold == 0.8
        assert generator.volatility_adjustment is False
        assert generator.risk_adjustment is False
    
    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient market data."""
        generator = SignalGenerator()
        
        # Empty market data
        market_data = MarketData()
        
        signal = generator.generate_signal("BTCETH", market_data)
        
        assert signal.pair_id == "BTCETH"
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.0
        assert "Insufficient data" in signal.reason
    
    def test_generate_entry_long_signal(self):
        """Test generation of long entry signal."""
        generator = SignalGenerator()
        
        # Create market data that should trigger long entry
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5
        )
        
        # Mock spread calculator to return values that trigger long entry
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.05):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=-2.5):
                signal = generator.generate_signal("BTCETH", market_data)
        
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.side == 'long'
        assert signal.current_z_score == -2.5
        assert signal.entry_price_asset1 == market_data.asset1_ask  # Buy asset1
        assert signal.entry_price_asset2 == market_data.asset2_bid  # Sell asset2
        assert "below long threshold" in signal.reason
    
    def test_generate_entry_short_signal(self):
        """Test generation of short entry signal."""
        generator = SignalGenerator()
        
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5
        )
        
        # Mock spread calculator to return values that trigger short entry
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.15):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=2.5):
                signal = generator.generate_signal("BTCETH", market_data)
        
        assert signal.signal_type == SignalType.ENTRY_SHORT
        assert signal.side == 'short'
        assert signal.current_z_score == 2.5
        assert signal.entry_price_asset1 == market_data.asset1_bid  # Sell asset1
        assert signal.entry_price_asset2 == market_data.asset2_ask  # Buy asset2
        assert "above short threshold" in signal.reason
    
    def test_generate_hold_signal(self):
        """Test generation of hold signal."""
        generator = SignalGenerator()
        
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5
        )
        
        # Mock spread calculator to return values within thresholds
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.08):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=1.0):
                signal = generator.generate_signal("BTCETH", market_data)
        
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.5
        assert "within entry thresholds" in signal.reason
    
    def test_generate_exit_signal_from_long(self):
        """Test generation of exit signal from long position."""
        generator = SignalGenerator()
        
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5
        )
        
        # Mock spread calculator to return values that trigger exit
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.08):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=0.3):
                signal = generator.generate_signal("BTCETH", market_data, current_position='long')
        
        assert signal.signal_type == SignalType.EXIT
        assert "within exit threshold" in signal.reason
    
    def test_generate_stop_loss_signal(self):
        """Test generation of stop loss signal."""
        generator = SignalGenerator()
        
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5
        )
        
        # Mock spread calculator to return values that trigger stop loss
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.02):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=-4.5):
                signal = generator.generate_signal("BTCETH", market_data, current_position='long')
        
        assert signal.signal_type == SignalType.STOP_LOSS
        assert "stop loss" in signal.reason.lower()
    
    def test_signal_strength_calculation(self):
        """Test signal strength calculation."""
        generator = SignalGenerator()
        
        # Test different signal strengths
        test_cases = [
            (1.0, 1.0, SignalStrength.WEAK),      # ratio = 1.0
            (1.5, 1.0, SignalStrength.MEDIUM),    # ratio = 1.5
            (2.0, 1.0, SignalStrength.STRONG),    # ratio = 2.0
            (3.0, 1.0, SignalStrength.VERY_STRONG) # ratio = 3.0
        ]
        
        for signal_magnitude, threshold, expected_strength in test_cases:
            strength = generator._calculate_signal_strength(signal_magnitude, threshold)
            assert strength == expected_strength
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        generator = SignalGenerator()
        
        # Test with different signal strengths
        market_data = MarketData(
            hedge_ratio_confidence=0.8,
            volatility_regime='normal'
        )
        
        confidence = generator._calculate_confidence(market_data, SignalStrength.STRONG)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for strong signal
    
    def test_volatility_adjustment(self):
        """Test volatility-based signal adjustments."""
        generator = SignalGenerator(volatility_adjustment=True)
        
        # Create base signal
        signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            reason="Test signal"
        )
        
        # Test high volatility adjustment
        market_data_high_vol = MarketData(volatility_regime='high')
        adjusted_signal = generator._apply_volatility_adjustment(signal, market_data_high_vol)
        
        assert adjusted_signal.confidence < 0.8  # Should be reduced
        assert "high volatility" in adjusted_signal.reason
        
        # Test low volatility adjustment
        market_data_low_vol = MarketData(volatility_regime='low')
        adjusted_signal = generator._apply_volatility_adjustment(signal, market_data_low_vol)
        
        assert adjusted_signal.confidence >= 0.8  # Should be maintained or increased
    
    def test_risk_adjustment(self):
        """Test risk-based signal adjustments."""
        generator = SignalGenerator(
            risk_adjustment=True,
            min_confidence_threshold=0.7
        )
        
        # Create low confidence signal
        signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            confidence=0.5,  # Below threshold
            reason="Test signal"
        )
        
        market_data = MarketData()
        adjusted_signal = generator._apply_risk_adjustment(signal, market_data)
        
        # Should be filtered to HOLD
        assert adjusted_signal.signal_type == SignalType.HOLD
        assert adjusted_signal.signal_strength == SignalStrength.WEAK
        assert "filtered" in adjusted_signal.reason
    
    def test_signal_history_tracking(self):
        """Test signal history tracking."""
        generator = SignalGenerator()
        
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            entry_threshold_long=-2.0,
            entry_threshold_short=2.0,
            exit_threshold=0.5
        )
        
        # Generate multiple signals
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.08):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=1.0):
                for i in range(5):
                    generator.generate_signal("BTCETH", market_data)
        
        # Check history
        history = generator.get_signal_history("BTCETH")
        assert len(history) == 5
        
        # Check statistics
        stats = generator.get_signal_statistics("BTCETH")
        assert stats['total_signals'] == 5
        assert 'signal_counts' in stats
        assert 'avg_confidence' in stats
    
    def test_error_handling(self):
        """Test error handling in signal generation."""
        generator = SignalGenerator()
        
        # Create market data that will cause an error in processing
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0
        )
        
        # Mock spread calculator to raise an exception
        with patch.object(generator.spread_calculator, 'calculate_spread', side_effect=Exception("Test error")):
            signal = generator.generate_signal("BTCETH", market_data)
        
        # Should return hold signal with error message
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.0
        assert "Error generating signal" in signal.reason


class TestMarketData:
    """Test cases for MarketData dataclass."""
    
    def test_creation(self):
        """Test creating MarketData."""
        timestamp = datetime.utcnow()
        
        market_data = MarketData(
            asset1_bid=49000.0,
            asset1_ask=49100.0,
            asset2_bid=2990.0,
            asset2_ask=3010.0,
            hedge_ratio=15.0,
            volatility_forecast=0.12,
            timestamp=timestamp
        )
        
        assert market_data.asset1_bid == 49000.0
        assert market_data.asset1_ask == 49100.0
        assert market_data.asset2_bid == 2990.0
        assert market_data.asset2_ask == 3010.0
        assert market_data.hedge_ratio == 15.0
        assert market_data.volatility_forecast == 0.12
        assert market_data.timestamp == timestamp
    
    def test_default_timestamp(self):
        """Test default timestamp creation."""
        market_data = MarketData()
        
        assert market_data.timestamp is not None
        assert isinstance(market_data.timestamp, datetime)


class TestTradingSignal:
    """Test cases for TradingSignal dataclass."""
    
    def test_creation(self):
        """Test creating TradingSignal."""
        timestamp = datetime.utcnow()
        
        signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            confidence=0.8,
            side='long',
            current_spread=0.05,
            current_z_score=-2.5,
            timestamp=timestamp,
            reason="Test signal"
        )
        
        assert signal.pair_id == "BTCETH"
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.signal_strength == SignalStrength.STRONG
        assert signal.confidence == 0.8
        assert signal.side == 'long'
        assert signal.current_spread == 0.05
        assert signal.current_z_score == -2.5
        assert signal.timestamp == timestamp
        assert signal.reason == "Test signal"
    
    def test_default_timestamp(self):
        """Test default timestamp creation."""
        signal = TradingSignal(
            pair_id="BTCETH",
            signal_type=SignalType.HOLD,
            signal_strength=SignalStrength.WEAK,
            confidence=0.3
        )
        
        assert signal.timestamp is not None
        assert isinstance(signal.timestamp, datetime)


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data."""
    return MarketData(
        asset1_bid=49000.0,
        asset1_ask=49100.0,
        asset2_bid=2990.0,
        asset2_ask=3010.0,
        hedge_ratio=15.0,
        hedge_ratio_confidence=0.85,
        volatility_forecast=0.12,
        entry_threshold_long=-2.0,
        entry_threshold_short=2.0,
        exit_threshold=0.5,
        volatility_regime='normal'
    )


@pytest.fixture
def sample_signal_generator():
    """Fixture providing a sample signal generator."""
    return SignalGenerator()


class TestSignalGeneratorIntegration:
    """Integration tests for signal generator components."""
    
    def test_complete_signal_workflow(self, sample_signal_generator, sample_market_data):
        """Test complete signal generation workflow."""
        generator = sample_signal_generator
        market_data = sample_market_data
        
        # Mock spread calculator for predictable results
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.08):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=-2.5):
                
                # Generate entry signal
                signal = generator.generate_signal("BTCETH", market_data)
                
                assert signal.pair_id == "BTCETH"
                assert signal.signal_type == SignalType.ENTRY_LONG
                assert signal.confidence > 0.0
                assert signal.current_spread == 0.08
                assert signal.current_z_score == -2.5
                
                # Check that history was updated
                history = generator.get_signal_history("BTCETH")
                assert len(history) == 1
                assert history[0] == signal
    
    def test_position_lifecycle_signals(self, sample_signal_generator, sample_market_data):
        """Test signals throughout position lifecycle."""
        generator = sample_signal_generator
        market_data = sample_market_data
        
        # Entry signal
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.05):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=-2.5):
                entry_signal = generator.generate_signal("BTCETH", market_data)
        
        assert entry_signal.signal_type == SignalType.ENTRY_LONG
        
        # Hold signal while in position
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.07):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=-1.5):
                hold_signal = generator.generate_signal("BTCETH", market_data, current_position='long')
        
        assert hold_signal.signal_type == SignalType.HOLD
        
        # Exit signal
        with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.08):
            with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=0.3):
                exit_signal = generator.generate_signal("BTCETH", market_data, current_position='long')
        
        assert exit_signal.signal_type == SignalType.EXIT
        
        # Check signal history
        history = generator.get_signal_history("BTCETH")
        assert len(history) == 3
        
        signal_types = [s.signal_type for s in history]
        assert SignalType.ENTRY_LONG in signal_types
        assert SignalType.HOLD in signal_types
        assert SignalType.EXIT in signal_types
    
    def test_multiple_pairs_signal_generation(self, sample_signal_generator):
        """Test signal generation for multiple trading pairs."""
        generator = sample_signal_generator
        
        pairs = ["BTCETH", "BTCADA", "ETHADA"]
        
        for pair_id in pairs:
            market_data = MarketData(
                asset1_bid=49000.0,
                asset1_ask=49100.0,
                asset2_bid=2990.0,
                asset2_ask=3010.0,
                hedge_ratio=15.0,
                entry_threshold_long=-2.0,
                entry_threshold_short=2.0,
                exit_threshold=0.5
            )
            
            with patch.object(generator.spread_calculator, 'calculate_spread', return_value=0.08):
                with patch.object(generator.spread_calculator, 'calculate_z_score', return_value=1.0):
                    signal = generator.generate_signal(pair_id, market_data)
            
            assert signal.pair_id == pair_id
            assert signal.signal_type == SignalType.HOLD
        
        # Check that each pair has its own history
        for pair_id in pairs:
            history = generator.get_signal_history(pair_id)
            assert len(history) == 1
            assert history[0].pair_id == pair_id
