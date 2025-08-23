"""Example usage of the backtesting engine."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from .engine import BacktestEngine
from .models import BacktestConfig, LinearSlippageModel, ConstantLatencyModel, TieredFeeModel
from .data_loader import DataLoader, SyntheticDataSource, CSVDataSource
from .strategy_adapter import BacktestStrategyAdapter
from common.logger import configure_logging, get_logger

# Configure logging
configure_logging(
    service_name="backtester",
    log_level="INFO",
    log_format="console"
)

logger = get_logger(__name__)


async def run_synthetic_backtest():
    """Run a backtest with synthetic data."""
    logger.info("Starting synthetic data backtest")
    
    # Configuration
    config = BacktestConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        initial_capital=100000.0,
        symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        slippage_model=LinearSlippageModel(base_slippage_bps=1.0),
        latency_model=ConstantLatencyModel(order_latency_ms=50.0),
        fee_model=TieredFeeModel(taker_fee_bps=1.5),
        market_impact_factor=1.0
    )
    
    # Data loader with synthetic data
    data_loader = DataLoader()
    synthetic_source = SyntheticDataSource(
        symbols=config.symbols,
        start_price=50000.0,
        volatility=0.02,
        tick_frequency_ms=1000  # 1 second ticks
    )
    data_loader.add_data_source("synthetic", synthetic_source, is_default=True)
    
    # Strategy
    strategy = BacktestStrategyAdapter(config.symbols)
    
    # Create and run backtest
    engine = BacktestEngine(config, data_loader, strategy.on_event)
    
    await engine.run()
    
    # Get results
    results = engine.get_results()
    
    # Print performance report
    print("\n" + "="*50)
    print("SYNTHETIC DATA BACKTEST RESULTS")
    print("="*50)
    
    metrics = results['metrics']
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Fees: ${metrics['total_fees']:.2f}")
    
    print(f"\nProcessed {engine.total_events_processed} events")
    print(f"Strategy errors: {engine.strategy_errors}")
    
    return results


async def run_csv_backtest(data_directory: str):
    """Run a backtest with CSV data."""
    logger.info("Starting CSV data backtest", data_directory=data_directory)
    
    # Check if data directory exists
    data_path = Path(data_directory)
    if not data_path.exists():
        logger.error("Data directory does not exist", directory=data_directory)
        return None
    
    # Configuration
    config = BacktestConfig(
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 7),  # 1 week
        initial_capital=100000.0,
        symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        slippage_model=LinearSlippageModel(
            base_slippage_bps=0.5,
            size_impact_factor=0.1,
            max_slippage_bps=5.0
        ),
        latency_model=ConstantLatencyModel(
            order_latency_ms=25.0,
            market_data_latency_ms=5.0
        ),
        fee_model=TieredFeeModel(
            maker_fee_bps=0.75,
            taker_fee_bps=1.0
        ),
        market_impact_factor=0.8
    )
    
    # Data loader with CSV data
    data_loader = DataLoader()
    csv_source = CSVDataSource(data_directory)
    data_loader.add_data_source("csv", csv_source, is_default=True)
    
    # Check available symbols
    available_symbols = data_loader.get_available_symbols()
    logger.info("Available symbols", symbols=available_symbols)
    
    if not available_symbols:
        logger.error("No data files found in directory")
        return None
    
    # Use only available symbols
    config.symbols = [s for s in config.symbols if s in available_symbols]
    
    if not config.symbols:
        logger.error("None of the requested symbols are available")
        return None
    
    # Strategy
    strategy = BacktestStrategyAdapter(config.symbols)
    
    # Create and run backtest
    engine = BacktestEngine(config, data_loader, strategy.on_event)
    
    await engine.run()
    
    # Get results
    results = engine.get_results()
    
    # Print performance report
    print("\n" + "="*50)
    print("CSV DATA BACKTEST RESULTS")
    print("="*50)
    
    metrics = results['metrics']
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Max Drawdown Duration: {metrics['max_drawdown_duration_days']:.1f} days")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Win: ${metrics['average_win']:.2f}")
    print(f"Average Loss: ${metrics['average_loss']:.2f}")
    print(f"Total Fees: ${metrics['total_fees']:.2f}")
    
    print(f"\nProcessed {engine.total_events_processed} events")
    print(f"Strategy errors: {engine.strategy_errors}")
    
    # Show final positions
    final_positions = results['final_positions']
    if final_positions:
        print("\nFinal Positions:")
        for symbol, position in final_positions.items():
            print(f"  {symbol}: {position['quantity']:.6f} @ ${position['average_price']:.2f} "
                  f"(PnL: ${position['unrealized_pnl'] + position['realized_pnl']:.2f})")
    
    return results


def create_sample_csv_data(output_directory: str):
    """Create sample CSV data files for testing."""
    import pandas as pd
    import numpy as np
    
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True)
    
    logger.info("Creating sample CSV data", directory=output_directory)
    
    # Generate sample data for each symbol
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    start_prices = {'BTCUSDT': 45000, 'ETHUSDT': 2800, 'ADAUSDT': 0.45}
    
    start_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 1, 7)
    
    for symbol in symbols:
        data = []
        current_time = start_time
        current_price = start_prices[symbol]
        
        while current_time <= end_time:
            # Generate price movement
            price_change = np.random.normal(0, current_price * 0.001)  # 0.1% volatility per minute
            current_price = max(current_price + price_change, 0.01)
            
            # Generate order book
            spread_bps = np.random.uniform(1, 5)
            spread = current_price * spread_bps / 10000
            
            best_bid = current_price - spread / 2
            best_ask = current_price + spread / 2
            
            # Create 5 levels each side
            bids = []
            asks = []
            
            for i in range(5):
                bid_price = best_bid - i * spread * 0.1
                ask_price = best_ask + i * spread * 0.1
                
                bid_qty = np.random.uniform(0.1, 10.0)
                ask_qty = np.random.uniform(0.1, 10.0)
                
                bids.append([bid_price, bid_qty])
                asks.append([ask_price, ask_qty])
            
            data.append({
                'timestamp': current_time.isoformat(),
                'bids': str(bids).replace("'", '"'),
                'asks': str(asks).replace("'", '"'),
                'sequence_id': len(data)
            })
            
            current_time += timedelta(minutes=1)  # 1-minute intervals
        
        # Save to CSV
        df = pd.DataFrame(data)
        csv_file = output_path / f"{symbol}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info("Created sample data file", 
                   symbol=symbol, 
                   file=str(csv_file), 
                   records=len(data))


async def run_live_vs_backtest_parity_test():
    """Run a parity test between live and backtest results."""
    logger.info("Starting live vs backtest parity test")
    
    # This would involve:
    # 1. Recording live market data for a period
    # 2. Running the strategy live (paper trading)
    # 3. Running the same data through the backtester
    # 4. Comparing the results
    
    # For now, we'll simulate this with synthetic data
    logger.info("Parity test simulation - using identical synthetic data")
    
    # Run two identical backtests to verify consistency
    results1 = await run_synthetic_backtest()
    results2 = await run_synthetic_backtest()
    
    # Compare results (they should be identical with same random seed)
    metrics1 = results1['metrics']
    metrics2 = results2['metrics']
    
    print("\n" + "="*50)
    print("PARITY TEST RESULTS")
    print("="*50)
    
    tolerance = 1e-6
    
    for key in ['total_return', 'final_portfolio_value', 'total_trades']:
        diff = abs(metrics1[key] - metrics2[key])
        status = "PASS" if diff < tolerance else "FAIL"
        print(f"{key}: {status} (diff: {diff})")
    
    return results1, results2


async def main():
    """Main function to run backtesting examples."""
    print("Crypto HFT Backtesting Engine Examples")
    print("="*50)
    
    # Example 1: Synthetic data backtest
    print("\n1. Running synthetic data backtest...")
    await run_synthetic_backtest()
    
    # Example 2: Create sample CSV data and run backtest
    print("\n2. Creating sample CSV data and running backtest...")
    sample_data_dir = "sample_data"
    create_sample_csv_data(sample_data_dir)
    await run_csv_backtest(sample_data_dir)
    
    # Example 3: Parity test
    print("\n3. Running parity test...")
    await run_live_vs_backtest_parity_test()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
