# HFT-020: Market Regime Classification System

## Overview

The Market Regime Classification system uses Hidden Markov Models (HMM) and unsupervised learning to identify and adapt to different market conditions in real-time. The system dynamically adjusts trading strategy parameters based on the classified market regime to optimize performance under varying market conditions.

## Architecture

### Core Components

1. **Regime Classifier Service** (`services/regime_classifier/`)
   - Real-time market regime classification using HMM
   - Feature extraction from market data
   - Adaptive strategy parameter adjustment
   - Alert generation for regime changes

2. **Feature Extractor** (`features/feature_extractor.py`)
   - Time series feature engineering
   - Technical indicators calculation
   - Market microstructure analysis
   - Volatility and trend feature extraction

3. **HMM Classifier** (`classifiers/hmm_classifier.py`)
   - Hidden Markov Model implementation
   - State-to-regime mapping
   - Model training and prediction
   - Performance evaluation

4. **Strategy Adapter** (`adapters/strategy_adapter.py`)
   - Regime-aware parameter adaptation
   - Performance-based learning
   - Risk-adjusted position sizing
   - Confidence-based adjustments

5. **Regime Monitor** (`monitoring/regime_monitor.py`)
   - Real-time visualization
   - Performance analytics
   - Alert dashboard
   - Regime transition analysis

## Market Regimes

The system identifies 12 distinct market regimes:

### Volatility-Based Regimes
- **LOW_VOL_BULL**: Low volatility, upward trend
- **LOW_VOL_BEAR**: Low volatility, downward trend  
- **LOW_VOL_RANGE**: Low volatility, sideways movement
- **HIGH_VOL_BULL**: High volatility, upward trend
- **HIGH_VOL_BEAR**: High volatility, downward trend
- **HIGH_VOL_RANGE**: High volatility, sideways movement

### Special Regimes
- **STABLE_RANGE**: Very stable, range-bound market
- **TRENDING_UP**: Strong upward trend
- **TRENDING_DOWN**: Strong downward trend
- **CRISIS**: Market crisis/extreme volatility
- **RECOVERY**: Post-crisis recovery phase
- **UNKNOWN**: Unclassified/transition state

## Feature Engineering

### Price-Based Features
- Multi-timeframe returns (1h, 4h, 24h, 7d)
- Cross-asset correlations
- Market beta calculations

### Volatility Features
- Realized volatility across timeframes
- GARCH volatility estimates
- Volatility clustering measures
- Regime indicators

### Volume Features
- Volume ratios vs. historical averages
- Volume-price relationships
- Trade intensity metrics

### Technical Indicators
- RSI (Relative Strength Index)
- MACD signals
- Bollinger Bands position
- Average True Range (ATR)

### Market Microstructure
- Bid-ask spreads
- Order book imbalance
- Trade intensity
- Liquidity measures

### External Data
- Funding rates
- Open interest changes
- Fear & Greed Index
- Social sentiment scores

## Strategy Adaptation

### Parameter Adjustment

Each regime has optimized strategy parameters:

```python
# Example: Crisis Regime Parameters
StrategyParameters(
    regime=MarketRegime.CRISIS,
    entry_z_score=5.0,        # Very selective entries
    exit_z_score=2.0,         # Quick exits
    max_position_size=0.2,    # Minimal position sizes
    kelly_fraction=0.05,      # Conservative sizing
    execution_urgency=0.9,    # Fast execution
    holding_period_target=60  # Short holding periods
)
```

### Adaptive Learning

The system continuously learns from performance:

- **Sharpe Ratio Feedback**: Adjusts aggressiveness based on risk-adjusted returns
- **Win Rate Analysis**: Modifies entry/exit thresholds
- **Drawdown Control**: Reduces risk after significant losses
- **Confidence Scaling**: Adjusts position sizes based on classification confidence

### Risk Management Integration

- **Emergency Halt**: Automatic trading halt in crisis regimes
- **Position Scaling**: Dynamic position sizing based on regime risk
- **Confidence Thresholds**: Reduced activity during uncertain classifications
- **Transition Smoothing**: Gradual parameter changes during regime transitions

## Configuration

### HMM Model Configuration

```python
RegimeModelConfig(
    n_components=6,              # Number of hidden states
    covariance_type="full",      # Covariance matrix type
    n_iter=100,                  # Maximum training iterations
    tol=1e-2,                    # Convergence tolerance
    min_confidence_threshold=0.3, # Minimum classification confidence
    training_window=2160,        # Training window (90 days)
    retrain_frequency=168        # Retrain weekly
)
```

### Service Configuration

```yaml
environment:
  - CLASSIFICATION_INTERVAL=300      # 5-minute classification cycle
  - FEATURE_LOOKBACK_HOURS=168      # 1-week feature window
  - HMM_N_COMPONENTS=6              # Number of regimes
  - MIN_CONFIDENCE_THRESHOLD=0.3    # Confidence threshold
  - RETRAIN_FREQUENCY=168           # Weekly retraining
  - REGIME_PERSISTENCE=3            # Confirmation periods
```

## Deployment

### Docker Deployment

```bash
# Build and start the regime classifier service
docker-compose up regime-classifier-service

# View logs
docker-compose logs -f regime-classifier-service

# Check health
curl http://localhost:8007/health
```

### Kubernetes Deployment

```bash
# Deploy using Helm
helm upgrade --install regime-classifier ./infrastructure/helm/charts/regime-classifier

# Check status
kubectl get pods -l app=regime-classifier
kubectl logs -l app=regime-classifier -f
```

## API Endpoints

### Health Check
```
GET /health
```

### Current Regime
```
GET /regime/current
Response: {
  "regime": "low_vol_bull",
  "confidence": 0.85,
  "confidence_level": "high",
  "timestamp": "2024-01-15T10:30:00Z",
  "regime_probabilities": {...},
  "regime_duration": 7200
}
```

### Strategy Parameters
```
GET /regime/parameters
Response: {
  "regime": "low_vol_bull",
  "entry_z_score": 1.5,
  "exit_z_score": 0.3,
  "max_position_size": 1.2,
  ...
}
```

### Recent Alerts
```
GET /regime/alerts?limit=10
Response: [
  {
    "timestamp": "2024-01-15T10:25:00Z",
    "alert_type": "regime_change",
    "regime": "high_vol_bear",
    "severity": "medium",
    "message": "Market regime changed...",
    ...
  }
]
```

### Force Classification
```
POST /regime/classify
Response: {
  "classification": {...},
  "parameters": {...},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Monitoring and Visualization

### Grafana Dashboard

Key metrics monitored:
- Current regime and confidence
- Regime transition frequency
- Classification accuracy
- Parameter adaptation history
- Alert frequency by severity

### Plotly Visualizations

1. **Regime Timeline**: Historical regime classifications with price overlay
2. **Transition Matrix**: Heatmap of regime transitions
3. **Performance Chart**: Performance metrics by regime
4. **Confidence Distribution**: Classification confidence analysis
5. **Alert Timeline**: Regime alerts over time

### Performance Metrics

- **Classification Accuracy**: Overall and regime-specific accuracy
- **Transition Detection**: Speed and accuracy of regime change detection
- **Trading Performance**: Returns and Sharpe ratios by regime
- **Risk Metrics**: Drawdowns and volatility by regime

## Integration with Execution Service

### Kafka Topics

- **Input**: `market-data` (price, volume, orderbook data)
- **Output**: `regime-classifications` (regime updates)
- **Output**: `regime-alerts` (critical regime changes)

### Execution Integration

```python
# Regime-aware position sizing
def calculate_position_size(base_size, regime_info):
    scaling_factor = regime_info['position_scaling_factor']
    max_size = regime_info['parameters']['max_position_size']
    
    adjusted_size = base_size * scaling_factor
    return min(adjusted_size, max_size)

# Regime-based entry/exit thresholds
def get_trading_thresholds(regime_info):
    params = regime_info['parameters']
    return {
        'entry_threshold': params['entry_z_score'],
        'exit_threshold': params['exit_z_score'],
        'stop_loss': params['stop_loss_z_score']
    }
```

## Testing

### Unit Tests

```bash
# Run regime classifier tests
pytest tests/services/regime_classifier/ -v

# Run specific test modules
pytest tests/services/regime_classifier/test_models.py
pytest tests/services/regime_classifier/test_feature_extractor.py
```

### Integration Tests

```bash
# Test full classification pipeline
pytest tests/integration/test_regime_classification.py

# Test Kafka integration
pytest tests/integration/test_regime_kafka.py
```

### Backtesting Integration

```python
# Test regime adaptation in backtesting
from backtester.engine import BacktestEngine
from services.regime_classifier.main import RegimeClassificationService

# Run backtest with regime adaptation
results = backtest_engine.run_with_regime_adaptation(
    start_date='2023-01-01',
    end_date='2023-12-31',
    regime_service=regime_service
)
```

## Performance Optimization

### Model Optimization

1. **Feature Selection**: Use statistical tests to select most predictive features
2. **Hyperparameter Tuning**: Optimize HMM parameters using cross-validation
3. **Ensemble Methods**: Combine multiple models for robust classification
4. **Online Learning**: Implement incremental model updates

### Computational Optimization

1. **Caching**: Cache feature calculations and model predictions
2. **Parallel Processing**: Parallelize feature extraction and model training
3. **Memory Management**: Optimize data structures and memory usage
4. **GPU Acceleration**: Use GPU for intensive computations

## Troubleshooting

### Common Issues

1. **Low Classification Confidence**
   - Check feature quality and data completeness
   - Verify model training data sufficiency
   - Review regime mapping logic

2. **Frequent Regime Changes**
   - Increase `REGIME_PERSISTENCE` parameter
   - Adjust `TRANSITION_SMOOTHING` factor
   - Review confidence thresholds

3. **Poor Performance in Specific Regimes**
   - Analyze regime-specific parameters
   - Review feature importance for that regime
   - Check for regime-specific data issues

### Debugging Tools

```bash
# Check service logs
docker-compose logs regime-classifier-service

# Monitor Kafka topics
kafka-console-consumer --topic regime-classifications

# Check model performance
curl http://localhost:8007/regime/statistics

# Export regime data for analysis
curl http://localhost:8007/regime/export
```

## Future Enhancements

### Planned Features

1. **Multi-Asset Regime Detection**: Extend to portfolio-level regime classification
2. **Regime Forecasting**: Predict future regime changes
3. **Alternative Models**: Implement Gaussian Mixture Models, Neural Networks
4. **Real-time Feature Engineering**: Stream processing for feature calculation
5. **Regime-Specific Strategies**: Develop specialized strategies per regime

### Research Directions

1. **Graph Neural Networks**: Model market relationships as dynamic graphs
2. **Reinforcement Learning**: Learn optimal parameter adaptations
3. **Sentiment Integration**: Incorporate news and social media sentiment
4. **Cross-Market Analysis**: Include traditional markets and commodities

## Conclusion

The Market Regime Classification system provides a sophisticated framework for adaptive trading strategy management. By automatically identifying market conditions and adjusting parameters accordingly, the system aims to improve risk-adjusted returns across varying market environments.

The system's modular architecture allows for easy extension and customization, while comprehensive monitoring and alerting ensure reliable operation in production environments.
