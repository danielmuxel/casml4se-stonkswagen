# Financial Time Series Data Processing for Machine Learning
## Implementation Guide

## Summary

This whitepaper by Fabrice Daniel addresses the critical challenges of processing financial time series data for machine learning applications. It systematically examines preprocessing methods (returns, MinMax, Standardization), validates their stationarity, and tests their ability to preserve price relationships. The paper demonstrates that improper data splitting can lead to data leakage, proposes slice-then-scale approaches for better model training, and introduces various labeling strategies including a novel %Q metric for measuring trade quality. The core insight is that financial data requires specialized handling due to its non-stationary nature and the temporal dependencies inherent in time series.

## Implementation Steps

### 1. Data Acquisition and Initial Preparation

**Objective**: Load historical OHLCV (Open, High, Low, Close, Volume) data

- Obtain daily or intraday price data for your target instrument
- Ensure data includes: Open, High, Low, Close, Volume (if available)
- Date range: sufficient history for meaningful training (paper uses 1993-2019 for SPY)
- Verify data quality: check for missing values, outliers, corporate actions

### 2. Stationarity Testing on Raw Data

**Objective**: Confirm raw prices are non-stationary (baseline check)

- Apply Augmented Dickey-Fuller (ADF) test to raw closing prices
- Expected result: ADF statistic > -3.96, high p-value (>0.05)
- This confirms prices follow a random walk and need transformation
- Use libraries: `statsmodels.tsa.stattools.adfuller` in Python

### 3. Calculate Returns

**Objective**: Transform prices into stationary returns series

- Compute simple returns: `r_t = (C_t - C_{t-1}) / C_{t-1}`
- Alternative: log returns: `r_t = log(C_t / C_{t-1})`
- Apply ADF test to returns series
- Expected result: ADF statistic < -3.96, p-value ≈ 0
- This confirms returns are stationary

### 4. Define Slicing Parameters

**Objective**: Set up the temporal window structure

- **Lookback period (n)**: number of consecutive observations per slice (e.g., 20 bars)
- **Time horizon**: prediction window (e.g., predict 20 bars ahead)
- **Step size**: increment between slices (typically 1 for maximum data usage)
- Example: For lookback=20, slice S_t contains observations from t-19 to t

### 5. Implement Slice-Then-Scale Approach

**Objective**: Create training samples with independent scaling per slice

**Critical order**: Slice first, then scale (not scale then slice)

**Process**:
- Create K overlapping slices from your time series
- For each slice independently, apply scaling method
- This ensures each training sample has the same range [0,1] or standardized distribution

**Scaling Methods**:

**MinMax Scaling** (per slice):
```
z = ((x - x_min) / (x_max - x_min)) * (max - min) + min
```
- Typically scale to [0, 1] or [-1, 1]
- Apply to: OHLC prices together (preserves relative positions)

**Standardization** (per slice):
```
z = (x - μ) / σ
```
- μ = mean of the slice
- σ = standard deviation of the slice
- Paper shows slightly better performance than MinMax

### 6. Handle Multiple Features

**Objective**: Scale different feature types appropriately

**Overlaid Indicators** (scale together with prices):
- Moving averages (SMA, EMA)
- Bollinger Bands
- Any indicator plotted on same scale as price
- Scale these WITH OHLC to preserve relative relationships

**Separated Indicators** (scale independently):
- Volume: use MinMax or Standardization separately
- RSI: divide by 100 (converts 0-100 range to 0-1)
- MACD, Stochastic: scale separately with MinMax/Standardization
- This preserves meaningful threshold levels (e.g., RSI 70/30 becomes 0.7/0.3)

### 7. Verify Price Relationship Preservation

**Objective**: Test that scaling preserves learnable patterns

**Create simple test cases**:
1. Binary classification: `C_t > C_{t-5}`
2. Moving average cross: `C_t > EMA5_t`
3. Highest close: `C_t > HC10_t` (highest of last 10 closes)

**Train simple LSTM model**:
- Architecture: 1 LSTM layer (64 units, tanh), 1 Dense output (2 units, softmax)
- 100 epochs, batch size 64, Adam optimizer
- Expected validation accuracy: >90% for simple relationships, >97% for very simple ones

**Interpretation**:
- If model can't learn these simple patterns, scaling failed to preserve information
- Paper shows Standardization performs slightly better than MinMax

### 8. Implement Proper Data Splitting

**Objective**: Avoid temporal data leakage

**CRITICAL**: Do NOT shuffle before splitting (common mistake)

**Correct procedure**:
1. **First**: Split chronologically into train/val/test sets
   - Training: slices S_0 to S_{ts-1}
   - Validation: slices S_ts to S_{ts+vs-1}
   - Test: slices S_{ts+vs} to S_{K-1}
   - Typical split: 70% train, 15% validation, 15% test

2. **Then**: Shuffle ONLY the training set
   - Validation and test sets remain in chronological order
   - This prevents future information leaking into training

**Why this matters**:
- Overlapping slices share up to 95% of data (for lookback=20, step=1)
- Random shuffle would place highly correlated samples in different sets
- This causes artificial validation performance inflation

### 9. Choose Labeling Strategy

**Objective**: Define prediction target based on trading goal

**Common Labels**:

**Binary Classification**:
- `N bars Up/Down`: C_{t+n} > C_t
- `Moving Average Cross`: MA_{t+n} > MA_t
- `Trend Direction`: based on linear regression or price position vs MA

**Regression**:
- `N bars price change`: C_{t+n} - C_t
- `N bars log returns`: log(C_{t+n} / C_t)
- `Trend Strength`: slope or R² of linear regression

**Advanced: %Q (Quality) Metric**:
```
%Q = (HH_{t+1:t+n} - C_t) / (HH_{t+1:t+n} - LL_{t+1:t+n})
```
Where:
- HH = Highest High in next n bars
- LL = Lowest Low in next n bars
- Interpretation: risk/reward ratio
  - %Q = 1: perfect uptrend (no drawdown)
  - %Q = 0: perfect downtrend (no drawup)
  - %Q = 0.5: equal up/down movement

**QClass (from %Q)**:
- Class 0 (Up): %Q >= 0.6
- Class 1 (Neutral): 0.4 < %Q < 0.6
- Class 2 (Down): %Q <= 0.4

### 10. Address Class Imbalance

**Objective**: Prevent model from only learning dominant trend direction

**Problem**: Long-term uptrend in equities causes label imbalance
- Model may predict only "Up" and still achieve 60-70% accuracy
- Check confusion matrix to detect this issue

**Solution**: Downsample majority class in training set
- Count instances of each class
- Randomly remove samples from majority class until balanced
- Apply only to training set, not validation/test
- Alternative: use class weights in loss function

### 11. Build Input Shape

**Objective**: Format data for model consumption

**Single feature** (e.g., close only):
- Shape: `(m, s)` where m=samples, s=timesteps
- Example: (1000, 20) for 1000 slices of 20 bars

**Multiple features** (e.g., OHLCV):
- Shape: `(m, s, f)` where f=features
- Example: (1000, 20, 5) for 1000 slices, 20 bars, 5 features

**Model compatibility**:
- LSTMs/GRUs: accept 3D input `(samples, timesteps, features)`
- Dense/CNN layers: may require flattening to 2D `(samples, timesteps*features)`
- Tree-based models (Random Forest, XGBoost): typically need 2D without scaling

### 12. Model Development and Testing

**Objective**: Train and validate your model

**Training recommendations**:
- Start with simple architecture (single LSTM layer)
- Use dropout (0.2) and recurrent dropout (0.2) for regularization
- Monitor both training and validation loss
- Early stopping if validation loss stops improving
- Adjust epochs (100-150 typically sufficient)

**Validation**:
- Evaluate on chronologically later data (validation set)
- Check confusion matrix for classification tasks
- Calculate precision, recall, F1-score
- For regression: MAE, MSE, RMSE, R²

**Testing**:
- Final evaluation on held-out test set
- Simulate trading strategy with transaction costs
- Calculate Sharpe ratio, maximum drawdown, win rate
- Backtest on completely unseen recent data

### 13. Generalization Across Instruments and Timeframes

**Objective**: Apply methodology to different markets

**Applicability**:
- Asset classes: Equities, Commodities, Forex, Cryptocurrencies
- Timeframes: Daily, hourly, 5-minute, 1-minute
- Adjustments for Forex: volume may be synthetic or unavailable

**Considerations**:
- Market-specific behaviors (e.g., mean reversion vs trending)
- Liquidity and bid-ask spreads
- Trading hours and gaps
- Volatility regimes

### 14. Iterate on Feature/Label Combinations

**Objective**: Optimize through experimentation

**Systematic approach**:
- Document baseline performance with simple features/labels
- Test different feature combinations systematically
- Compare various labeling strategies for your trading objective
- Often more impactful than hyperparameter tuning
- Use validation set performance as selection criterion

**Key insight from paper**: Feature engineering and label selection often matter more than model architecture optimization.

---

## Key Takeaways

1. **Slice first, then scale** - Critical for proper training sample normalization
2. **Split chronologically, then shuffle training only** - Prevents data leakage
3. **Verify scaling preserves patterns** - Test with simple classification tasks
4. **Scale different indicators appropriately** - Overlaid together, separated independently
5. **Address class imbalance** - Downsample majority class in training
6. **Choose labels aligned with trading goals** - %Q metric focuses on tradability
7. **Feature/label selection > hyperparameter tuning** - Often has greater impact

## References

- Brockwell & Davis (2002) - Introduction to Time Series and Forecasting
- Hyndman & Athanasopoulos (2018) - Forecasting: Principles and Practice
- Tsay (2010) - Analysis of Financial Time Series
- De Prado (2018) - Advances in Financial Machine Learning