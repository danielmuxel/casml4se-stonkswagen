# Backtest Horizon Explained

## What Does "1-Step-Ahead" Mean?

**Backtest horizon** controls how far into the future you predict during evaluation.

### Visual Example

```
Actual Data:  •───•───•───•───•───•───•───•
Time:         t0  t1  t2  t3  t4  t5  t6  t7

┌─────────────────────────────────────────────────────┐
│ 1-Step-Ahead Forecast (horizon=1)                   │
├─────────────────────────────────────────────────────┤
│ At t3, predict t4:                                  │
│   Train on: [t0, t1, t2, t3]                       │
│   Predict:  t4 ✓ (next immediate point)            │
│                                                     │
│ At t4, predict t5:                                  │
│   Train on: [t0, t1, t2, t3, t4]                   │
│   Predict:  t5 ✓                                    │
│                                                     │
│ Result: Very accurate (short-term)                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 6-Step-Ahead Forecast (horizon=6)                   │
├─────────────────────────────────────────────────────┤
│ At t1, predict t7:                                  │
│   Train on: [t0, t1]                               │
│   Predict:  t7 (6 steps ahead)                     │
│                                                     │
│ Result: Less accurate (long-term)                   │
└─────────────────────────────────────────────────────┘
```

## Why Does This Matter?

### 1-Step-Ahead (Backtest Horizon = 1)
- ✅ **Most accurate** - only predicting next value
- ✅ **Best for evaluation** - shows if model can track short-term patterns
- ✅ **Catches overfitting** - hard to fake good performance
- ⚠️  May look "too accurate" (but it's legitimate!)

### Multi-Step-Ahead (Backtest Horizon = 6 or 12)
- ✅ **More realistic** - shows long-term forecasting ability
- ✅ **Easier to visualize** - more visible lag between forecast and actual
- ⚠️  Less accurate (which is expected and normal)
- ⚠️  Can hide data leakage issues

## In Your Notebook

```python
BACKTEST_HORIZON = 1      # Evaluate: How well can model predict next step?
FORECAST_HORIZON = 12     # Production: Predict 12 steps into future
```

### Why Different?

**Backtest (horizon=1):**
- Goal: Evaluate model accuracy honestly
- Question: "Can the model track the series step-by-step?"
- Result: Forecast will track actual closely (this is GOOD!)

**Production (horizon=12):**
- Goal: Predict far into the future for planning
- Question: "What will happen 12 hours from now?"
- Result: Forecast for actual future data we don't have yet

## Example Output

### With horizon=1 (Current)
```
Time    Actual  Forecast  Error
14:00   107     106.8     0.2  ✓ Very close
15:00   110     109.5     0.5  ✓ Very close
16:00   108     107.8     0.2  ✓ Very close

MAPE: 2.5% (Excellent!)
```

### With horizon=6
```
Time    Actual  Forecast  Error
14:00   107     102.3     4.7  ✗ Larger error
15:00   110     105.1     4.9  ✗ Larger error
16:00   108     103.2     4.8  ✗ Larger error

MAPE: 8.2% (Still good, but less accurate)
```

## Common Misconception

❌ **Wrong thinking:** "horizon=1 forecasts are too accurate, must be data leakage!"

✅ **Correct thinking:** "horizon=1 forecasts SHOULD be accurate for a good model!"

**Why?**
- With proper train/test split (which we have now), accuracy is legitimate
- ARIMA is designed to predict next value based on recent patterns
- For stable time series, next value is often predictable

## When to Use Each

| Use Case | Horizon | Why |
|----------|---------|-----|
| **Model evaluation** | 1-3 | Test if model learns patterns |
| **Short-term planning** | 1-6 | Next hour, next day |
| **Long-term planning** | 12-24 | Next week, next month |
| **Detecting data leakage** | 1 | Harder to fake good performance |

## Try It Yourself

In the notebook, change:
```python
BACKTEST_HORIZON = 6  # Try 6-step-ahead instead
```

You'll see:
- Forecast and actual have more separation
- MAPE increases (less accurate)
- More visible "lag" in the plot
- Still no data leakage! (because of train/test split)

The key insight: **With proper train/test split, ANY horizon is valid. Choose based on your use case, not to make metrics look better!**
