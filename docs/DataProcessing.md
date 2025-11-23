# Notes about Data PreProcessing for Financial Data. 


## https://arxiv.org/pdf/1907.03010 Financial Time Series Data Processing for Machine
Learning
### Summarized by claude:
**Key Findings  
Stationarity & Scaling Methods**

Raw prices are non-stationary (ADF test: -0.23, p-value: 0.99), requiring transformation  
Returns make data stationary (ADF: -14.95, p-value: 0.00)  
Scaling approach matters: The author recommends slicing first, then scaling each slice independently (rather than scaling the entire dataset first) to normalize training examples into the same range  
Standardization slightly outperforms MinMax scaling in preserving price relationships, validated through empirical tests where simple LSTM models learned basic price patterns (e.g., C_t > C_t-5) with 97-99% accuracy  

**Critical Data Split Method**  
The paper highlights a major pitfall in time series splitting: shuffling before splitting causes data leakage because overlapping slices share up to 95% of their data.  

**Correct approach:**

Split chronologically into train/validation/test sets first  
Only shuffle the training set afterward  

This prevents the model from partially fitting to validation data during training.  

**Labeling Strategies**  
Multiple labeling options are presented beyond simple "next return" prediction:  

Classification: N-bar up/down, trend direction, QClass (quality-based classes)  
Regression: Price change, log returns, trend strength, %Q metric  

The %Q metric is particularly novel—it measures move quality as a risk/reward ratio:

%Q = 1: perfect upward move (no drawdown)  
%Q = 0: perfect downward move (no drawup)  
%Q = 0.5: equal up/down movement  

**Feature Scaling Recommendations**

Overlaid indicators (moving averages, Bollinger Bands): scale together with prices to preserve relationships  
Volume and separated indicators: scale independently  
Bounded indicators (RSI): divide by maximum value to preserve meaningful threshold levels (e.g., 70/30 becomes 0.7/0.3)  

**Long-Term Trend Bias**  
Stock markets' upward bias can cause models to only predict "Up" movements. The solution is downsampling the majority class during training to achieve balanced class distribution.  

**Practical Impact**
The paper emphasizes that preprocessing choices (features, scaling, labeling) often impact results more significantly than hyperparameter tuning, making these decisions critical for developing profitable trading strategies.

### Notes
A timeseries should ideally be stationary, that means that there should not be an inherent trend. 

### Augmented Dickey-Fuller test
The Augmented Dickey-Fuller (ADF) test is a statistical test that checks whether a time series is stationary or not.  
**What is Stationarity?**
A stationary time series has statistical properties (mean, variance) that don't change over time. Think of it this way:  

- Non-stationary: Stock prices that trend upward over years - the mean keeps increasing
- Stationary: Daily returns that fluctuate around zero - the mean stays constant

**How the ADF Test Works**
The test checks for a "unit root" in the time series, which is a mathematical way of saying "does this series have a trend or random walk pattern?"
Reading the results:

- ADF statistic: A more negative number = more likely to be stationary
- p-value: Lower is better (typically want p < 0.05)

- p-value < 0.05: Reject the null hypothesis → series is stationary ✓
- p-value > 0.05: Cannot reject null hypothesis → series is non-stationary ✗
