# yfinance Reference Card & Cheat Sheet

## Basic Components

### Installation and Setup

```python
# Install
pip install yfinance

# Import
import yfinance as yf
```

### Ticker Objects

```python
# Create a Ticker object
ticker = yf.Ticker("AAPL")

# Create multiple Ticker objects
tickers = yf.Tickers("AAPL MSFT GOOG")
```

### Basic Data Retrieval

```python
# Get basic information
ticker.info

# Get recent market data
data = ticker.history(period="1mo")

# Get specific data columns
close_prices = ticker.history(period="1mo")["Close"]
```

### Historical Price Data

```python
# Various time periods
data_1d = ticker.history(period="1d")  # 1 day
data_5d = ticker.history(period="5d")  # 5 days
data_1mo = ticker.history(period="1mo")  # 1 month
data_3mo = ticker.history(period="3mo")  # 3 months
data_6mo = ticker.history(period="6mo")  # 6 months
data_1y = ticker.history(period="1y")    # 1 year
data_2y = ticker.history(period="2y")    # 2 years
data_5y = ticker.history(period="5y")    # 5 years
data_10y = ticker.history(period="10y")  # 10 years
data_ytd = ticker.history(period="ytd")  # Year to date
data_max = ticker.history(period="max")  # Maximum available

# Date ranges with start and end
data_range = ticker.history(start="2020-01-01", end="2020-12-31")

# Specifying intervals (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
data_hourly = ticker.history(period="1mo", interval="1h")
data_weekly = ticker.history(period="1y", interval="1wk")
```

### Financial Data

```python
# Financials
income_stmt = ticker.income_stmt         # Income statement
quarterly_income = ticker.quarterly_income_stmt  # Quarterly income statement
balance_sheet = ticker.balance_sheet     # Balance sheet 
quarterly_balance = ticker.quarterly_balance_sheet  # Quarterly balance sheet
cash_flow = ticker.cashflow              # Cash flow
quarterly_cash_flow = ticker.quarterly_cashflow  # Quarterly cash flow

# Earnings information
earnings = ticker.earnings               # Annual earnings
quarterly_earnings = ticker.quarterly_earnings  # Quarterly earnings
```

## Advanced Components

### Downloads and Multiple Tickers

```python
# Download data for multiple tickers at once
data = yf.download("AAPL MSFT GOOG", period="1y")

# Download with additional parameters
data = yf.download(
    tickers="SPY AAPL",
    period="1y", 
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    prepost=True,
    threads=True
)
```

### Actions and Dividends

```python
# Get stock actions (dividends, splits)
actions = ticker.actions

# Get just dividends or splits
dividends = ticker.dividends
splits = ticker.splits

# Get capital gains (for funds)
capital_gains = ticker.capital_gains
```

### Options Data

```python
# Get available options expiration dates
expirations = ticker.options

# Get options chain for specific expiration date
options_chain = ticker.option_chain('YYYY-MM-DD')

# Access calls and puts separately
calls = options_chain.calls
puts = options_chain.puts
```

### Institutional Holders and Recommendations

```python
# Major holders
major_holders = ticker.major_holders  # Major holders breakdown
institutional_holders = ticker.institutional_holders  # Institutions
mutualfund_holders = ticker.mutualfund_holders  # Mutual funds

# Analyst recommendations
recommendations = ticker.recommendations
recommendations_summary = ticker.recommendations_summary

# Analyst price targets
target = ticker.target_price
upgrade_downgrade = ticker.upgrades_downgrades
```

### ESG Data and News

```python
# ESG (Environmental, Social, Governance) data
esg = ticker.sustainability

# Get recent company news
news = ticker.news
```

### Index and Mutual Fund Components

```python
# For indices or mutual funds, get underlying holdings
holdings = ticker.holdings
holding_performance = ticker.holding_performance
```

## Advanced Techniques & Examples

### Data Processing and Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get multiple stocks and calculate daily returns
tickers_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
data = yf.download(tickers_list, period='1y')['Adj Close']
returns = data.pct_change()

# Calculate correlation matrix
correlation = returns.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Stock Returns')
plt.show()
```

### Portfolio Backtesting

```python
# Simple portfolio backtesting example
portfolio = yf.download(['SPY', 'AAPL', 'MSFT', 'GOOG'], period='5y')['Adj Close']

# Calculate returns
returns = portfolio.pct_change().dropna()

# Equal weight portfolio
weights = [0.25, 0.25, 0.25, 0.25]
portfolio_return = (returns * weights).sum(axis=1)

# Cumulative returns
cumulative_returns = (1 + portfolio_return).cumprod() - 1

# Plot
plt.figure(figsize=(12, 6))
cumulative_returns.plot()
plt.title('Portfolio Cumulative Returns')
plt.ylabel('Return')
plt.grid(True)
plt.show()
```

### Technical Analysis Integration

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Get data
ticker = 'AAPL'
data = yf.download(ticker, period='1y')

# Calculate moving averages
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Calculate MACD
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Plot
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data['SMA_50'], label='50-day SMA', alpha=0.7)
plt.plot(data.index, data['SMA_200'], label='200-day SMA', alpha=0.7)
plt.title(f'{ticker} Price and Moving Averages')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data.index, data['MACD'], label='MACD')
plt.plot(data.index, data['Signal_Line'], label='Signal Line')
plt.bar(data.index, data['MACD'] - data['Signal_Line'], label='Histogram')
plt.title('MACD Indicator')
plt.legend()

plt.tight_layout()
plt.show()
```

### Working with Multiple Time Frames

```python
# Get data for multiple timeframes
ticker = 'AAPL'
daily_data = yf.download(ticker, period='1y', interval='1d')
weekly_data = yf.download(ticker, period='1y', interval='1wk')
monthly_data = yf.download(ticker, period='5y', interval='1mo')

# Calculate metrics for each timeframe
for df, timeframe in [(daily_data, 'Daily'), (weekly_data, 'Weekly'), (monthly_data, 'Monthly')]:
    df[f'{timeframe}_Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df[f'{timeframe}_RSI'] = calculate_rsi(df['Close'])  # implement or import RSI calculation

# Combine relevant metrics into a single dataframe for analysis
# [Implementation details will depend on specific analysis needs]
```

## Best Practices

### Error Handling

```python
try:
    data = yf.download("AAPL", period="1y")
    if data.empty:
        print("No data returned for ticker")
    else:
        # Process data
        pass
except Exception as e:
    print(f"Error occurred: {e}")
```

### Rate Limiting & Caching

```python
import time
import os
import pandas as pd

def get_stock_data(ticker, period="1y", interval="1d", cache_dir="cache"):
    """Get stock data with caching to avoid excessive API calls"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = f"{cache_dir}/{ticker}_{period}_{interval}.csv"
    
    # Check if cache exists and is recent (less than 1 day old)
    if os.path.exists(cache_file):
        file_mod_time = os.path.getmtime(cache_file)
        if (time.time() - file_mod_time) < 86400:  # 86400 seconds = 1 day
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # If no cache or old cache, download fresh data
    data = yf.download(ticker, period=period, interval=interval)
    
    # Save to cache
    if not data.empty:
        data.to_csv(cache_file)
    
    return data

# Example usage
aapl_data = get_stock_data("AAPL")
```

### Data Validation

```python
def validate_stock_data(data):
    """Validate downloaded stock data for common issues"""
    
    if data is None or data.empty:
        return False, "No data returned"
    
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Dataset contains {missing_values} missing values")
    
    # Check for stale data
    last_date = data.index[-1].date()
    today = pd.Timestamp.now().date()
    days_old = (today - last_date).days
    
    if days_old > 5:  # Assuming business days, allow for weekends
        print(f"Warning: Last data point is {days_old} days old")
    
    # Check for abnormal volumes (might indicate errors)
    avg_volume = data['Volume'].mean()
    if any(data['Volume'] > avg_volume * 100):
        print("Warning: Extremely high volume detected - verify data")
    
    return True, "Data validation passed with notes"

# Example
ticker = "AAPL"
data = yf.download(ticker, period="1mo")
is_valid, message = validate_stock_data(data)
```

### Performance Optimization

```python
# For multiple tickers, use download() with threads parameter
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "FB", "TSLA", "NFLX"]
data = yf.download(tickers, period="1y", threads=True)

# For large datasets, consider chunking your requests
def get_long_term_data(ticker, start_year, end_year):
    chunks = []
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"Downloading {ticker} data for {year}...")
        yearly_data = yf.download(ticker, start=start_date, end=end_date)
        chunks.append(yearly_data)
        time.sleep(1)  # Be polite to the API
    
    # Combine all yearly data
    full_data = pd.concat(chunks)
    return full_data

# Example
aapl_historical = get_long_term_data("AAPL", 2000, 2020)
```

## Common Issues & Solutions

### Missing Data Handling

```python
# Forward fill missing values
data = data.ffill()

# Or use interpolation
data = data.interpolate(method='linear')

# For financial ratios, it's often better to handle separately
financial_data = ticker.income_stmt
financial_data = financial_data.ffill(axis=1)  # Fill forward across time
```

### Adjusting for Splits and Dividends

```python
# Auto-adjust prices (default)
data = yf.download("AAPL", auto_adjust=True)  # Default is True

# Get raw data without adjustments
data_raw = yf.download("AAPL", auto_adjust=False)

# Get both adjusted close and unadjusted OHLC
data_with_adj = yf.download("AAPL", auto_adjust=False, actions=True)
```

### Workarounds for API Limitations

```python
# For limited daily API calls, implement caching (see Best Practices above)

# For more reliable fundamentals data, combine with basic web scraping
# Example with BeautifulSoup (requires installation)
import requests
from bs4 import BeautifulSoup

def get_sector_info(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/profile"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract and return sector info
    # [Implementation details]
```