from datetime import datetime

from data_retriever import DataRetriever

# Create retriever
retriever = DataRetriever()

# Get BTC data
df = retriever.get_data(
    symbol="BTC/USDT",
    source="BINANCE",
    start_time=datetime(2015, 1, 1),
    end_time=datetime(2025, 10, 25),
    interval="15m",
)

print(f"Got {len(df)} rows")
print(df.head())
