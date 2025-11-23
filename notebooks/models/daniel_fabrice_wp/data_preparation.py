"""
Data Preparation Module
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

This module handles:
- Data loading and validation
- Technical indicators calculation
- Chronological data splitting
- Slice creation and labeling
- Data balancing
"""

import numpy as np
import pandas as pd
import mlflow
from typing import Tuple, Dict, List


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Load OHLC data and perform basic validation.
    Expected columns: timestamp, open, high, low, close, volume
    """
    print("Loading data...")
    df = pd.read_csv(file_path, delimiter=";")

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Validate OHLC consistency
    assert (df['low'] <= df['high']).all(), "Low > High detected!"
    assert (df['low'] <= df['close']).all(), "Low > Close detected!"
    assert (df['low'] <= df['open']).all(), "Low > Open detected!"
    assert (df['close'] <= df['high']).all(), "Close > High detected!"
    assert (df['open'] <= df['high']).all(), "Open > High detected!"
    assert (df['volume'] >= 0).all(), "Negative volume detected!"

    print(f"Data loaded: {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Log data info to MLflow
    mlflow.log_param("data_rows", len(df))
    mlflow.log_param("data_start", str(df['timestamp'].min()))
    mlflow.log_param("data_end", str(df['timestamp'].max()))
    mlflow.log_param("data_timespan_days", (df['timestamp'].max() - df['timestamp'].min()).days)

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator (0-100 range)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and signal line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators following the paper's recommendations.
    Overlaid indicators: Scale WITH prices
    Separated indicators: Scale INDEPENDENTLY
    """
    df = df.copy()

    # Overlaid Indicators (will be scaled with prices)
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)

    # Bounded Indicators (0-100 range)
    df['rsi'] = calculate_rsi(df['close'], period=14)

    # Unbounded Indicators
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['atr'] = calculate_atr(df, period=14)

    # Drop NaN rows from indicator calculation
    df = df.dropna().reset_index(drop=True)

    print(f"Technical indicators added. Rows after dropna: {len(df)}")
    mlflow.log_param("data_rows_after_indicators", len(df))

    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70,
                        val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CRITICAL: Split data chronologically BEFORE creating slices.
    Paper Section 4: "Split into training/validation/test sets"
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"\nChronological Split:")
    print(f"  Train: {len(df_train)} rows ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
    print(f"  Val:   {len(df_val)} rows ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
    print(f"  Test:  {len(df_test)} rows ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")

    # Log split info to MLflow
    mlflow.log_param("train_rows", len(df_train))
    mlflow.log_param("val_rows", len(df_val))
    mlflow.log_param("test_rows", len(df_test))
    mlflow.log_param("train_start", str(df_train['timestamp'].min()))
    mlflow.log_param("train_end", str(df_train['timestamp'].max()))
    mlflow.log_param("val_start", str(df_val['timestamp'].min()))
    mlflow.log_param("val_end", str(df_val['timestamp'].max()))
    mlflow.log_param("test_start", str(df_test['timestamp'].min()))
    mlflow.log_param("test_end", str(df_test['timestamp'].max()))

    return df_train, df_val, df_test


def calculate_qclass_label(df: pd.DataFrame, idx: int, horizon: int) -> int:
    """
    Calculate QClass label based on %Q metric from paper Section 6.
    %Q = (HH - Close_t) / (HH - LL)

    QClass:
        0 (Up): %Q >= 0.6
        1 (Neutral): 0.4 < %Q < 0.6
        2 (Down): %Q <= 0.4
    """
    if idx + horizon >= len(df):
        return None

    close_t = df.iloc[idx]['close']
    future_window = df.iloc[idx + 1:idx + horizon + 1]

    HH = future_window['high'].max()
    LL = future_window['low'].min()

    if HH == LL:  # Avoid division by zero
        return 1  # Neutral

    pct_q = (HH - close_t) / (HH - LL)

    # Convert to QClass
    if pct_q >= 0.6:
        return 0  # Up
    elif pct_q <= 0.4:
        return 2  # Down
    else:
        return 1  # Neutral


def create_slices_and_labels(df: pd.DataFrame, lookback: int, horizon: int,
                             feature_groups: Dict) -> Tuple[List, List, List]:
    """
    Create overlapping slices and labels.
    Paper Section 3: "Building a training set S consists on creating a series of K slices"
    """
    all_features = (feature_groups['price_overlay'] +
                    feature_groups['volume'] +
                    feature_groups['bounded'] +
                    feature_groups['unbounded'])

    slices = []
    labels = []
    metadata = []

    total_iterations = len(df) - horizon - lookback
    print(f"Creating slices: lookback={lookback}, horizon={horizon}")
    print(f"Total potential slices: {total_iterations}")

    # Create slices with lookback and prediction horizon
    skipped = 0
    for i in range(lookback, len(df) - horizon):
        # Print progress every 10% of iterations
        if (i - lookback) % max(1, total_iterations // 10) == 0:
            progress = ((i - lookback) / total_iterations) * 100
            print(f"  Progress: {progress:.1f}% ({i - lookback}/{total_iterations} slices processed)")

        # Extract slice (lookback periods)
        slice_df = df.iloc[i - lookback:i][all_features].copy()

        # Calculate label (QClass from paper Section 6)
        label = calculate_qclass_label(df, i, horizon)

        if label is not None:  # Skip if can't calculate label
            slices.append(slice_df)
            labels.append(label)
            metadata.append({
                'start_idx': i - lookback,
                'end_idx': i,
                'target_idx': i + horizon,
                'timestamp': df.iloc[i]['timestamp']
            })
        else:
            skipped += 1

    print(f"Created {len(slices)} slices (skipped {skipped} due to invalid labels)")

    # Log class distribution
    if labels:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution:")
        class_names = {0: 'Up', 1: 'Neutral', 2: 'Down'}
        for cls, count in zip(unique, counts):
            print(f"  {class_names.get(cls, cls)}: {count} ({count / len(labels) * 100:.1f}%)")

    return slices, labels, metadata


def balance_training_data(slices: List, labels: List, metadata: List,
                          target_ratio: float = 2.0) -> Tuple[List, List, List]:
    """
    Balance classes by undersampling majority class.
    Paper Section 7: "The simplest way to fight this consists on downsampling
    the Up labels on the training set"
    """
    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)

    print(f"\nOriginal class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count / len(labels) * 100:.1f}%)")
        mlflow.log_metric(f"train_class_{cls}_original_count", count)
        mlflow.log_metric(f"train_class_{cls}_original_pct", count / len(labels) * 100)

    # Find minority class count
    min_count = counts.min()
    target_count = int(min_count * target_ratio)

    balanced_indices = []
    for cls in unique:
        cls_indices = np.where(labels_array == cls)[0]
        if len(cls_indices) > target_count:
            # Undersample
            sampled_indices = np.random.choice(cls_indices, target_count, replace=False)
        else:
            sampled_indices = cls_indices
        balanced_indices.extend(sampled_indices)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    balanced_slices = [slices[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    balanced_metadata = [metadata[i] for i in balanced_indices]

    # Show new distribution
    balanced_labels_array = np.array(balanced_labels)
    unique, counts = np.unique(balanced_labels_array, return_counts=True)
    print(f"\nBalanced class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count / len(balanced_labels) * 100:.1f}%)")
        mlflow.log_metric(f"train_class_{cls}_balanced_count", count)
        mlflow.log_metric(f"train_class_{cls}_balanced_pct", count / len(balanced_labels) * 100)

    return balanced_slices, balanced_labels, balanced_metadata


def scale_slice(slice_df: pd.DataFrame, feature_groups: Dict) -> np.ndarray:
    """
    Scale each slice INDEPENDENTLY following paper Section 3.
    CRITICAL: "Scaling each slice independently can make the training easier"

    Different scaling for different feature groups:
    - Price + Overlays: Standardization (together)
    - Volume: Standardization (separate)
    - Bounded (RSI): Divide by 100
    - Unbounded (MACD, ATR): Standardization (separate)
    """
    scaled_features = []

    # Group 1: Price + Overlay indicators (scale together)
    price_overlay_data = slice_df[feature_groups['price_overlay']].values
    mean = price_overlay_data.mean()
    std = price_overlay_data.std()
    if std > 0:
        scaled_price = (price_overlay_data - mean) / std
    else:
        scaled_price = price_overlay_data - mean
    scaled_features.append(scaled_price)

    # Group 2: Volume (scale separately)
    volume_data = slice_df[feature_groups['volume']].values
    mean = volume_data.mean()
    std = volume_data.std()
    if std > 0:
        scaled_volume = (volume_data - mean) / std
    else:
        scaled_volume = volume_data - mean
    scaled_features.append(scaled_volume)

    # Group 3: Bounded indicators (divide by 100)
    if feature_groups['bounded']:
        bounded_data = slice_df[feature_groups['bounded']].values
        scaled_bounded = bounded_data / 100.0
        scaled_features.append(scaled_bounded)

    # Group 4: Unbounded indicators (scale separately)
    if feature_groups['unbounded']:
        unbounded_data = slice_df[feature_groups['unbounded']].values
        mean = unbounded_data.mean()
        std = unbounded_data.std()
        if std > 0:
            scaled_unbounded = (unbounded_data - mean) / std
        else:
            scaled_unbounded = unbounded_data - mean
        scaled_features.append(scaled_unbounded)

    # Concatenate all scaled features
    scaled_slice = np.concatenate(scaled_features, axis=1)
    return scaled_slice
