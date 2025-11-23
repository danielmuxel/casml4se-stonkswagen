"""
Bitcoin Time Series Data Preparation and Training Pipeline
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

Usage:
    python bitcoin_ml_pipeline.py --data_path btc_5min.csv --epochs 100
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
import os

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

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


# ============================================================================
# PYTORCH DATASET & DATAMODULE
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series slices."""

    def __init__(self, slices: List[pd.DataFrame], labels: List[int],
                 feature_groups: Dict, shuffle: bool = False):
        self.slices = slices
        self.labels = labels
        self.feature_groups = feature_groups

        # CRITICAL: Only shuffle training set (paper Section 4)
        if shuffle:
            indices = np.random.permutation(len(self.slices))
            self.slices = [self.slices[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        # Scale slice independently
        scaled_slice = scale_slice(self.slices[idx], self.feature_groups)

        # Convert to tensors
        X = torch.FloatTensor(scaled_slice)
        y = torch.LongTensor([self.labels[idx]])

        return X, y.squeeze()


class TimeSeriesDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for time series."""

    def __init__(self, data_path: str, lookback: int = 20, horizon: int = 12,
                 batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define feature groups (paper Section 5)
        self.feature_groups = {
            'price_overlay': ['open', 'high', 'low', 'close', 'sma_5', 'sma_20',
                              'ema_10', 'bb_upper', 'bb_lower'],
            'volume': ['volume'],
            'bounded': ['rsi'],
            'unbounded': ['macd', 'macd_signal', 'atr']
        }

        # Calculate total features
        self.n_features = sum(len(v) for v in self.feature_groups.values())

    def setup(self, stage=None):
        # Load and prepare data
        df = load_and_validate_data(self.data_path)
        df = add_technical_indicators(df)

        # CRITICAL: Chronological split FIRST (paper Section 4)
        df_train, df_val, df_test = chronological_split(df)

        # Create slices for each split
        train_slices, train_labels, train_meta = create_slices_and_labels(
            df_train, self.lookback, self.horizon, self.feature_groups)
        val_slices, val_labels, val_meta = create_slices_and_labels(
            df_val, self.lookback, self.horizon, self.feature_groups)
        test_slices, test_labels, test_meta = create_slices_and_labels(
            df_test, self.lookback, self.horizon, self.feature_groups)

        # Balance training data only (paper Section 7)
        train_slices, train_labels, train_meta = balance_training_data(
            train_slices, train_labels, train_meta)

        # Create datasets (shuffle ONLY training - paper Section 4)
        self.train_dataset = TimeSeriesDataset(
            train_slices, train_labels, self.feature_groups, shuffle=True)
        self.val_dataset = TimeSeriesDataset(
            val_slices, val_labels, self.feature_groups, shuffle=False)
        self.test_dataset = TimeSeriesDataset(
            test_slices, test_labels, self.feature_groups, shuffle=False)

        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Val:   {len(self.val_dataset)}")
        print(f"  Test:  {len(self.test_dataset)}")

        # Log to MLflow
        mlflow.log_metric("train_slices", len(self.train_dataset))
        mlflow.log_metric("val_slices", len(self.val_dataset))
        mlflow.log_metric("test_slices", len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)


# ============================================================================
# SIMPLE LSTM MODEL
# ============================================================================

class SimpleLSTMClassifier(pl.LightningModule):
    """
    Simple LSTM classifier based on paper Section 3, Table 2.
    Paper uses: "1 LSTM 64 units (tanh), 1 Dense output 2 units (softmax)"

    We extend to 3 classes for QClass (Up/Neutral/Down).
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_classes: int = 3, learning_rate: float = 0.001,
                 dropout: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Store predictions for test set
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        # Classification
        logits = self.fc(last_output)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Store predictions and targets for confusion matrix
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'test_acc': acc}

    def on_test_epoch_end(self):
        # Calculate and log per-class metrics
        target_names = ['Up', 'Neutral', 'Down']
        report = classification_report(self.test_targets, self.test_predictions,
                                       target_names=target_names, output_dict=True)

        # Log per-class metrics to MLflow
        for class_name in target_names:
            mlflow.log_metric(f"test_{class_name.lower()}_precision", report[class_name]['precision'])
            mlflow.log_metric(f"test_{class_name.lower()}_recall", report[class_name]['recall'])
            mlflow.log_metric(f"test_{class_name.lower()}_f1", report[class_name]['f1-score'])

        # Log overall metrics
        mlflow.log_metric("test_accuracy", report['accuracy'])
        mlflow.log_metric("test_macro_avg_f1", report['macro avg']['f1-score'])
        mlflow.log_metric("test_weighted_avg_f1", report['weighted avg']['f1-score'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bitcoin Time Series ML Pipeline with MLflow logging',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getenv('DATA_PATH'), 'crypto', 'bitcoin.csv'),
                        help='Path to CSV file with columns: timestamp,open,high,low,close,volume')
    parser.add_argument('--experiment_name', type=str, default='bitcoin_timeseries2',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name (optional)')
    parser.add_argument('--lookback', type=int, default=20,
                        help='Lookback period (number of bars per slice)')
    parser.add_argument('--horizon', type=int, default=12,
                        help='Prediction horizon (bars ahead to predict)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=40,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set random seeds
    pl.seed_everything(args.seed)

    print("=" * 80)
    print("Bitcoin Time Series ML Pipeline with MLflow Tracking")
    print("Based on: 'Financial Time Series Data Processing for Machine Learning'")
    print("=" * 80)
    print(f"\nMLflow Configuration:")
    print(f"  Tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
    print(f"  Experiment Name: {args.experiment_name}")
    print("=" * 80)

    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)



    # ========================================================================
    # START MLFLOW RUN (AFTER preprocessing is done)
    # ========================================================================
    # Setup MLflow logger for PyTorch Lightning

    if mlflow.active_run():
        print("\nWarning: Found active MLflow run. Ending it before starting new run...")
        mlflow.end_run()
    mlflow.start_run(run_name=args.run_name)

    start_time = time.time()
    mlflow.log_metric("start_timestamp", start_time)

    mlflow.pytorch.autolog()
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    print(f"\nMLflow Run ID: {mlflow_logger.run_id}")
    print(f"MLflow Run Name: {args.run_name}")

    print("\n" + "=" * 80)
    print("Starting data preprocessing...")
    print("=" * 80)

    # Initialize data module
    data_module = TimeSeriesDataModule(
        data_path=args.data_path,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Setup data (this runs all preprocessing) - MOVED OUTSIDE MLFLOW RUN
    # This is where the long-running preprocessing happens
    data_module.setup()

    print("\n" + "=" * 80)
    print("Preprocessing completed!")
    print("=" * 80)



    # Log all hyperparameters
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("lookback", args.lookback)
    mlflow.log_param("horizon", args.horizon)
    mlflow.log_param("hidden_size", args.hidden_size)
    mlflow.log_param("dropout", args.dropout)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("max_epochs", args.epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("num_workers", args.num_workers)

    # Log paper methodology
    mlflow.log_param("scaling_method", "standardization")
    mlflow.log_param("scaling_strategy", "per_slice_independent")
    mlflow.log_param("split_method", "chronological")
    mlflow.log_param("label_type", "qclass")
    mlflow.log_param("class_balancing", "undersampling")

    # Log feature information
    mlflow.log_param("n_features", data_module.n_features)
    mlflow.log_param("feature_groups", str(list(data_module.feature_groups.keys())))
    mlflow.log_param("price_overlay_features", len(data_module.feature_groups['price_overlay']))
    mlflow.log_param("volume_features", len(data_module.feature_groups['volume']))
    mlflow.log_param("bounded_features", len(data_module.feature_groups['bounded']))
    mlflow.log_param("unbounded_features", len(data_module.feature_groups['unbounded']))

    # Log dataset sizes (these were printed during preprocessing)
    mlflow.log_metric("train_slices", len(data_module.train_dataset))
    mlflow.log_metric("val_slices", len(data_module.val_dataset))
    mlflow.log_metric("test_slices", len(data_module.test_dataset))

    # Initialize model
    model = SimpleLSTMClassifier(
        n_features=data_module.n_features,
        hidden_size=args.hidden_size,
        num_classes=3,  # QClass: Up, Neutral, Down
        learning_rate=args.learning_rate,
        dropout=args.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:")
    print(f"  Input features: {data_module.n_features}")
    print(f"  LSTM hidden size: {args.hidden_size}")
    print(f"  Output classes: 3 (Up/Neutral/Down)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Log model info
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("num_classes", 3)
    mlflow.log_param("total_parameters", total_params)
    mlflow.log_param("trainable_parameters", trainable_params)



    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='bitcoin-lstm-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=mlflow_logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.fit(model, data_module)

    # Log best model path
    mlflow.log_param("best_model_path", checkpoint_callback.best_model_path)
    mlflow.log_metric("best_val_loss", checkpoint_callback.best_model_score.item())

    # Test
    print("\n" + "=" * 80)
    print("Testing on held-out test set...")
    print("=" * 80)
    test_results = trainer.test(model, data_module, ckpt_path='best')

    # Generate classification report
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    target_names = ['Up (0)', 'Neutral (1)', 'Down (2)']
    report = classification_report(model.test_targets, model.test_predictions,
                                   target_names=target_names)
    print(report)

    # Save classification report as artifact
    report_path = "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Generate and log confusion matrix
    print("\nConfusion Matrix:")
    print("=" * 80)
    cm = confusion_matrix(model.test_targets, model.test_predictions)
    print("Predicted ->")
    print(f"           Up  Neutral  Down")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:8s} {row[0]:5d}  {row[1]:5d}  {row[2]:5d}")

    # Save confusion matrix as artifact
    cm_df = pd.DataFrame(cm,
                         index=['True Up', 'True Neutral', 'True Down'],
                         columns=['Pred Up', 'Pred Neutral', 'Pred Down'])
    cm_path = "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    mlflow.log_artifact(cm_path)

    # Log model to MLflow
    print("\nLogging model to MLflow...")
    mlflow.pytorch.log_model(model, "model")

    # Log best checkpoint as artifact
    if checkpoint_callback.best_model_path:
        mlflow.log_artifact(checkpoint_callback.best_model_path, "checkpoints")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"MLflow Run ID: {mlflow_logger.run_id}")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"View results at: {os.getenv('MLFLOW_TRACKING_URI', 'MLflow UI')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
