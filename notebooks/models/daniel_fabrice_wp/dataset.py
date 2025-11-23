"""
PyTorch Dataset and DataModule
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

This module handles:
- PyTorch Dataset for time series slices
- PyTorch Lightning DataModule for data management
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import mlflow
from typing import List, Dict

from data_preparation import (
    load_and_validate_data,
    add_technical_indicators,
    chronological_split,
    create_slices_and_labels,
    balance_training_data,
    scale_slice
)


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
