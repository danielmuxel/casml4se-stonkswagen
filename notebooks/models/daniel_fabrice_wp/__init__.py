"""
Bitcoin Time Series ML Pipeline
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

This package provides modular components for time series forecasting:
- data_preparation: Data loading, validation, and preprocessing
- dataset: PyTorch Dataset and DataModule
- model: LSTM classifier model
- experiment: Main training script
"""

from data_preparation import (
    load_and_validate_data,
    add_technical_indicators,
    chronological_split,
    create_slices_and_labels,
    balance_training_data,
    scale_slice,
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_qclass_label
)

from dataset import TimeSeriesDataset, TimeSeriesDataModule
from model import SimpleLSTMClassifier, AdvancedLSTMWithAttention
from gridsearch import GridSearchCV, run_gridsearch, get_default_param_grid

__all__ = [
    # Data preparation functions
    'load_and_validate_data',
    'add_technical_indicators',
    'chronological_split',
    'create_slices_and_labels',
    'balance_training_data',
    'scale_slice',
    'calculate_rsi',
    'calculate_macd',
    'calculate_atr',
    'calculate_qclass_label',
    # Dataset classes
    'TimeSeriesDataset',
    'TimeSeriesDataModule',
    # Model classes
    'SimpleLSTMClassifier',
    'AdvancedLSTMWithAttention',
    # Grid search classes and functions
    'GridSearchCV',
    'run_gridsearch',
    'get_default_param_grid',
]
