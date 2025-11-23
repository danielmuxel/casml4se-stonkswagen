# Bitcoin Time Series ML Pipeline
Note: will be migrated away from notebook folder, once rework is done. 

Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

## Project Structure

The code has been refactored into modular components for better readability and maintainability:

```
daniel_fabrice_wp/
├── __init__.py           # Package initialization and exports
├── data_preparation.py   # Data loading, validation, and preprocessing
├── dataset.py            # PyTorch Dataset and DataModule
├── model.py              # LSTM classifier model
├── experiment.py         # Main training script
└── README.md             # This file
```

## Module Overview

### `data_preparation.py`
Handles all data preparation tasks:
- **Data Loading**: `load_and_validate_data()` - Loads OHLC data with validation
- **Technical Indicators**: 
  - `add_technical_indicators()` - Adds SMA, EMA, Bollinger Bands, RSI, MACD, ATR
  - `calculate_rsi()`, `calculate_macd()`, `calculate_atr()` - Individual indicator calculations
- **Data Splitting**: `chronological_split()` - Splits data into train/val/test chronologically
- **Slice Creation**: `create_slices_and_labels()` - Creates time series slices with labels
- **Label Calculation**: `calculate_qclass_label()` - Calculates QClass labels (Up/Neutral/Down)
- **Data Balancing**: `balance_training_data()` - Balances classes via undersampling
- **Scaling**: `scale_slice()` - Scales each slice independently with feature-group-specific methods

### `dataset.py`
PyTorch data handling:
- **TimeSeriesDataset**: PyTorch Dataset for time series slices
- **TimeSeriesDataModule**: PyTorch Lightning DataModule that orchestrates the entire data pipeline

### `model.py`
Model architecture:
- **SimpleLSTMClassifier**: LSTM-based classifier for 3-class prediction (Up/Neutral/Down)
  - Implements training, validation, and test steps
  - Includes metric logging to MLflow
  - Learning rate scheduling with ReduceLROnPlateau

### `experiment.py`
Main training script:
- Command-line argument parsing
- MLflow experiment tracking setup
- Model training orchestration
- Evaluation and reporting
- Artifact logging (model, checkpoints, reports)

## Usage

### As a Script
```bash
python -m notebooks.models.daniel_fabrice_wp.experiment \
    --data_path path/to/bitcoin.csv \
    --epochs 250 \
    --lookback 20 \
    --horizon 12 \
    --batch_size 64 \
    --learning_rate 0.001
```

### As a Module
```python
from notebooks.models.daniel_fabrice_wp import (
    TimeSeriesDataModule,
    SimpleLSTMClassifier,
    load_and_validate_data,
    add_technical_indicators
)

# Load and prepare data
df = load_and_validate_data('bitcoin.csv')
df = add_technical_indicators(df)

# Create data module
data_module = TimeSeriesDataModule(
    data_path='bitcoin.csv',
    lookback=20,
    horizon=12,
    batch_size=64
)
data_module.setup()

# Create model
model = SimpleLSTMClassifier(
    n_features=data_module.n_features,
    hidden_size=64,
    learning_rate=0.001
)
```

## Key Features

1. **Modular Design**: Each component has a clear, single responsibility
2. **Paper Implementation**: Follows the methodology from Fabrice Daniel's 2019 paper
3. **MLflow Integration**: Complete experiment tracking and artifact logging
4. **Type Hints**: Full type annotations for better code documentation
5. **Documentation**: Comprehensive docstrings explaining each function

## Methodology

The implementation follows these key principles from the paper:

1. **Independent Slice Scaling**: Each time series slice is scaled independently
2. **Feature Group Scaling**: Different feature types use appropriate scaling methods:
   - Price + Overlays: Standardization (together)
   - Volume: Standardization (separate)
   - Bounded indicators (RSI): Division by 100
   - Unbounded indicators (MACD, ATR): Standardization (separate)
3. **Chronological Split**: Data is split chronologically to prevent look-ahead bias
4. **QClass Labels**: Three-class labels based on price movement quantiles
5. **Class Balancing**: Training data is balanced via undersampling

## Dependencies

- PyTorch
- PyTorch Lightning
- pandas
- numpy
- scikit-learn
- MLflow
- python-dotenv

## Configuration

Set the following environment variables in `.env`:
- `DATA_PATH`: Base path to data directory
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
