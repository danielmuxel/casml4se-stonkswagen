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
├── model.py              # LSTM classifier models
├── experiment.py         # Main training script
├── gridsearch.py         # Hyperparameter tuning via grid search
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
Model architectures:
- **SimpleLSTMClassifier**: LSTM-based classifier for 3-class prediction (Up/Neutral/Down)
  - Implements training, validation, and test steps
  - Includes metric logging to MLflow
  - Learning rate scheduling with ReduceLROnPlateau
- **AdvancedLSTMWithAttention**: Advanced LSTM classifier with intelligent features
  - Bidirectional LSTM layers for capturing past and future context
  - Multi-head self-attention mechanism for temporal dependencies
  - Residual connections and layer normalization
  - 1D convolution for adaptive feature extraction
  - Advanced regularization (dropout, layer dropout, label smoothing)
  - AdamW optimizer with Cosine Annealing warm restarts
  - Significantly more parameters and capacity than SimpleLSTMClassifier

### `experiment.py`
Main training script:
- Command-line argument parsing
- MLflow experiment tracking setup
- Model training orchestration
- Evaluation and reporting
- Artifact logging (model, checkpoints, reports)

### `gridsearch.py`
Hyperparameter optimization:
- **GridSearchCV**: Class for exhaustive grid search over parameter combinations
- **run_gridsearch()**: Convenience function to run grid search with default or custom parameter grids
- **get_default_param_grid()**: Returns sensible default parameter grids for each model type
- MLflow tracking for all experiments
- Automatic best model selection based on validation accuracy
- Results exported to CSV for analysis

## Usage

### As a Script

#### Using SimpleLSTMClassifier
```bash
python -m notebooks.models.daniel_fabrice_wp.experiment \
    --data_path path/to/bitcoin.csv \
    --model_type simple \
    --epochs 250 \
    --lookback 20 \
    --horizon 12 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_size 64 \
    --num_layers 3 \
    --dropout 0.2
```

#### Using AdvancedLSTMWithAttention
```bash
python -m notebooks.models.daniel_fabrice_wp.experiment \
    --data_path path/to/bitcoin.csv \
    --model_type advanced \
    --epochs 250 \
    --lookback 20 \
    --horizon 12 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_attention_heads 4 \
    --dropout 0.3 \
    --layer_dropout 0.1 \
    --weight_decay 1e-5
```

### As a Module

#### Using SimpleLSTMClassifier
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

# Create simple model
model = SimpleLSTMClassifier(
    n_features=data_module.n_features,
    hidden_size=64,
    num_layers=3,
    learning_rate=0.001,
    dropout=0.2
)
```

#### Using AdvancedLSTMWithAttention
```python
from notebooks.models.daniel_fabrice_wp import (
    TimeSeriesDataModule,
    AdvancedLSTMWithAttention
)

# Create data module
data_module = TimeSeriesDataModule(
    data_path='bitcoin.csv',
    lookback=20,
    horizon=12,
    batch_size=64
)
data_module.setup()

# Create advanced model with attention
model = AdvancedLSTMWithAttention(
    n_features=data_module.n_features,
    hidden_size=128,  # Must be divisible by num_attention_heads
    num_layers=2,
    num_attention_heads=4,
    dropout=0.3,
    layer_dropout=0.1,
    learning_rate=0.001,
    weight_decay=1e-5,
    use_conv=True,
    conv_kernel_size=3
)
```

### Hyperparameter Tuning with Grid Search

#### Quick Start - Using Default Parameter Grids
```python
from notebooks.models.daniel_fabrice_wp import run_gridsearch

# Run grid search for SimpleLSTMClassifier with default parameters
grid_search = run_gridsearch(
    data_path='path/to/bitcoin.csv',
    model_type='simple',
    epochs=100,
    experiment_name='bitcoin_gridsearch'
)

# View best parameters
print(f"Best Parameters: {grid_search.best_params}")
print(f"Best Validation Accuracy: {grid_search.best_score:.4f}")

# Get results as DataFrame
results_df = grid_search.get_results_dataframe()
print(results_df.head(10))
```

#### Custom Parameter Grid - SimpleLSTMClassifier
```python
from notebooks.models.daniel_fabrice_wp import run_gridsearch

# Define custom parameter grid
param_grid = {
    'hidden_size': [32, 64, 128, 256],
    'num_layers': [2, 3, 4, 5],
    'dropout': [0.1, 0.2, 0.3, 0.4],
    'learning_rate': [0.0001, 0.001, 0.01]
}

# Run grid search
grid_search = run_gridsearch(
    data_path='path/to/bitcoin.csv',
    model_type='simple',
    param_grid=param_grid,
    lookback=20,
    horizon=12,
    batch_size=64,
    epochs=100,
    num_workers=4
)

# Access best model run ID (for loading from MLflow)
best_run_id = grid_search.best_run_id
```

#### Custom Parameter Grid - AdvancedLSTMWithAttention
```python
from notebooks.models.daniel_fabrice_wp import run_gridsearch

# Define custom parameter grid for advanced model
param_grid = {
    'hidden_size': [128, 256, 512],
    'num_layers': [2, 3],
    'num_attention_heads': [4, 8, 16],
    'dropout': [0.2, 0.3, 0.4],
    'layer_dropout': [0.1, 0.2],
    'learning_rate': [0.0001, 0.001],
    'weight_decay': [1e-5, 1e-4, 1e-3]
}

# Run grid search
grid_search = run_gridsearch(
    data_path='path/to/bitcoin.csv',
    model_type='advanced',
    param_grid=param_grid,
    epochs=100
)
```

#### Advanced Usage - GridSearchCV Class
```python
from notebooks.models.daniel_fabrice_wp import GridSearchCV, get_default_param_grid

# Create grid search object
grid_search = GridSearchCV(
    model_type='simple',
    data_path='path/to/bitcoin.csv',
    experiment_name='my_custom_experiment',
    parent_run_name='gridsearch_run_1',
    seed=42
)

# Get default parameter grid and modify it
param_grid = get_default_param_grid('simple')
param_grid['hidden_size'] = [64, 128]  # Reduce search space

# Fit grid search
grid_search.fit(
    param_grid=param_grid,
    lookback=20,
    horizon=12,
    batch_size=64,
    epochs=100,
    num_workers=4
)

# Access results
print(f"Best score: {grid_search.best_score}")
print(f"Best params: {grid_search.best_params}")
print(f"Best run ID: {grid_search.best_run_id}")

# Get all results
results_df = grid_search.get_results_dataframe()
results_df.to_csv('my_gridsearch_results.csv', index=False)
```

#### Default Parameter Grids

**SimpleLSTMClassifier (default grid)**:
- `hidden_size`: [32, 64, 128]
- `num_layers`: [2, 3, 4]
- `dropout`: [0.1, 0.2, 0.3]
- `learning_rate`: [0.0001, 0.001, 0.01]
- **Total combinations**: 108

**AdvancedLSTMWithAttention (default grid)**:
- `hidden_size`: [64, 128, 256]
- `num_layers`: [2, 3]
- `num_attention_heads`: [4, 8]
- `dropout`: [0.2, 0.3, 0.4]
- `layer_dropout`: [0.1, 0.2]
- `learning_rate`: [0.0001, 0.001]
- `weight_decay`: [1e-5, 1e-4]
- **Total combinations**: 288

#### Grid Search Features

1. **MLflow Integration**: All experiments are logged to MLflow with nested runs
2. **Automatic Best Model Selection**: Selects best model based on validation accuracy
3. **Results Export**: Exports all results to CSV for further analysis
4. **Parameter Validation**: Automatically validates and adjusts parameters (e.g., ensures `hidden_size` is divisible by `num_attention_heads`)
5. **Early Stopping**: Each model uses early stopping to prevent overfitting
6. **Progress Tracking**: Shows progress and validation accuracy for each combination
7. **Error Handling**: Continues grid search even if individual models fail

#### Running Grid Search from Command Line
```bash
# Run grid search module directly
python -m notebooks.models.daniel_fabrice_wp.gridsearch
```

This will run example grid searches for both model types with reduced epochs for demonstration.

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
