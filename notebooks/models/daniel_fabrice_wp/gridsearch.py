"""
Grid Search Module for Hyperparameter Tuning
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

This module provides grid search functionality for hyperparameter optimization:
- Exhaustive grid search over parameter combinations
- MLflow tracking for all experiments
- Best model selection based on validation metrics
- Support for both SimpleLSTMClassifier and AdvancedLSTMWithAttention
"""

import os
import itertools
import time
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from dataset import TimeSeriesDataModule
from model import SimpleLSTMClassifier, AdvancedLSTMWithAttention


class GridSearchCV:
    """
    Grid Search Cross Validation for LSTM models.
    
    This class performs exhaustive search over specified parameter values
    for an estimator, logging all experiments to MLflow.
    """
    
    def __init__(
        self,
        model_type: str = 'simple',
        data_path: str = None,
        experiment_name: str = 'bitcoin_gridsearch',
        parent_run_name: str = None,
        seed: int = 42
    ):
        """
        Initialize Grid Search.
        
        Args:
            model_type: Type of model to use ('simple' or 'advanced')
            data_path: Path to data file
            experiment_name: MLflow experiment name
            parent_run_name: Name for parent MLflow run
            seed: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.parent_run_name = parent_run_name or f"gridsearch_hp2_{model_type}_{int(time.time())}"
        self.seed = seed
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
        self.best_run_id = None
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
    def _create_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Create all combinations of parameters from grid.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values
            
        Returns:
            List of parameter dictionaries
        """
        # Get all parameter names and their values
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Create all combinations
        combinations = list(itertools.product(*values))
        
        # Convert to list of dictionaries
        param_combinations = [
            dict(zip(keys, combination))
            for combination in combinations
        ]
        
        return param_combinations
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and adjust parameters if needed.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Validated parameter dictionary
        """
        validated = params.copy()
        
        # For advanced model, ensure hidden_size is divisible by num_attention_heads
        if self.model_type == 'advanced':
            if 'hidden_size' in validated and 'num_attention_heads' in validated:
                hidden_size = validated['hidden_size']
                num_heads = validated['num_attention_heads']
                
                if hidden_size % num_heads != 0:
                    # Adjust hidden_size to be divisible
                    adjusted_hidden_size = ((hidden_size // num_heads) + 1) * num_heads
                    print(f"  Warning: Adjusting hidden_size from {hidden_size} to {adjusted_hidden_size}")
                    validated['hidden_size'] = adjusted_hidden_size
        
        return validated
    
    def _train_single_model(
        self,
        params: Dict[str, Any],
        data_module: TimeSeriesDataModule,
        epochs: int = 100
    ) -> Tuple[float, str]:
        """
        Train a single model with given parameters.
        
        Args:
            params: Model parameters
            data_module: Data module
            epochs: Maximum number of epochs
            
        Returns:
            Tuple of (validation score, run_id)
        """
        # Start nested MLflow run
        with mlflow.start_run(nested=True) as run:
            run_id = run.info.run_id
            
            # Log all parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Enable autolog for PyTorch
            mlflow.pytorch.autolog()
            
            # Create MLflow logger with active run info
            mlflow_logger = MLFlowLogger(
                experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
                tracking_uri=mlflow.get_tracking_uri(),
                run_id=mlflow.active_run().info.run_id
            )
            
            # Initialize model
            if self.model_type == 'simple':
                model = SimpleLSTMClassifier(
                    n_features=data_module.n_features,
                    hidden_size=params.get('hidden_size', 64),
                    num_classes=3,
                    num_layers=params.get('num_layers', 3),
                    learning_rate=params.get('learning_rate', 0.001),
                    dropout=params.get('dropout', 0.2)
                )
            elif self.model_type == 'advanced':
                model = AdvancedLSTMWithAttention(
                    n_features=data_module.n_features,
                    hidden_size=params.get('hidden_size', 128),
                    num_classes=3,
                    num_layers=params.get('num_layers', 2),
                    num_attention_heads=params.get('num_attention_heads', 4),
                    dropout=params.get('dropout', 0.3),
                    layer_dropout=params.get('layer_dropout', 0.1),
                    learning_rate=params.get('learning_rate', 0.001),
                    weight_decay=params.get('weight_decay', 1e-5),
                    use_conv=params.get('use_conv', True),
                    conv_kernel_size=params.get('conv_kernel_size', 3)
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            
            # Callbacks
            checkpoint_callback = ModelCheckpoint(
                monitor='val_acc',
                dirpath=f'checkpoints/gridsearch/{run_id}',
                filename='model-{epoch:02d}-{val_acc:.4f}',
                save_top_k=1,
                mode='max'
            )
            
            early_stop_callback = EarlyStopping(
                monitor='val_acc',
                patience=10,
                mode='max',
                verbose=True
            )
            
            # Trainer
            trainer = pl.Trainer(
                max_epochs=epochs,
                callbacks=[checkpoint_callback, early_stop_callback],
                logger=mlflow_logger,
                accelerator='auto',
                devices=1,
                log_every_n_steps=10,
                deterministic=True,
                enable_progress_bar=True,
                enable_model_summary=True
            )
            
            # Train
            trainer.fit(model, data_module)
            
            # Get validation score
            val_results = trainer.validate(model, data_module, ckpt_path='best', verbose=False)
            val_acc = val_results[0]['val_acc']
            
            # Log final metrics
            mlflow.log_metric("final_val_acc", val_acc)
            mlflow.log_metric("best_epoch", checkpoint_callback.best_model_score)
            
            return val_acc, run_id
    
    def fit(
        self,
        param_grid: Dict[str, List[Any]],
        lookback: int = 20,
        horizon: int = 12,
        batch_size: int = 64,
        epochs: int = 100,
        num_workers: int = 4
    ) -> "GridSearchCV":
        """
        Fit grid search.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values
            lookback: Lookback period
            horizon: Prediction horizon
            batch_size: Batch size
            epochs: Maximum number of epochs per model
            num_workers: Number of dataloader workers
            
        Returns:
            self
        """
        # Ensure no active MLflow runs from previous executions
        if mlflow.active_run():
            print("Warning: Closing active MLflow run before starting grid search...")
            mlflow.end_run()
        
        pl.seed_everything(self.seed)
        
        # Create parameter combinations
        param_combinations = self._create_param_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        print("=" * 80)
        print(f"Grid Search: {self.model_type} model")
        print(f"Total combinations to evaluate: {total_combinations}")
        print("=" * 80)
        
        # Start parent MLflow run
        with mlflow.start_run(run_name=self.parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            
            # Initialize data module (shared across all runs)
            # This must be done inside the MLflow run context as it logs to MLflow
            print("\nInitializing data module...")
            data_module = TimeSeriesDataModule(
                data_path=self.data_path,
                lookback=lookback,
                horizon=horizon,
                batch_size=batch_size,
                num_workers=num_workers
            )
            data_module.setup()
            # Log grid search configuration
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("total_combinations", total_combinations)
            mlflow.log_param("lookback", lookback)
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("max_epochs", epochs)
            mlflow.log_param("seed", self.seed)
            
            # Evaluate each parameter combination
            for idx, params in enumerate(param_combinations, 1):
                print(f"\n[{idx}/{total_combinations}] Evaluating: {params}")
                
                # Validate parameters
                validated_params = self._validate_params(params)
                
                try:
                    # Train model
                    val_acc, run_id = self._train_single_model(
                        validated_params,
                        data_module,
                        epochs
                    )
                    
                    print(f"  Validation Accuracy: {val_acc:.4f}")
                    
                    # Store results
                    result = {
                        'params': validated_params,
                        'val_acc': val_acc,
                        'run_id': run_id
                    }
                    self.results.append(result)
                    
                    # Update best model
                    if val_acc > self.best_score:
                        self.best_score = val_acc
                        self.best_params = validated_params
                        self.best_run_id = run_id
                        print(f"  *** New best score: {self.best_score:.4f} ***")
                        
                except Exception as e:
                    print(f"  Error training model: {str(e)}")
                    continue
            
            # Log best results to parent run
            mlflow.log_metric("best_val_acc", self.best_score)
            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
            mlflow.log_param("best_run_id", self.best_run_id)
            
            # Create results summary
            results_df = pd.DataFrame([
                {**r['params'], 'val_acc': r['val_acc'], 'run_id': r['run_id']}
                for r in self.results
            ])
            results_df = results_df.sort_values('val_acc', ascending=False)
            
            # Save results
            results_path = "gridsearch_results.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
            
            print("\n" + "=" * 80)
            print("Grid Search Completed!")
            print("=" * 80)
            print(f"Best Validation Accuracy: {self.best_score:.4f}")
            print(f"Best Parameters: {self.best_params}")
            print(f"Best Run ID: {self.best_run_id}")
            print(f"Results saved to: {results_path}")
            print("=" * 80)
        
        return self
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame([
            {**r['params'], 'val_acc': r['val_acc'], 'run_id': r['run_id']}
            for r in self.results
        ])
        return results_df.sort_values('val_acc', ascending=False)


def get_default_param_grid(model_type: str = 'simple') -> Dict[str, List[Any]]:
    """
    Get default parameter grid for grid search.
    
    Args:
        model_type: Type of model ('simple' or 'advanced')
        
    Returns:
        Dictionary of parameter names to lists of values
    """
    if model_type == 'simple':
        return {
            'hidden_size': [32, 64, 128],
            'num_layers': [2, 3, 4],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01]
        }
    elif model_type == 'advanced':
        return {
            'hidden_size': [64, 128, 256],  # Will be adjusted to be divisible by num_attention_heads
            'num_layers': [2, 3],
            'num_attention_heads': [4, 8],
            'dropout': [0.2, 0.3, 0.4],
            'layer_dropout': [0.1, 0.2],
            'learning_rate': [0.0001, 0.001],
            'weight_decay': [1e-5, 1e-4]
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_gridsearch(
    data_path: str = None,
    model_type: str = 'simple',
    param_grid: Optional[Dict[str, List[Any]]] = None,
    experiment_name: str = 'bitcoin_gridsearch',
    lookback: int = 20,
    horizon: int = 12,
    batch_size: int = 64,
    epochs: int = 100,
    num_workers: int = 4,
    seed: int = 42
) -> GridSearchCV:
    """
    Convenience function to run grid search.
    
    Args:
        data_path: Path to data file
        model_type: Type of model ('simple' or 'advanced')
        param_grid: Parameter grid (uses default if None)
        experiment_name: MLflow experiment name
        lookback: Lookback period
        horizon: Prediction horizon
        batch_size: Batch size
        epochs: Maximum epochs per model
        num_workers: Number of dataloader workers
        seed: Random seed
        
    Returns:
        Fitted GridSearchCV object
    """
    # Use default param grid if not provided
    if param_grid is None:
        param_grid = get_default_param_grid(model_type)
    
    # Set default data path
    if data_path is None:
        data_path = os.path.join(os.getenv('DATA_PATH'), 'cryptos', 'bitcoin.csv')
    
    # Create and fit grid search
    grid_search = GridSearchCV(
        model_type=model_type,
        data_path=data_path,
        experiment_name=experiment_name,
        seed=seed
    )
    
    grid_search.fit(
        param_grid=param_grid,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers
    )
    
    return grid_search


if __name__ == "__main__":
    """
    Production-ready grid search with comprehensive parameter grids.
    Run this script and let it complete (several hours recommended).
    """
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Ensure no active MLflow runs from previous executions
    if mlflow.active_run():
        print("Closing previous MLflow run...")
        mlflow.end_run()
    
    print("=" * 80)
    print("BITCOIN TIME SERIES - COMPREHENSIVE GRID SEARCH")
    print("=" * 80)
    print("This will run overnight. All results tracked in MLflow.")
    print("=" * 80 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # ========================================================================
    # RUN 1: SimpleLSTMClassifier with comprehensive grid
    # ========================================================================
    execsimple = False
    if execsimple:
        print("\n" + "=" * 80)
        print("RUN 1: SimpleLSTMClassifier - Comprehensive Grid Search")
        print("=" * 80)

        custom_grid_simple = {
            'hidden_size': [64, 128, 256, 512],
            'num_layers': [2, 3, 4],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.001, 0.01]
        }

        total_simple = (len(custom_grid_simple['hidden_size']) *
                       len(custom_grid_simple['num_layers']) *
                       len(custom_grid_simple['dropout']) *
                       len(custom_grid_simple['learning_rate']))

        print(f"Parameter Grid:")
        print(f"  hidden_size: {custom_grid_simple['hidden_size']}")
        print(f"  num_layers: {custom_grid_simple['num_layers']}")
        print(f"  dropout: {custom_grid_simple['dropout']}")
        print(f"  learning_rate: {custom_grid_simple['learning_rate']}")
        print(f"  Total combinations: {total_simple}")
        print("=" * 80 + "\n")

        grid_search_simple = run_gridsearch(
            model_type='simple',
            param_grid=custom_grid_simple,
            experiment_name='bitcoin_gridsearch_simple_comprehensive',
            epochs=150,
            batch_size=64,
            num_workers=8
        )

        print("\n" + "=" * 80)
        print("SimpleLSTMClassifier Grid Search Complete!")
        print("=" * 80)
        print(f"Best Validation Accuracy: {grid_search_simple.best_score:.4f}")
        print(f"Best Parameters: {grid_search_simple.best_params}")
        print(f"Best Run ID: {grid_search_simple.best_run_id}")
        print("\nTop 10 Results:")
        print(grid_search_simple.get_results_dataframe().head(10).to_string(index=False))
    
    # ========================================================================
    # RUN 2: AdvancedLSTMWithAttention with comprehensive grid
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("RUN 2: AdvancedLSTMWithAttention - Comprehensive Grid Search")
    print("=" * 80)

    custom_grid_advanced = {
        'hidden_size': [128, 256, 512],
        'num_layers': [2, 3, 4],
        'num_attention_heads': [4, 8, 16],
        'dropout': [0],
        'layer_dropout': [0],
        'learning_rate': [0.001],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    }
    
    total_advanced = (len(custom_grid_advanced['hidden_size']) * 
                     len(custom_grid_advanced['num_layers']) * 
                     len(custom_grid_advanced['num_attention_heads']) * 
                     len(custom_grid_advanced['dropout']) * 
                     len(custom_grid_advanced['layer_dropout']) * 
                     len(custom_grid_advanced['learning_rate']) * 
                     len(custom_grid_advanced['weight_decay']))
    
    print(f"Parameter Grid:")
    print(f"  hidden_size: {custom_grid_advanced['hidden_size']}")
    print(f"  num_layers: {custom_grid_advanced['num_layers']}")
    print(f"  num_attention_heads: {custom_grid_advanced['num_attention_heads']}")
    print(f"  dropout: {custom_grid_advanced['dropout']}")
    print(f"  layer_dropout: {custom_grid_advanced['layer_dropout']}")
    print(f"  learning_rate: {custom_grid_advanced['learning_rate']}")
    print(f"  weight_decay: {custom_grid_advanced['weight_decay']}")
    print(f"  Total combinations: {total_advanced}")
    print("=" * 80 + "\n")
    
    grid_search_advanced = run_gridsearch(
        model_type='advanced',
        param_grid=custom_grid_advanced,
        experiment_name='bitcoin_gridsearch_advanced_comprehensive',
        epochs=150,
        batch_size=64,
        num_workers=8
    )
    
    print("\n" + "=" * 80)
    print("AdvancedLSTMWithAttention Grid Search Complete!")
    print("=" * 80)
    print(f"Best Validation Accuracy: {grid_search_advanced.best_score:.4f}")
    print(f"Best Parameters: {grid_search_advanced.best_params}")
    print(f"Best Run ID: {grid_search_advanced.best_run_id}")
    print("\nTop 10 Results:")
    print(grid_search_advanced.get_results_dataframe().head(10).to_string(index=False))
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    
    print("\n\n" + "=" * 80)
    print("ALL GRID SEARCHES COMPLETED!")
    print("=" * 80)
    print(f"Total Duration: {hours}h {minutes}m")
    print(f"Total Combinations Evaluated: {total_simple + total_advanced}")
    print("\nSimpleLSTMClassifier:")
    print(f"  Best Val Acc: {grid_search_simple.best_score:.4f}")
    print(f"  Best Params: {grid_search_simple.best_params}")
    print("\nAdvancedLSTMWithAttention:")
    print(f"  Best Val Acc: {grid_search_advanced.best_score:.4f}")
    print(f"  Best Params: {grid_search_advanced.best_params}")
    print("\nOverall Winner:")
    if grid_search_simple.best_score > grid_search_advanced.best_score:
        print(f"  SimpleLSTMClassifier with {grid_search_simple.best_score:.4f} accuracy")
        print(f"  Run ID: {grid_search_simple.best_run_id}")
    else:
        print(f"  AdvancedLSTMWithAttention with {grid_search_advanced.best_score:.4f} accuracy")
        print(f"  Run ID: {grid_search_advanced.best_run_id}")
    print("\nResults saved to:")
    print("  - gridsearch_results.csv (both runs)")
    print("  - MLflow tracking server")
    print("=" * 80)
