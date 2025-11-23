
"""
Bitcoin Time Series Data Preparation and Training Pipeline
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)
"""
import time
import os
import warnings
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import classification_report, confusion_matrix

import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

from dataset import TimeSeriesDataModule
from model import SimpleLSTMClassifier

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()


def train(
    data_path: str = None,
    experiment_name: str = 'bitcoin_timeseries2',
    run_name: str = None,
    lookback: int = 20,
    horizon: int = 12,
    hidden_size: int = 64,
    dropout: float = 0.2,
    batch_size: int = 64,
    epochs: int = 250,
    learning_rate: float = 0.001,
    num_workers: int = 40,
    seed: int = 42
):
    """
    Train a Bitcoin time series prediction model.
    
    Args:
        data_path: Path to CSV file with columns: timestamp,open,high,low,close,volume
        experiment_name: MLflow experiment name
        run_name: MLflow run name (optional)
        lookback: Lookback period (number of bars per slice)
        horizon: Prediction horizon (bars ahead to predict)
        hidden_size: LSTM hidden size
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        num_workers: Number of dataloader workers
        seed: Random seed
    """
    # Set default data path if not provided
    if data_path is None:
        data_path = os.path.join(os.getenv('DATA_PATH'), 'cryptos', 'bitcoin-tiny.csv')
    
    # Set random seeds
    pl.seed_everything(seed)

    print("=" * 80)
    print("Bitcoin Time Series ML Pipeline with MLflow Tracking")
    print("Based on: 'Financial Time Series Data Processing for Machine Learning'")
    print("=" * 80)
    print(f"\nMLflow Configuration:")
    print(f"  Tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
    print(f"  Experiment Name: {experiment_name}")
    print("=" * 80)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # ========================================================================
    # START MLFLOW RUN
    # ========================================================================
    if mlflow.active_run():
        print("\nWarning: Found active MLflow run. Ending it before starting new run...")
        mlflow.end_run()
    mlflow.start_run(run_name=run_name)

    start_time = time.time()
    mlflow.log_metric("start_timestamp", start_time)

    mlflow.pytorch.autolog()
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    print(f"\nMLflow Run ID: {mlflow_logger.run_id}")
    print(f"MLflow Run Name: {run_name}")

    print("\n" + "=" * 80)
    print("Starting data preprocessing...")
    print("=" * 80)

    # Initialize data module
    data_module = TimeSeriesDataModule(
        data_path=data_path,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Setup data (this runs all preprocessing)
    data_module.setup()

    print("\n" + "=" * 80)
    print("Preprocessing completed!")
    print("=" * 80)

    # Log all hyperparameters
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("lookback", lookback)
    mlflow.log_param("horizon", horizon)
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("max_epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_workers", num_workers)

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
        hidden_size=hidden_size,
        num_classes=3,  # QClass: Up, Neutral, Down
        learning_rate=learning_rate,
        dropout=dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:")
    print(f"  Input features: {data_module.n_features}")
    print(f"  LSTM hidden size: {hidden_size}")
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
        max_epochs=epochs,
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


if __name__ == "__main__":
    train()
