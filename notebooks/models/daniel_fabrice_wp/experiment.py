"""
Bitcoin Time Series Data Preparation and Training Pipeline
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

Usage:
    python experiment.py --data_path btc_5min.csv --epochs 100
"""
import time
import os
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description='aBitcoin Time Series ML Pipeline with MLflow logging',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getenv('DATA_PATH'), 'cryptos', 'bitcoin-tiny.csv'),
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
    # START MLFLOW RUN
    # ========================================================================
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

    # Setup data (this runs all preprocessing)
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
