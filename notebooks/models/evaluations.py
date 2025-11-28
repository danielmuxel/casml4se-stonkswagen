
import os
from dotenv import load_dotenv
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae, smape
from darts.models import (
    TBATS, Theta, KalmanForecaster,AutoARIMA, ARIMA, Prophet, NaiveSeasonal, XGBModel,
    LightGBMModel, CatBoostModel, RandomForest,
    LinearRegressionModel, RNNModel,
    NBEATSModel, NHiTSModel, TFTModel, TCNModel, TransformerModel
)
from typing import List, Dict
import numpy as np
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
import torch


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("ROCm version:", torch.version.hip if hasattr(torch.version, 'hip') else "N/A")
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Capability:", torch.cuda.get_device_capability(0))

    # Test basic GPU operation
    try:
        x = torch.rand(5, 3).cuda()
        print("Basic GPU operation: SUCCESS")
        print(x)
    except Exception as e:
        print("Basic GPU operation: FAILED")
        print(f"Error: {e}")


datapath = "data/evaluations/"

# Download source data from s3
from gw2ml.data.s3_sync import download_folder_from_s3

# Load environment variables
load_dotenv()


#download_folder_from_s3(s3_folder_prefix='datasources/gw2/raw/1763495310', local_folder=f"{datapath}/gw2/")
#download_folder_from_s3(s3_folder_prefix='datasources/crypto/raw/1763488125', local_folder=f"{datapath}/crypto/")


def prepare_data_gw2_univariate(input, scaling: bool = True, split_percentage=0.8):
    df = pd.read_csv(input, delimiter=";")

    value_cols = ["buy_unit_price"]
    df.head()
    tsdf = df[value_cols + ['fetched_at']].copy()

    # Convert to datetime and set as index
    tsdf['fetched_at'] = pd.to_datetime(tsdf['fetched_at'])
    # Localize timezone to UTC if it has timezone info, or remove it
    if tsdf['fetched_at'].dt.tz is not None:
        tsdf['fetched_at'] = tsdf['fetched_at'].dt.tz_localize(None)
    tsdf = tsdf.set_index('fetched_at')

    # Resample to exact 5-minute intervals, forward-filling missing values
    tsdf_resampled = tsdf[value_cols].resample('5min').mean().interpolate(method='linear')
    series = TimeSeries.from_dataframe(tsdf_resampled, value_cols=value_cols)
    train, test = series.split_after(split_percentage)

    if scaling:
        scaler = Scaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

    return train, test


def prepare_data_crypto_univariate(input, scaling: bool = True, split_percentage=0.8):
    df = pd.read_csv(input, delimiter=";")

    value_cols = ["close"]
    df.head()
    tsdf = df[value_cols + ['timestamp']].copy()

    # Convert to datetime and set as index
    tsdf['timestamp'] = pd.to_datetime(tsdf['timestamp'])
    # # Localize timezone to UTC if it has timezone info, or remove it
    # if tsdf['fetched_at'].dt.tz is not None:
    #     tsdf['fetched_at'] = tsdf['fetched_at'].dt.tz_localize(None)
    tsdf = tsdf.set_index('timestamp')

    # Resample to exact 5-minute intervals, forward-filling missing values
    tsdf_resampled = tsdf[value_cols].resample('5min').mean().interpolate(method='linear')
    series = TimeSeries.from_dataframe(tsdf_resampled, value_cols=value_cols)
    train, test = series.split_after(split_percentage)

    if scaling:
        scaler = Scaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

    return train, test


def evaluate_model(model, train_series, test_series, forecast_horizon: int = 24):
    """
    Evaluate a single model on a train/test split

    Args:
        model: Darts forecasting model
        train_series: Training TimeSeries
        test_series: Test TimeSeries
        forecast_horizon: Number of steps to forecast

    Returns:
        Dictionary with evaluation metrics and prediction (or None if error)
    """
    try:
        # Fit the model
        model.fit(train_series)

        # Make prediction
        prediction = model.predict(n=min(forecast_horizon, len(test_series)))

        # Calculate metrics
        metrics = {
            'rmse': rmse(test_series, prediction),
            'mae': mae(test_series, prediction),
            'smape': smape(test_series, prediction)
        }
        # Only compute MAPE if all values are strictly positive
        if (test_series.values() > 0).all():
            metrics['mape'] = mape(test_series, prediction)
        else:
            metrics['mape'] = np.nan

        return metrics, prediction
    except Exception as e:
        print(f"    Error in model evaluation: {e}")
        return None, None


def create_forecast_comparison_plot(train_series, test_series, prediction, model_name, item_name):
    """
    Create a comparison plot of actual vs predicted values

    Args:
        train_series: Training TimeSeries
        test_series: Test TimeSeries
        prediction: Predicted TimeSeries
        model_name: Name of the model
        item_name: Name of the dataset/item

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the prediction time range
    pred_start = prediction.start_time()
    pred_end = prediction.end_time()

    # Calculate one day before prediction start (288 * 5min intervals = 24 hours)
    train_start = pred_start - pd.Timedelta(days=1)

    # Filter training data to show last day
    train_series_filtered = train_series.slice(train_start, train_series.end_time())

    # Filter test series to prediction time range
    test_series_filtered = test_series.slice(pred_start, pred_end)

    # Plot last day of training data
    train_series_filtered.plot(ax=ax, label='Training Data (Last Day)', linewidth=1.5)

    # Plot actual test data (only prediction period)
    test_series_filtered.plot(ax=ax, label='Actual Test Data', linewidth=2, color='green')

    # Plot predictions
    prediction.plot(ax=ax, label='Forecast', linewidth=2, color='red', linestyle='--')

    ax.set_title(f'{model_name} - {item_name}\nForecast vs Actual', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_residual_plot(test_series, prediction, model_name, item_name):
    """
    Create a residual plot showing prediction errors

    Args:
        test_series: Test TimeSeries
        prediction: Predicted TimeSeries
        model_name: Name of the model
        item_name: Name of the dataset/item

    Returns:
        matplotlib figure
    """
    # Calculate residuals
    residuals = test_series.values()[:len(prediction)] - prediction.values()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Residuals over time
    ax1.plot(prediction.time_index, residuals, marker='o', linestyle='-', linewidth=1, markersize=3)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_title(f'{model_name} - {item_name}\nResiduals Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Residuals histogram
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual Value', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def evaluate_models_on_datasets(models: Dict, datasets: List[Dict], forecast_horizon: int = 24):
    """
    Evaluate multiple models on multiple datasets

    Args:
        models: Dictionary of model_name: model_instance
        datasets: List of dictionaries with 'train', 'test', and 'item' keys
        forecast_horizon: Number of steps to forecast

    Returns:
        DataFrame with results
    """
    results = []

    for dataset in datasets:
        item_name = dataset['item']
        train = dataset['train']
        test = dataset['test']

        print(f"\nEvaluating on: {item_name}")

        for model_name, model in models.items():
            print(f"  Model: {model_name}")

            try:
                # Log to MLflow with nested runs
                with mlflow.start_run(run_name=f"{model_name}_{item_name}", nested=True):
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("item", item_name)
                    mlflow.log_param("forecast_horizon", forecast_horizon)

                    metrics, prediction = evaluate_model(
                        model,
                        train,
                        test,
                        forecast_horizon
                    )

                    # Check if evaluation was successful
                    if metrics is None or prediction is None:
                        error_msg = "Model evaluation failed"
                        mlflow.log_param("status", "failed")
                        mlflow.log_param("error", error_msg)
                        results.append({
                            'item': item_name,
                            'model': model_name,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'smape': np.nan,
                            'mape': np.nan,
                            'error': error_msg
                        })
                        continue

                    mlflow.log_param("status", "success")

                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log predictions as artifact
                    try:
                        prediction.to_csv(f"prediction_{item_name}.csv")
                        mlflow.log_artifact(f"prediction_{item_name}.csv")
                        os.remove(f"prediction_{item_name}.csv")  # Clean up local file
                    except Exception as e:
                        print(f"    Warning: Could not log prediction artifact: {e}")
                        mlflow.log_param("prediction_artifact_error", str(e))

                    # Optionally log test data for reference
                    try:
                        test.to_csv(f"test_{item_name}.csv")
                        mlflow.log_artifact(f"test_{item_name}.csv")
                        os.remove(f"test_{item_name}.csv")  # Clean up local file
                    except Exception as e:
                        print(f"    Warning: Could not log test artifact: {e}")
                        mlflow.log_param("test_artifact_error", str(e))

                    # Create and log forecast comparison plot
                    try:
                        forecast_fig = create_forecast_comparison_plot(
                            train, test, prediction, model_name, item_name
                        )
                        mlflow.log_figure(forecast_fig, f"forecast_comparison_{item_name}.png")
                        plt.close(forecast_fig)
                    except Exception as e:
                        print(f"    Warning: Could not create/log forecast plot: {e}")
                        mlflow.log_param("forecast_plot_error", str(e))

                    # Create and log residual plot
                    try:
                        residual_fig = create_residual_plot(
                            test, prediction, model_name, item_name
                        )
                        mlflow.log_figure(residual_fig, f"residuals_{item_name}.png")
                        plt.close(residual_fig)
                    except Exception as e:
                        print(f"    Warning: Could not create/log residual plot: {e}")
                        mlflow.log_param("residual_plot_error", str(e))

                    result = {
                        'item': item_name,
                        'model': model_name,
                        **metrics
                    }
                    results.append(result)

            except Exception as e:
                print(f"    Error: {e}")
                # Log error to MLflow
                try:
                    with mlflow.start_run(run_name=f"{model_name}_{item_name}_ERROR", nested=True):
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("item", item_name)
                        mlflow.log_param("status", "error")
                        mlflow.log_param("error_type", type(e).__name__)
                        mlflow.log_param("error_message", str(e))
                except Exception as mlflow_error:
                    print(f"    Could not log error to MLflow: {mlflow_error}")

                results.append({
                    'item': item_name,
                    'model': model_name,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'smape': np.nan,
                    'mape': np.nan,
                    'error': str(e)
                })

    return pd.DataFrame(results)


def print_comparison_summary(results_df: pd.DataFrame):
    """
    Print a summary comparison of models
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Check for errors
    if 'error' in results_df.columns:
        error_count = results_df['error'].notna().sum()
        if error_count > 0:
            print(f"\n⚠️  {error_count} model evaluation(s) failed")
            print("\nFailed evaluations:")
            failed = results_df[results_df['error'].notna()][['item', 'model', 'error']]
            print(failed.to_string())
            print("\n" + "-" * 80)

    # Filter out failed runs for metrics calculations
    successful_results = results_df[results_df['error'].isna()] if 'error' in results_df.columns else results_df

    if len(successful_results) == 0:
        print("\n❌ No successful model evaluations to summarize")
        return

    # Average performance by model
    print("\nAverage Performance by Model:")
    avg_by_model = successful_results.groupby('model')[['mape', 'rmse', 'mae', 'smape']].mean()
    print(avg_by_model.to_string())

    # Best model per metric
    print("\nBest Model per Metric:")
    for metric in ['mape', 'rmse', 'mae', 'smape']:
        if metric in successful_results.columns:
            valid_metric = successful_results[successful_results[metric].notna()]
            if len(valid_metric) > 0:
                best_idx = valid_metric[metric].idxmin()
                best_model = valid_metric.loc[best_idx, 'model']
                best_value = valid_metric.loc[best_idx, metric]
                print(f"  {metric.upper()}: {best_model} ({best_value:.4f})")

    # Performance by dataset type
    successful_results['dataset_type'] = successful_results['item'].apply(
        lambda x: 'crypto' if 'crypto' in x.lower() or any(
            crypto in x.lower() for crypto in ['btc', 'eth', 'usdt']) else 'gw2'
    )

    print("\nAverage Performance by Dataset Type:")
    avg_by_type = successful_results.groupby(['dataset_type', 'model'])[['mape', 'rmse', 'mae', 'smape']].mean()
    print(avg_by_type.to_string())


# Main execution
gw2_serieses = []
crypto_serieses = []

for file in os.listdir(f"{datapath}/gw2/"):
    print(file)
    train, test = prepare_data_gw2_univariate(f"{datapath}/gw2/{file}")

    gw2_serieses.append({"train": train,
                         "test": test,
                         "item": file})

for file in os.listdir(f"{datapath}/crypto/"):
    print(file)
    train, test = prepare_data_crypto_univariate(f"{datapath}/crypto/{file}")

    crypto_serieses.append({"train": train,
                            "test": test,
                            "item": file})

if __name__ == "__main__":
    early_stopper = EarlyStopping(
        monitor="train_loss",      # Metric to monitor
        patience=5,                # Number of epochs with no improvement after which training will be stopped
        min_delta=0.001,           # Minimum change to qualify as an improvement
        mode="min"                  # "min" for loss, "max" for accuracy-like metrics
    )
    # Define models to compare
    models = {

        # Traditional statistical
        'Theta': Theta(),
        'KalmanForecaster': KalmanForecaster(),
        # Machine Learning
        'LightGBM': LightGBMModel(lags=12),
        'CatBoost': CatBoostModel(lags=12),
        'RandomForest': RandomForest(lags=12),
        'LinearRegression': LinearRegressionModel(lags=12),

        #'ARIMA_auto': ARIMA(),  # Keep auto version for comparison
        'AutoArima': AutoARIMA(),
        'Prophet': Prophet(),
        'NaiveSeasonal': NaiveSeasonal(K=12),
        'XGBoost': XGBModel(lags=12),



        # Deep Learning (use a gpu....)
        #   then go to sleep and realize that after 2 hours something failed and you can go try again...
        'NBEATS': NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            n_epochs=20,  # Maximum epochs
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": 1,
                "callbacks": [early_stopper]
            }
        ),
        'NHiTS': NHiTSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            n_epochs=20,
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": 1,
                "callbacks": [early_stopper]
            }
        ),
        'TCN': TCNModel(
            input_chunk_length=24,
            output_chunk_length=12,
            n_epochs=20,
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": 1,
                "callbacks": [early_stopper]
            }
        ),
          }

    # Set MLflow experiment
    mlflow.set_experiment("model_comparison_evaluation")



    # Run evaluation on GW2 datasets
    print("\n" + "=" * 80)
    print("Evaluating on GW2 datasets...")
    print("=" * 80)
    with mlflow.start_run(run_name="GW2_Evaluation"):
        gw2_results = evaluate_models_on_datasets(models, gw2_serieses, forecast_horizon=24)

    # Run evaluation on Crypto datasets
    print("\n" + "=" * 80)
    print("Evaluating on Crypto datasets...")
    print("=" * 80)
    with mlflow.start_run(run_name="Crypto_Evaluation"):
        crypto_results = evaluate_models_on_datasets(models, crypto_serieses, forecast_horizon=24)

    # Combine results
    all_results = pd.concat([gw2_results, crypto_results], ignore_index=True)

    # Print summary
    print_comparison_summary(all_results)

    # Save results
    all_results.to_csv(f"{datapath}/evaluation_results.csv", index=False)
    print(f"\nResults saved to {datapath}/evaluation_results.csv")