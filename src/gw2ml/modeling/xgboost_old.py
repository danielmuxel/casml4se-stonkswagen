
import mlflow
from darts.metrics import mape
from darts.models import XGBModel

def run_xgboost_hyperparameter_search(
    train_series,
    val_series,
    lags_range,
    n_estimators_range,
    max_depth_range,
    learning_rate_range,
    subsample_range,
    colsample_bytree_range,
    experiment_name="xgboost_univariate_search"
):
    """
    Performs a hyperparameter search for an XGBoost model on a given time series.

    Logs the experiment to MLflow, including parameters for each run and the resulting
    MAPE metric. The best performing model's parameters and score are logged to the
    parent run.

    Args:
        train_series: The Darts TimeSeries to be used for training.
        val_series: The Darts TimeSeries to be used for validation.
        lags_range: A list of 'lags' values to try.
        n_estimators_range: A list of 'n_estimators' values to try.
        max_depth_range: A list of 'max_depth' values to try.
        learning_rate_range: A list of 'learning_rate' values to try.
        subsample_range: A list of 'subsample' values to try.
        colsample_bytree_range: A list of 'colsample_bytree' values to try.
        experiment_name: The name of the MLflow experiment.

    Returns:
        A tuple containing the best parameters dictionary and the best MAPE score.
    """
    mlflow.set_experiment(experiment_name)

    best_mape = float('inf')
    best_params = None

    # Calculate total combinations for progress tracking
    total_combinations = (
        len(lags_range) * len(n_estimators_range) * len(max_depth_range) *
        len(learning_rate_range) * len(subsample_range) * len(colsample_bytree_range)
    )

    # Parent run for the hyperparameter search
    with mlflow.start_run(run_name="XGBOOST_Hyperparameter_Search") as parent_run:
        # Log search configuration
        mlflow.log_param("search_lags", str(lags_range))
        mlflow.log_param("search_n_estimators", str(n_estimators_range))
        mlflow.log_param("search_max_depth", str(max_depth_range))
        mlflow.log_param("search_learning_rate", str(learning_rate_range))
        mlflow.log_param("search_subsample", str(subsample_range))
        mlflow.log_param("search_colsample_bytree", str(colsample_bytree_range))
        mlflow.log_param("total_combinations", total_combinations)

        combination_count = 0
        for lags in lags_range:
            for n_estimators in n_estimators_range:
                for max_depth in max_depth_range:
                    for learning_rate in learning_rate_range:
                        for subsample in subsample_range:
                            for colsample_bytree in colsample_bytree_range:
                                combination_count += 1
                                run_name = (
                                    f"XGB_{lags}_{n_estimators}_{max_depth}_"
                                    f"{learning_rate}_{subsample}_{colsample_bytree}"
                                )
                                print(
                                    f"Testing XGBModel {combination_count}/{total_combinations}: "
                                    f"lags={lags}, n_est={n_estimators}, depth={max_depth}, "
                                    f"lr={learning_rate}, sub={subsample}, col={colsample_bytree}"
                                )

                                # Child run for each parameter combination
                                with mlflow.start_run(run_name=run_name, nested=True):
                                    params = {
                                        "lags": lags,
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "output_chunk_length": 1,
                                        "reg_alpha": 0.0,
                                        "reg_lambda": 0.0,
                                        "model_type": "XGBModel",
                                    }
                                    mlflow.log_params(params)

                                    try:
                                        model = XGBModel(
                                            lags=lags,
                                            output_chunk_length=1,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            learning_rate=learning_rate,
                                            subsample=subsample,
                                            colsample_bytree=colsample_bytree,
                                            reg_alpha=0.0,
                                            reg_lambda=0.0,
                                        )
                                        model.fit(train_series)
                                        forecast = model.predict(len(val_series))
                                        error = mape(val_series, forecast)
                                        mlflow.log_metric("mape", error)

                                        if error < best_mape:
                                            best_mape = error
                                            best_params = params
                                            print(f"New best: {best_params} with MAPE: {error:.2f}%")
                                            
                                            # Log best model to the parent run
                                            with mlflow.start_run(run_id=parent_run.info.run_id):
                                                mlflow.log_metric("best_mape", error, step=combination_count)
                                                mlflow.log_params({f"best_{k}": v for k, v in params.items()})


                                    except Exception as e:
                                        mlflow.log_param("status", "failed")
                                        mlflow.log_param("error", str(e))
                                        print(f"Failed XGBModel combination: {run_name} with error: {e}")

        # Log final results in parent run
        with mlflow.start_run(run_id=parent_run.info.run_id):
            mlflow.log_metric("final_best_mape", best_mape)
            if best_params:
                mlflow.log_params({f"final_best_{k}": v for k,v in best_params.items()})

    print(f"\nBest model parameters: {best_params} with MAPE: {best_mape:.2f}%")
    return best_params, best_mape
