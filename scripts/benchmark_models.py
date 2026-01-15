#!/usr/bin/env python3
"""
Benchmark-Script für Forecasting-Modelle.

Vergleicht alle Modelle (ARIMA, ExponentialSmoothing, XGBoost, Chronos2)
auf dem aktuellen System und generiert einen Performance-Report.

Verwendung:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --item-id 19976 --days 30
    python scripts/benchmark_models.py --output results/benchmark.json
"""

from __future__ import annotations

# Suppress verbose logging BEFORE any imports
import logging
import os
import warnings

# Suppress PyTorch Lightning verbosity
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set logging levels
for logger_name in ["pytorch_lightning", "lightning.pytorch", "lightning", "transformers"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Falling back to prediction.*")

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class SystemInfo:
    """System-Informationen."""
    platform: str
    python_version: str
    cpu: str
    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    gpu_backend: str  # cuda, rocm, mps, cpu
    torch_version: str
    timestamp: str


@dataclass
class BenchmarkResult:
    """Ergebnis eines Model-Benchmarks."""
    model_name: str
    fit_time_sec: float
    predict_time_sec: float
    backtest_time_sec: float
    total_time_sec: float
    device: str
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Vollständiger Benchmark-Report."""
    system: SystemInfo
    config: Dict[str, Any]
    results: List[BenchmarkResult]
    fastest_model: str
    relative_speeds: Dict[str, float]


def get_system_info() -> SystemInfo:
    """Sammle System-Informationen."""
    import torch
    import psutil

    # GPU detection
    gpu_name = None
    gpu_memory_gb = None
    gpu_backend = "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Check for ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            gpu_backend = "rocm"
        else:
            gpu_backend = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_backend = "mps"
        gpu_name = "Apple Silicon (MPS)"

    # CPU info
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
    except ImportError:
        cpu = platform.processor() or "Unknown"

    return SystemInfo(
        platform=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu=cpu,
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count(),
        ram_gb=psutil.virtual_memory().total / 1024**3,
        gpu_name=gpu_name,
        gpu_memory_gb=round(gpu_memory_gb, 1) if gpu_memory_gb else None,
        gpu_backend=gpu_backend,
        torch_version=torch.__version__,
        timestamp=datetime.now().isoformat(),
    )


def create_test_data(n_points: int = 5000) -> "TimeSeries":
    """Erstelle synthetische Testdaten."""
    from darts import TimeSeries

    # Simuliere realistische Preisdaten mit Trend und Noise
    np.random.seed(42)
    t = np.arange(n_points)
    trend = 1000 + 0.1 * t
    seasonal = 50 * np.sin(2 * np.pi * t / 288)  # Tägliche Saisonalität
    noise = np.random.randn(n_points) * 20
    data = (trend + seasonal + noise).astype(np.float32)

    return TimeSeries.from_values(data)


def load_real_data(item_id: int, days_back: int) -> Optional["TimeSeries"]:
    """Lade echte Daten aus der Datenbank."""
    try:
        from gw2ml.data.loaders import load_gw2_series
        data = load_gw2_series(item_id, days_back=days_back)
        return data.series
    except Exception as e:
        print(f"  Warnung: Konnte Daten nicht laden ({e}), verwende synthetische Daten")
        return None


def benchmark_model(
    model_class,
    model_params: Dict[str, Any],
    series: "TimeSeries",
    forecast_horizon: int = 12,
) -> BenchmarkResult:
    """Benchmarke ein einzelnes Modell."""
    import torch
    import warnings

    model_name = model_params.get("_name", model_class.__name__)
    device = "cpu"
    is_local_model = model_params.get("_local", False)

    # Suppress verbose warnings during benchmark
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        # Erstelle Modell
        params = {k: v for k, v in model_params.items() if not k.startswith("_")}
        model = model_class(**params)

        # Device ermitteln
        if hasattr(model, '_device'):
            device = model._device

        # 1. Fit benchmark
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start = time.perf_counter()
        model.fit(series)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fit_time = time.perf_counter() - start

        # 2. Predict benchmark
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model.predict(n=forecast_horizon)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predict_time = time.perf_counter() - start

        # 3. Historical forecasts benchmark (backtest)
        # LocalForecastingModels (ARIMA, ExpSmoothing) require retrain=True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=forecast_horizon,
            stride=forecast_horizon,  # Nicht jeder Schritt, sonst zu langsam
            retrain=is_local_model,  # Local models need retrain=True
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backtest_time = time.perf_counter() - start

        # Cleanup
        if hasattr(model, 'cleanup'):
            model.cleanup()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return BenchmarkResult(
            model_name=model_name,
            fit_time_sec=round(fit_time, 3),
            predict_time_sec=round(predict_time, 3),
            backtest_time_sec=round(backtest_time, 3),
            total_time_sec=round(fit_time + predict_time + backtest_time, 3),
            device=device,
        )

    except Exception as e:
        return BenchmarkResult(
            model_name=model_name,
            fit_time_sec=0,
            predict_time_sec=0,
            backtest_time_sec=0,
            total_time_sec=0,
            device=device,
            error=str(e),
        )


def run_benchmark(
    item_id: Optional[int] = None,
    days_back: int = 30,
    n_points: int = 5000,
    forecast_horizon: int = 12,
) -> BenchmarkReport:
    """Führe kompletten Benchmark durch."""
    import logging
    import warnings

    # Suppress verbose logging from PyTorch Lightning and other libraries
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("darts").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    from gw2ml.modeling.arima import ARIMAModel
    from gw2ml.modeling.exponential_smoothing import ExponentialSmoothingModel
    from gw2ml.modeling.xgboost import XGBoostModel
    from gw2ml.modeling.chronos import Chronos2

    print("=" * 60)
    print("FORECASTING MODEL BENCHMARK")
    print("=" * 60)

    # System Info
    print("\n[1/4] Sammle System-Informationen...")
    system_info = get_system_info()
    print(f"  Platform: {system_info.platform}")
    print(f"  CPU: {system_info.cpu} ({system_info.cpu_cores} Kerne)")
    print(f"  RAM: {system_info.ram_gb:.1f} GB")
    print(f"  GPU: {system_info.gpu_name or 'Keine'}")
    if system_info.gpu_memory_gb:
        print(f"  GPU Memory: {system_info.gpu_memory_gb} GB")
    print(f"  Backend: {system_info.gpu_backend}")

    # Daten laden
    print("\n[2/4] Lade Daten...")
    series = None
    data_source = "synthetic"

    if item_id:
        series = load_real_data(item_id, days_back)
        if series:
            data_source = f"item_{item_id}"
            n_points = len(series)

    if series is None:
        series = create_test_data(n_points)
        data_source = "synthetic"

    print(f"  Quelle: {data_source}")
    print(f"  Datenpunkte: {len(series)}")
    print(f"  Forecast Horizon: {forecast_horizon}")

    # Modell-Konfigurationen
    # _local=True für LocalForecastingModels (benötigen retrain=True im Backtest)
    models = [
        (ARIMAModel, {"_name": "ARIMA", "_local": True, "p": 1, "d": 1, "q": 1}),
        (ExponentialSmoothingModel, {"_name": "ExpSmoothing", "_local": True, "trend": "add"}),
        (XGBoostModel, {"_name": "XGBoost", "lags": 24, "n_estimators": 200}),
        (Chronos2, {"_name": "Chronos2", "model_hub_name": "autogluon/chronos-2-small"}),
    ]

    # Benchmarks ausführen
    print("\n[3/4] Führe Benchmarks durch...")
    results = []

    for model_class, params in models:
        name = params["_name"]
        print(f"\n  Benchmarking {name}...", end=" ", flush=True)
        result = benchmark_model(model_class, params, series, forecast_horizon)

        if result.error:
            print(f"FEHLER: {result.error}")
        else:
            print(f"OK ({result.total_time_sec:.2f}s, {result.device})")

        results.append(result)

    # Ergebnisse berechnen
    print("\n[4/4] Berechne Ergebnisse...")

    # Finde schnellstes Modell (ohne Fehler)
    valid_results = [r for r in results if not r.error]
    if not valid_results:
        print("  FEHLER: Alle Modelle sind fehlgeschlagen!")
        fastest = results[0].model_name
        relative = {r.model_name: float('inf') for r in results}
    else:
        fastest = min(valid_results, key=lambda r: r.total_time_sec)
        baseline = fastest.total_time_sec

        relative = {}
        for r in results:
            if r.error:
                relative[r.model_name] = float('inf')
            else:
                relative[r.model_name] = round(r.total_time_sec / baseline, 2)

    config = {
        "data_source": data_source,
        "n_points": len(series),
        "forecast_horizon": forecast_horizon,
        "item_id": item_id,
        "days_back": days_back,
    }

    return BenchmarkReport(
        system=system_info,
        config=config,
        results=results,
        fastest_model=fastest.model_name,
        relative_speeds=relative,
    )


def print_report(report: BenchmarkReport) -> None:
    """Drucke formatierten Report."""
    print("\n" + "=" * 60)
    print("BENCHMARK ERGEBNISSE")
    print("=" * 60)

    # Config
    print(f"\nKonfiguration:")
    print(f"  Datenquelle: {report.config['data_source']}")
    print(f"  Datenpunkte: {report.config['n_points']}")
    print(f"  Forecast Horizon: {report.config['forecast_horizon']}")

    # Tabelle Header
    print("\n" + "-" * 80)
    print(f"{'Modell':<15} {'Device':<8} {'Fit':>8} {'Predict':>8} {'Backtest':>8} {'Total':>8} {'Relativ':>8}")
    print("-" * 80)

    # Sortiere nach Total Zeit
    sorted_results = sorted(report.results, key=lambda r: r.total_time_sec if not r.error else float('inf'))

    for r in sorted_results:
        rel = report.relative_speeds.get(r.model_name, float('inf'))

        if r.error:
            print(f"{r.model_name:<15} {'N/A':<8} {'ERROR':>8} {'':<8} {'':<8} {'':<8} {'N/A':>8}")
        else:
            rel_str = f"{rel:.1f}x" if rel != float('inf') else "N/A"
            print(f"{r.model_name:<15} {r.device:<8} {r.fit_time_sec:>7.2f}s {r.predict_time_sec:>7.3f}s {r.backtest_time_sec:>7.2f}s {r.total_time_sec:>7.2f}s {rel_str:>8}")

    print("-" * 80)

    # Fazit
    print(f"\nSchnellstes Modell: {report.fastest_model} (1.0x)")

    # System Info
    print(f"\nSystem: {report.system.gpu_name or 'CPU only'} ({report.system.gpu_backend})")
    print(f"Timestamp: {report.system.timestamp}")


def save_report(report: BenchmarkReport, output_path: Path) -> None:
    """Speichere Report als JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = {
        "system": asdict(report.system),
        "config": report.config,
        "results": [asdict(r) for r in report.results],
        "fastest_model": report.fastest_model,
        "relative_speeds": report.relative_speeds,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nReport gespeichert: {output_path}")


def main():
    # Default output path
    script_dir = Path(__file__).parent.parent
    default_output = script_dir / "models" / "benchmark.json"

    parser = argparse.ArgumentParser(description="Benchmark Forecasting-Modelle")
    parser.add_argument("--item-id", type=int, help="GW2 Item-ID für echte Daten")
    parser.add_argument("--days", type=int, default=30, help="Tage zurück (default: 30)")
    parser.add_argument("--points", type=int, default=5000, help="Synthetische Datenpunkte (default: 5000)")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast Horizon (default: 12)")
    parser.add_argument("--output", type=str, default=str(default_output), help=f"JSON Output-Pfad (default: {default_output})")
    parser.add_argument("--no-save", action="store_true", help="Ergebnisse nicht speichern")

    args = parser.parse_args()

    report = run_benchmark(
        item_id=args.item_id,
        days_back=args.days,
        n_points=args.points,
        forecast_horizon=args.horizon,
    )

    print_report(report)

    if not args.no_save:
        save_report(report, Path(args.output))


if __name__ == "__main__":
    main()
