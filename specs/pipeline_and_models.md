# Spezifikation: Pipeline & Model Framework

> **Version:** 1.0  
> **Datum:** 2024-12-26  
> **Status:** Draft

---

## 1. Übersicht

Dieses Dokument spezifiziert ein erweiterbares Framework für Time Series Forecasting von Guild Wars 2 Preisdaten. Das Framework besteht aus:

1. **BaseModel** – Abstrakte Basisklasse für alle Forecasting-Modelle
2. **BasePipeline** – Abstrakte Basisklasse für Training, Backtesting und Evaluation
3. **Konkrete Modelle** – ARIMA, Exponential Smoothing (erweiterbar)
4. **StandardPipeline** – Konkrete Pipeline mit MLflow-Integration
5. **Model Registry** – Zentrale Registrierung und Factory für Modelle

---

## 2. Anforderungen

### 2.1 Funktionale Anforderungen

| ID | Anforderung | Priorität |
|----|-------------|-----------|
| F1 | Neue Modelle sollen einfach durch Ableitung von `BaseModel` erstellt werden können | Hoch |
| F2 | Pipeline muss Backtesting (Walk-Forward Validation) unterstützen | Hoch |
| F3 | Alle Experimente werden in MLflow geloggt (Parameter, Metriken, Plots) | Hoch |
| F4 | Daten werden direkt aus der Datenbank geladen (keine Interpolation) | Hoch |
| F5 | Standard-Metriken: MAPE, RMSE, MAE, SMAPE | Hoch |
| F6 | Forecast-Visualisierung als Artifact | Mittel |
| F7 | Factory-Funktion für einfache Pipeline-Erstellung | Mittel |

### 2.2 Nicht-funktionale Anforderungen

| ID | Anforderung |
|----|-------------|
| NF1 | Verwendung von Darts als Forecasting-Library |
| NF2 | Python 3.10+ kompatibel |
| NF3 | Type Hints für alle öffentlichen Methoden |
| NF4 | Bestehende `gw2ml.data` Module werden wiederverwendet |

---

## 3. Architektur

### 3.1 Komponentendiagramm

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Registry                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   ARIMA     │  │  ExpSmooth  │  │  Custom Models (später) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     StandardPipeline                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │  Load    │→ │  Train   │→ │ Backtest │→ │    Evaluate     │  │
│  │  Data    │  │  Model   │  │          │  │  + MLflow Log   │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      gw2ml.data                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │ database_client  │  │           retriever                  │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Verzeichnisstruktur

```
src/gw2ml/
├── modeling/
│   ├── __init__.py                 # Exports: BaseModel, ARIMAModel, etc.
│   ├── base.py                     # BaseModel ABC
│   ├── arima.py                    # ARIMAModel
│   ├── exponential_smoothing.py    # ExponentialSmoothingModel
│   └── registry.py                 # ModelRegistry + @register_model + create_pipeline()
│
├── pipeline/
│   ├── __init__.py                 # Exports: BasePipeline, StandardPipeline, etc.
│   ├── base.py                     # BasePipeline ABC
│   ├── standard.py                 # StandardPipeline
│   └── config.py                   # BacktestConfig, PipelineConfig
│
├── data/                           # (existiert bereits)
│   ├── database_client.py
│   ├── retriever.py
│   └── ...
│
└── evaluation/                     # (existiert bereits)
    └── plotting.py
```

---

## 4. Klassen-Spezifikation

### 4.1 BaseModel (Abstract)

**Datei:** `src/gw2ml/modeling/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from darts import TimeSeries


class BaseModel(ABC):
    """
    Abstrakte Basisklasse für alle Forecasting-Modelle.
    
    Jedes konkrete Modell muss diese Klasse erweitern und die
    abstrakten Methoden implementieren.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Name des Modells (z.B. 'ARIMA', 'ExponentialSmoothing')"""
        pass
    
    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Standard-Hyperparameter für das Modell"""
        pass
    
    @property
    def supports_covariates(self) -> bool:
        """Ob das Modell externe Kovariaten unterstützt. Default: False"""
        return False
    
    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """
        Erstellt und konfiguriert das interne Darts-Modell.
        
        Args:
            **kwargs: Modell-spezifische Parameter
            
        Returns:
            Konfiguriertes Darts-Modell
        """
        pass
    
    @abstractmethod
    def fit(self, series: TimeSeries, **kwargs) -> "BaseModel":
        """
        Trainiert das Modell auf der gegebenen TimeSeries.
        
        Args:
            series: Trainings-TimeSeries
            **kwargs: Zusätzliche Trainingsparameter
            
        Returns:
            self (für Method Chaining)
        """
        pass
    
    @abstractmethod
    def predict(self, n: int, **kwargs) -> TimeSeries:
        """
        Erstellt einen Forecast für n Zeitschritte.
        
        Args:
            n: Anzahl der Schritte für den Forecast
            **kwargs: Zusätzliche Predict-Parameter
            
        Returns:
            TimeSeries mit den Vorhersagen
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Gibt die aktuellen Modell-Parameter zurück"""
        pass
    
    def historical_forecasts(
        self,
        series: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        **kwargs
    ) -> TimeSeries:
        """
        Führt Walk-Forward Backtesting durch.
        
        Nutzt intern Darts' historical_forecasts() Methode.
        
        Args:
            series: Vollständige TimeSeries
            start: Startpunkt als Anteil (0.0-1.0) oder absoluter Index
            forecast_horizon: Anzahl Schritte pro Forecast
            stride: Schritte zwischen Forecasts
            retrain: Ob das Modell bei jedem Step neu trainiert wird
            
        Returns:
            TimeSeries mit allen Backtest-Forecasts
        """
        pass
```

### 4.2 ARIMAModel

**Datei:** `src/gw2ml/modeling/arima.py`

```python
from darts.models import ARIMA
from .base import BaseModel
from .registry import register_model


@register_model
class ARIMAModel(BaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) Modell.
    
    Geeignet für univariate Time Series ohne Saisonalität.
    Für saisonale Daten sollte SARIMA verwendet werden.
    """
    
    name = "ARIMA"
    
    default_params = {
        "p": 1,      # Autoregressive Ordnung
        "d": 1,      # Differenzierungs-Ordnung
        "q": 1,      # Moving Average Ordnung
    }
    
    def __init__(
        self,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs
    ):
        """
        Args:
            p: AR-Ordnung (Autoregressive)
            d: Differenzierung (Integrated)
            q: MA-Ordnung (Moving Average)
            seasonal_order: Optional (P, D, Q, m) für SARIMA
            **kwargs: Weitere Darts ARIMA Parameter
        """
        self.params = {
            "p": p,
            "d": d,
            "q": q,
            "seasonal_order": seasonal_order,
            **kwargs
        }
        self._model: Optional[ARIMA] = None
```

### 4.3 ExponentialSmoothingModel

**Datei:** `src/gw2ml/modeling/exponential_smoothing.py`

```python
from darts.models import ExponentialSmoothing
from .base import BaseModel
from .registry import register_model


@register_model
class ExponentialSmoothingModel(BaseModel):
    """
    Exponential Smoothing (Holt-Winters) Modell.
    
    Unterstützt verschiedene Trend- und Saisonalitäts-Komponenten.
    """
    
    name = "ExponentialSmoothing"
    
    default_params = {
        "trend": "add",           # "add", "mul", oder None
        "seasonal": None,         # "add", "mul", oder None
        "seasonal_periods": None, # z.B. 288 für Tagessaisonalität bei 5-Min-Daten
        "damped": False,          # Gedämpfter Trend
    }
    
    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped: bool = False,
        **kwargs
    ):
        """
        Args:
            trend: Trend-Komponente ("add", "mul", None)
            seasonal: Saisonale Komponente ("add", "mul", None)
            seasonal_periods: Länge einer Saison (z.B. 288 = 1 Tag bei 5-Min)
            damped: Ob der Trend gedämpft werden soll
            **kwargs: Weitere Darts ExponentialSmoothing Parameter
        """
        self.params = {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "damped": damped,
            **kwargs
        }
        self._model: Optional[ExponentialSmoothing] = None
```

### 4.4 Model Registry

**Datei:** `src/gw2ml/modeling/registry.py`

```python
from typing import Dict, Type, Optional, Any


class ModelRegistry:
    """
    Zentrale Registry für alle verfügbaren Modelle.
    
    Ermöglicht das dynamische Registrieren und Abrufen von Modell-Klassen.
    """
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_class: Type[BaseModel]) -> Type[BaseModel]:
        """Registriert eine Modell-Klasse"""
        cls._models[model_class.name] = model_class
        return model_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Gibt die Modell-Klasse für den gegebenen Namen zurück"""
        if name not in cls._models:
            raise ValueError(
                f"Modell '{name}' nicht gefunden. "
                f"Verfügbar: {list(cls._models.keys())}"
            )
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list[str]:
        """Gibt alle registrierten Modell-Namen zurück"""
        return list(cls._models.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Erstellt eine Modell-Instanz"""
        model_class = cls.get(name)
        return model_class(**kwargs)


def register_model(cls: Type[BaseModel]) -> Type[BaseModel]:
    """Decorator zum Registrieren von Modellen"""
    return ModelRegistry.register(cls)


def create_pipeline(
    model_name: str,
    experiment_name: str,
    model_params: Optional[Dict[str, Any]] = None,
    pipeline_config: Optional["PipelineConfig"] = None
) -> "StandardPipeline":
    """
    Factory-Funktion zum Erstellen einer Pipeline mit Modell.
    
    Args:
        model_name: Name des Modells (z.B. "ARIMA")
        experiment_name: Name des MLflow Experiments
        model_params: Optionale Modell-Parameter
        pipeline_config: Optionale Pipeline-Konfiguration
        
    Returns:
        Konfigurierte StandardPipeline
        
    Example:
        >>> pipeline = create_pipeline("ARIMA", "my_experiment", {"p": 2, "d": 1})
        >>> results = pipeline.run(item_id=19976)
    """
    from ..pipeline import StandardPipeline, PipelineConfig
    
    model_params = model_params or {}
    model = ModelRegistry.create(model_name, **model_params)
    config = pipeline_config or PipelineConfig()
    
    return StandardPipeline(model, experiment_name, config)
```

### 4.5 BasePipeline (Abstract)

**Datei:** `src/gw2ml/pipeline/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from darts import TimeSeries
import pandas as pd

from ..modeling.base import BaseModel


class BasePipeline(ABC):
    """
    Abstrakte Basisklasse für ML-Pipelines.
    
    Definiert den Standard-Workflow: Load → Train → Backtest → Evaluate
    """
    
    def __init__(self, model: BaseModel, experiment_name: str):
        """
        Args:
            model: Das zu verwendende Forecasting-Modell
            experiment_name: Name des MLflow Experiments
        """
        self.model = model
        self.experiment_name = experiment_name
    
    @abstractmethod
    def load_data(
        self,
        item_id: int,
        days_back: int = 30,
        value_column: str = "buy_unit_price"
    ) -> TimeSeries:
        """
        Lädt Daten aus der Datenbank und konvertiert zu TimeSeries.
        
        Args:
            item_id: GW2 Item ID
            days_back: Anzahl Tage in die Vergangenheit
            value_column: Spalte für die Werte ("buy_unit_price" oder "sell_unit_price")
            
        Returns:
            Darts TimeSeries
        """
        pass
    
    @abstractmethod
    def split(
        self,
        series: TimeSeries,
        ratio: float = 0.8
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Teilt die TimeSeries in Training und Test.
        
        Args:
            series: Vollständige TimeSeries
            ratio: Anteil für Training (0.0-1.0)
            
        Returns:
            Tuple (train_series, test_series)
        """
        pass
    
    @abstractmethod
    def train(self, train_series: TimeSeries, **kwargs) -> None:
        """
        Trainiert das Modell.
        
        Args:
            train_series: Trainings-TimeSeries
            **kwargs: Zusätzliche Trainingsparameter
        """
        pass
    
    @abstractmethod
    def backtest(
        self,
        series: TimeSeries,
        config: "BacktestConfig"
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Führt Walk-Forward Backtesting durch.
        
        Args:
            series: Vollständige TimeSeries
            config: Backtest-Konfiguration
            
        Returns:
            Tuple (actuals, forecasts) als TimeSeries
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        actuals: TimeSeries,
        forecasts: TimeSeries
    ) -> Dict[str, float]:
        """
        Berechnet Evaluations-Metriken.
        
        Args:
            actuals: Echte Werte
            forecasts: Vorhergesagte Werte
            
        Returns:
            Dict mit Metriken: {"mape": ..., "rmse": ..., "mae": ..., "smape": ...}
        """
        pass
    
    @abstractmethod
    def run(
        self,
        item_id: int,
        days_back: int = 30,
        mode: str = "backtest",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Führt die komplette Pipeline aus.
        
        Args:
            item_id: GW2 Item ID
            days_back: Anzahl Tage Daten
            mode: "backtest" oder "single_split"
            **kwargs: Weitere Parameter
            
        Returns:
            Dict mit Ergebnissen (metrics, forecasts, run_id, etc.)
        """
        pass
```

### 4.6 Pipeline Konfiguration

**Datei:** `src/gw2ml/pipeline/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Konfiguration für Walk-Forward Backtesting"""
    
    start: float = 0.5
    """Startpunkt als Anteil der Serie (0.5 = ab 50%)"""
    
    forecast_horizon: int = 12
    """Anzahl Schritte pro Forecast (12 * 5min = 1 Stunde)"""
    
    stride: int = 1
    """Schritte zwischen Forecasts (1 = jeden Step)"""
    
    retrain: bool = False
    """Ob das Modell bei jedem Step neu trainiert wird"""
    
    overlap_end: bool = True
    """Ob Forecasts über das Serienende hinausgehen dürfen"""


@dataclass
class PipelineConfig:
    """Allgemeine Pipeline-Konfiguration"""
    
    value_column: str = "buy_unit_price"
    """Spalte für die Werte"""
    
    split_ratio: float = 0.8
    """Train/Test Split Ratio (nur für mode='single_split')"""
    
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    """Backtest-Konfiguration"""
    
    log_plots: bool = True
    """Ob Plots als MLflow Artifacts geloggt werden"""
    
    log_forecasts: bool = True
    """Ob Forecasts als CSV geloggt werden"""
```

### 4.7 StandardPipeline

**Datei:** `src/gw2ml/pipeline/standard.py`

```python
import mlflow
from darts import TimeSeries
from darts.metrics import mape, rmse, mae, smape
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from .base import BasePipeline
from .config import PipelineConfig, BacktestConfig
from ..modeling.base import BaseModel
from ..data.retriever import get_item_prices  # Annahme: existiert oder wird erstellt


class StandardPipeline(BasePipeline):
    """
    Standard-Pipeline mit MLflow-Integration.
    
    Unterstützt zwei Modi:
    - "backtest": Walk-Forward Validation (empfohlen)
    - "single_split": Einfacher Train/Test Split (schneller)
    """
    
    def __init__(
        self,
        model: BaseModel,
        experiment_name: str,
        config: Optional[PipelineConfig] = None
    ):
        super().__init__(model, experiment_name)
        self.config = config or PipelineConfig()
        self._forecasts: Optional[TimeSeries] = None
        self._actuals: Optional[TimeSeries] = None
    
    def load_data(
        self,
        item_id: int,
        days_back: int = 30,
        value_column: Optional[str] = None
    ) -> TimeSeries:
        """Lädt GW2 Preisdaten und konvertiert zu TimeSeries"""
        value_col = value_column or self.config.value_column
        
        # Daten aus DB laden (nutzt existierende gw2ml.data Module)
        df = get_item_prices(item_id, days_back)
        
        # Zu TimeSeries konvertieren
        df['fetched_at'] = pd.to_datetime(df['fetched_at'])
        if df['fetched_at'].dt.tz is not None:
            df['fetched_at'] = df['fetched_at'].dt.tz_localize(None)
        df = df.set_index('fetched_at').sort_index()
        
        series = TimeSeries.from_dataframe(
            df,
            value_cols=[value_col]
        )
        
        return series
    
    def split(
        self,
        series: TimeSeries,
        ratio: Optional[float] = None
    ) -> Tuple[TimeSeries, TimeSeries]:
        """Teilt in Training und Test"""
        split_ratio = ratio or self.config.split_ratio
        return series.split_after(split_ratio)
    
    def train(self, train_series: TimeSeries, **kwargs) -> None:
        """Trainiert das Modell"""
        self.model.fit(train_series, **kwargs)
    
    def backtest(
        self,
        series: TimeSeries,
        config: Optional[BacktestConfig] = None
    ) -> Tuple[TimeSeries, TimeSeries]:
        """Walk-Forward Backtesting"""
        cfg = config or self.config.backtest
        
        forecasts = self.model.historical_forecasts(
            series=series,
            start=cfg.start,
            forecast_horizon=cfg.forecast_horizon,
            stride=cfg.stride,
            retrain=cfg.retrain,
            overlap_end=cfg.overlap_end
        )
        
        # Actuals für den Backtest-Zeitraum extrahieren
        actuals = series.slice_intersect(forecasts)
        
        return actuals, forecasts
    
    def evaluate(
        self,
        actuals: TimeSeries,
        forecasts: TimeSeries
    ) -> Dict[str, float]:
        """Berechnet Metriken"""
        metrics = {
            "rmse": rmse(actuals, forecasts),
            "mae": mae(actuals, forecasts),
            "smape": smape(actuals, forecasts),
        }
        
        # MAPE nur wenn alle Werte positiv
        if (actuals.values() > 0).all():
            metrics["mape"] = mape(actuals, forecasts)
        else:
            metrics["mape"] = float("nan")
        
        return metrics
    
    def _create_forecast_plot(
        self,
        series: TimeSeries,
        forecasts: TimeSeries,
        title: str
    ) -> plt.Figure:
        """Erstellt Visualisierung"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        series.plot(ax=ax, label="Actual", linewidth=1.5)
        forecasts.plot(ax=ax, label="Forecast", linewidth=2, color="red", linestyle="--")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def run(
        self,
        item_id: int,
        days_back: int = 30,
        mode: str = "backtest",
        backtest_config: Optional[BacktestConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Führt die komplette Pipeline aus.
        
        Args:
            item_id: GW2 Item ID
            days_back: Anzahl Tage Daten
            mode: "backtest" oder "single_split"
            backtest_config: Optionale Backtest-Konfiguration
            
        Returns:
            {
                "metrics": {"mape": ..., "rmse": ..., ...},
                "forecasts": TimeSeries,
                "actuals": TimeSeries,
                "model_params": {...},
                "mlflow_run_id": "..."
            }
        """
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=f"{self.model.name}_item_{item_id}") as run:
            # Log Konfiguration
            mlflow.log_param("model_name", self.model.name)
            mlflow.log_param("item_id", item_id)
            mlflow.log_param("days_back", days_back)
            mlflow.log_param("mode", mode)
            mlflow.log_params(self.model.get_params())
            
            # Daten laden
            series = self.load_data(item_id, days_back)
            mlflow.log_param("series_length", len(series))
            
            if mode == "backtest":
                # Backtest Mode
                cfg = backtest_config or self.config.backtest
                mlflow.log_param("backtest_start", cfg.start)
                mlflow.log_param("backtest_horizon", cfg.forecast_horizon)
                mlflow.log_param("backtest_stride", cfg.stride)
                mlflow.log_param("backtest_retrain", cfg.retrain)
                
                # Training auf erstem Teil
                train_end = int(len(series) * cfg.start)
                train_series = series[:train_end]
                self.train(train_series)
                
                # Backtesting
                actuals, forecasts = self.backtest(series, cfg)
                
            else:  # single_split
                train, test = self.split(series)
                self.train(train)
                forecasts = self.model.predict(len(test))
                actuals = test
            
            # Evaluation
            metrics = self.evaluate(actuals, forecasts)
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            
            # Plots
            if self.config.log_plots:
                fig = self._create_forecast_plot(
                    series, forecasts,
                    f"{self.model.name} - Item {item_id}"
                )
                mlflow.log_figure(fig, "forecast_plot.png")
                plt.close(fig)
            
            # Forecasts als Artifact
            if self.config.log_forecasts:
                forecasts.to_csv("forecasts.csv")
                mlflow.log_artifact("forecasts.csv")
            
            self._forecasts = forecasts
            self._actuals = actuals
            
            return {
                "metrics": metrics,
                "forecasts": forecasts,
                "actuals": actuals,
                "model_params": self.model.get_params(),
                "mlflow_run_id": run.info.run_id
            }
```

---

## 5. Verwendungsbeispiele

### 5.1 Einfache Verwendung

```python
from gw2ml.modeling import ARIMAModel
from gw2ml.pipeline import StandardPipeline

# Modell erstellen
model = ARIMAModel(p=2, d=1, q=2)

# Pipeline erstellen
pipeline = StandardPipeline(model, experiment_name="arima_copper_ore")

# Ausführen mit Backtesting
results = pipeline.run(
    item_id=19976,      # Mystic Coin
    days_back=30,
    mode="backtest"
)

print(f"MAPE: {results['metrics']['mape']:.2f}%")
print(f"RMSE: {results['metrics']['rmse']:.2f}")
```

### 5.2 Mit Factory

```python
from gw2ml.modeling import create_pipeline

pipeline = create_pipeline(
    model_name="ExponentialSmoothing",
    experiment_name="exp_smooth_test",
    model_params={"trend": "add", "damped": True}
)

results = pipeline.run(item_id=19976, days_back=14)
```

### 5.3 Backtest-Konfiguration anpassen

```python
from gw2ml.pipeline import BacktestConfig

config = BacktestConfig(
    start=0.6,              # Ab 60% der Daten
    forecast_horizon=24,    # 2 Stunden voraus
    stride=12,              # Alle 12 Steps (1 Stunde)
    retrain=True            # Bei jedem Step neu trainieren
)

results = pipeline.run(
    item_id=19976,
    mode="backtest",
    backtest_config=config
)
```

### 5.4 Eigenes Modell erstellen

```python
from gw2ml.modeling import BaseModel, register_model
from darts.models import Prophet


@register_model
class ProphetModel(BaseModel):
    name = "Prophet"
    default_params = {"yearly_seasonality": False}
    
    def __init__(self, **kwargs):
        self.params = {**self.default_params, **kwargs}
        self._model = None
    
    def build_model(self, **kwargs):
        return Prophet(**{**self.params, **kwargs})
    
    def fit(self, series, **kwargs):
        self._model = self.build_model()
        self._model.fit(series)
        return self
    
    def predict(self, n, **kwargs):
        return self._model.predict(n)
    
    def get_params(self):
        return self.params


# Jetzt verfügbar:
from gw2ml.modeling import ModelRegistry
print(ModelRegistry.list_models())  # ['ARIMA', 'ExponentialSmoothing', 'Prophet']
```

---

## 6. MLflow Integration

### 6.1 Geloggte Daten

| Kategorie | Inhalt |
|-----------|--------|
| **Parameters** | model_name, item_id, days_back, mode, alle model_params, backtest_config |
| **Metrics** | mape, rmse, mae, smape |
| **Artifacts** | forecast_plot.png, forecasts.csv |

### 6.2 Experiment-Struktur

```
Experiment: "arima_copper_ore"
├── Run: ARIMA_item_19976
│   ├── Parameters
│   │   ├── model_name: "ARIMA"
│   │   ├── p: 2
│   │   ├── d: 1
│   │   ├── q: 2
│   │   ├── item_id: 19976
│   │   └── ...
│   ├── Metrics
│   │   ├── mape: 2.13
│   │   ├── rmse: 15.4
│   │   └── ...
│   └── Artifacts
│       ├── forecast_plot.png
│       └── forecasts.csv
```

---

## 7. Erweiterbarkeit

### 7.1 Neue Modelle hinzufügen

1. Erstelle neue Datei in `src/gw2ml/modeling/`
2. Erbe von `BaseModel`
3. Implementiere alle abstrakten Methoden
4. Dekoriere mit `@register_model`

### 7.2 Geplante Erweiterungen

| Modell | Status | Bemerkung |
|--------|--------|-----------|
| ARIMA | 🔜 Zu implementieren | Basis-Statistik |
| ExponentialSmoothing | 🔜 Zu implementieren | Basis-Statistik |
| XGBoost | 🔄 Existiert bereits | Refactoring nötig |
| Prophet | 📋 Geplant | Nach Basis-Modellen |
| LSTM/Transformer | 📋 Später | Custom Deep Learning |

---

## 8. Offene Punkte

- [ ] `get_item_prices()` Funktion in `gw2ml.data` erstellen/anpassen
- [ ] Unit Tests für BaseModel und BasePipeline
- [ ] Integration Tests mit echter DB-Verbindung
- [ ] Hyperparameter-Search als optionale Pipeline-Erweiterung

---

## 9. Anhang

### 9.1 GW2 Item IDs (Beispiele)

| Item | ID |
|------|-----|
| Mystic Coin | 19976 |
| Iron Ore | 19699 |
| Mithril Ore | 19700 |
| Glob of Ectoplasm | 19721 |

### 9.2 Zeitintervalle

- Daten werden alle **5 Minuten** abgefragt
- 1 Tag = 288 Datenpunkte
- 1 Woche = 2016 Datenpunkte
- 1 Monat ≈ 8640 Datenpunkte
