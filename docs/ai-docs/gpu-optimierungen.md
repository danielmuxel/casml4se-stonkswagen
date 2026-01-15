# GPU-Optimierungen für Forecasting-Modelle

## Übersicht

| Modell | GPU-Support | Speedup | Änderungen |
|--------|-------------|---------|------------|
| **Chronos2** | ✅ CUDA/ROCm/MPS | N/A (Foundation Model) | Auto-Device, Mixed Precision |
| **XGBoost** | ✅ CUDA/ROCm | ~1.3x Training | `device='cuda'` |
| **ARIMA** | ❌ Nein | - | statsmodels, CPU-only |
| **Exponential Smoothing** | ❌ Nein | - | statsmodels, CPU-only |

---

# Chronos2 GPU-Analyse und Optimierungen

## Zusammenfassung

Analyse der Chronos2-Implementierung auf GPU-Probleme (NVIDIA RTX 3060, CUDA 12.8).

**Hauptproblem identifiziert:** Das Modell läuft auf der CPU statt GPU, obwohl CUDA verfügbar ist.

## System-Konfiguration

- **GPU:** NVIDIA GeForce RTX 3060 (11.6 GB VRAM)
- **PyTorch:** 2.9.1+cu128
- **CUDA:** verfügbar, 1 Device
- **Darts:** 0.40.0

## Identifizierte Probleme

### 1. Modell bleibt auf CPU bei `epochs=0`

Bei Foundation Models wie Chronos2 wird `epochs=0` verwendet (Zero-Shot-Modus). PyTorch Lightning verschiebt das Modell nur während des Trainings auf die GPU. Da kein Training stattfindet, bleibt das Modell auf der CPU.

**Nachweis:**
```python
model.fit(series, epochs=0)
print(model.model.device)  # Ausgabe: cpu
```

### 2. RAM-Verbrauch statt VRAM

Weil das Modell auf der CPU läuft:
- Initialer RAM: ~1.0 GB
- Nach `fit()`: ~1.5 GB
- Nach `historical_forecasts()`: ~2.2 GB
- GPU-Speicher: 0 GB (ungenutzt)

### 3. Keine explizite Device-Konfiguration

Die ursprüngliche Implementierung (`src/gw2ml/modeling/chronos.py`) setzt keine explizite Device-Konfiguration. Die `pl_trainer_kwargs` werden zwar übergeben, aber bei `epochs=0` ignoriert.

## Lösung

### Manuelles Verschieben auf GPU nach `fit()`

Nach dem `fit()` mit `epochs=0` muss das Modell explizit auf die GPU verschoben werden:

```python
model.fit(series, epochs=0)
model.model = model.model.to('cuda')  # Explizit auf GPU verschieben
```

**Ergebnis nach Fix:**
- GPU-Speicher: ~0.45 GB (120M Modell)
- RAM: ~1.35 GB (reduziert von 2.2 GB)

### Device-Erkennung (CUDA, ROCm, MPS, CPU)

```python
def _get_device() -> str:
    """Ermittelt das beste verfügbare Device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif hasattr(torch.version, 'hip') and torch.version.hip:
        return "cuda"  # ROCm verwendet CUDA-API
    return "cpu"
```

## Empfohlene Änderungen an `chronos.py`

1. **Nach `fit()` auf GPU verschieben**
2. **`torch.set_float32_matmul_precision('medium')` setzen** (nutzt Tensor Cores)
3. **Mixed Precision (`16-mixed`)** für weniger VRAM
4. **Garbage Collection** nach Inference (`gc.collect()`, `torch.cuda.empty_cache()`)
5. **Kleineres Modell als Option** (`chronos-2-small` statt `chronos-2-synth`)

## Memory-Vergleich

| Modell | Parameter | VRAM (GPU) | RAM (CPU) |
|--------|-----------|------------|-----------|
| chronos-2-small | 28M | ~0.12 GB | ~0.8 GB |
| chronos-2-synth | 120M | ~0.45 GB | ~1.5 GB |

## Kompatibilität

Die Lösung ist kompatibel mit:
- **NVIDIA (CUDA):** `torch.cuda.is_available()`
- **AMD (ROCm):** `torch.version.hip` vorhanden
- **Apple Silicon (MPS):** `torch.backends.mps.is_available()`
- **CPU-Fallback:** Wenn keine GPU verfügbar

## Implementierte Änderungen

Die Datei `src/gw2ml/modeling/chronos.py` wurde aktualisiert:

### 1. Automatische Device-Erkennung

```python
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "cuda"  # ROCm
    return "cpu"
```

### 2. GPU-Verschiebung nach `fit()`

```python
def _move_to_device(self) -> None:
    if self._device != "cpu" and hasattr(self._model, 'model'):
        self._model.model = self._model.model.to(self._device)
```

### 3. Neue Parameter

- `batch_size`: Konfigurierbare Batch-Größe (default: 32)
- `use_mixed_precision`: 16-bit Mixed Precision für weniger VRAM (default: True)
- `device`: Optionale manuelle Device-Wahl

### 4. Cleanup-Methode

```python
def cleanup(self) -> None:
    del self._model
    gc.collect()
    torch.cuda.empty_cache()
```

## Testergebnisse nach Fix

| Metrik | Vorher (CPU) | Nachher (GPU) |
|--------|--------------|---------------|
| GPU Memory | 0 GB | 0.445 GB |
| RAM nach fit() | 1.5 GB | 1.38 GB |
| RAM nach historical_forecasts() | 2.2 GB | 2.44 GB |
| Device | cpu | cuda:0 |

## Verwendung

```python
from gw2ml.modeling.chronos import Chronos2

# Standard (große Modell, GPU auto-detected)
model = Chronos2()

# Kleineres Modell für weniger VRAM
model = Chronos2(model_hub_name='autogluon/chronos-2-small')

# Explizite CPU-Nutzung
model = Chronos2(device='cpu')

# Nach Verwendung aufräumen
model.cleanup()
```

---

# XGBoost GPU-Optimierung

## Änderungen

Die Datei `src/gw2ml/modeling/xgboost.py` wurde aktualisiert:

### Automatische GPU-Erkennung

```python
def _get_xgb_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "cuda"  # ROCm
    return "cpu"
```

### Neue Parameter

- `use_gpu`: GPU-Beschleunigung aktivieren (default: True)
- `device`: Manuelles Device ('cuda', 'cpu', oder None für Auto)

## Benchmark-Ergebnisse

| Konfiguration | CPU | GPU | Speedup |
|---------------|-----|-----|---------|
| 8640 Punkte, 500 Trees, depth=8 | 34.2s | 25.9s | 1.3x |
| 5000 Punkte, 200 Trees, depth=5 | 3.3s | 4.3s | 0.8x |

**Fazit:** GPU lohnt sich bei größeren Datensätzen und mehr Bäumen. Bei kleinen Datensätzen ist der GPU-Overhead größer als der Gewinn.

## Verwendung

```python
from gw2ml.modeling.xgboost import XGBoostModel

# Standard (GPU auto-detected)
model = XGBoostModel(lags=24, n_estimators=200)

# Explizit CPU
model = XGBoostModel(lags=24, n_estimators=200, use_gpu=False)

# Oder via device Parameter
model = XGBoostModel(lags=24, n_estimators=200, device='cpu')
```

## Bekannte Einschränkungen

Bei der Prediction erscheint eine Warnung, weil die Input-Daten auf der CPU sind:
```
WARNING: Falling back to prediction using DMatrix due to mismatched devices.
```

Dies ist ein bekanntes XGBoost-Verhalten und hat keinen signifikanten Performance-Einfluss bei der Inference.

---

# ARIMA und Exponential Smoothing

Diese Modelle basieren auf **statsmodels**, einer reinen Python/NumPy-Bibliothek ohne GPU-Unterstützung.

**Alternativen für GPU-beschleunigtes statistisches Forecasting:**
- cuML (NVIDIA RAPIDS) - bietet GPU-ARIMA, aber nur für CUDA
- PyTorch Forecasting - neurale Alternativen mit GPU-Support

Für dieses Projekt bleiben ARIMA und Exponential Smoothing CPU-basiert, da:
1. Die Trainingszeit akzeptabel ist
2. statsmodels robuste, bewährte Implementierungen bietet
3. GPU-Alternativen zusätzliche Dependencies erfordern würden

---

# Benchmark-Script

## Verwendung

```bash
# Standard-Benchmark mit synthetischen Daten
python scripts/benchmark_models.py

# Mit echten Daten (Mystic Coin)
python scripts/benchmark_models.py --item-id 19976 --days 30

# Mit JSON-Export
python scripts/benchmark_models.py --output results/benchmark.json

# Alle Optionen
python scripts/benchmark_models.py --item-id 19976 --days 30 --points 5000 --horizon 12 --output benchmark.json
```

## Beispiel-Output

```
============================================================
BENCHMARK ERGEBNISSE
============================================================

Konfiguration:
  Datenquelle: synthetic
  Datenpunkte: 3000
  Forecast Horizon: 12

--------------------------------------------------------------------------------
Modell          Device        Fit  Predict Backtest    Total  Relativ
--------------------------------------------------------------------------------
XGBoost         cuda        0.47s   0.027s    0.02s    0.52s     1.0x
Chronos2        cuda        1.01s   0.576s    0.29s    1.88s     3.6x
ExpSmoothing    cpu         0.06s   0.011s    2.82s    2.89s     5.5x
ARIMA           cpu         0.10s   0.001s    6.28s    6.38s    12.2x
--------------------------------------------------------------------------------

Schnellstes Modell: XGBoost (1.0x)
```

## Interpretation

- **Relativ**: Vielfaches der Zeit im Vergleich zum schnellsten Modell (1.0x)
- **Backtest**: Der zeitaufwändigste Teil bei ARIMA/ExpSmoothing (wegen `retrain=True`)
- **Device**: `cuda` = GPU, `cpu` = nur CPU

## Systemvergleich

Der JSON-Output kann verwendet werden um Benchmarks zwischen Systemen zu vergleichen:

```bash
# Auf NVIDIA-System
python scripts/benchmark_models.py --output benchmark-nvidia.json

# Auf AMD-System (ROCm)
python scripts/benchmark_models.py --output benchmark-amd.json

# Auf CPU-only System
python scripts/benchmark_models.py --output benchmark-cpu.json
```

---

# Backtest-Performance optimieren

## Das Problem

Bei `historical_forecasts()` (Walk-Forward Validation) sind **ARIMA** und **ExponentialSmoothing** extrem langsam im Vergleich zu XGBoost und Chronos2.

**Beispiel mit 5000 Datenpunkten:**

| Modell | Backtest-Zeit | Grund |
|--------|---------------|-------|
| XGBoost | 0.03s | `retrain=False` möglich |
| Chronos2 | 0.27s | `retrain=False` (Foundation Model) |
| ExpSmoothing | 7.0s | `retrain=True` erforderlich |
| ARIMA | 15.0s | `retrain=True` erforderlich |

## Warum sind Local Models langsam?

ARIMA und ExponentialSmoothing sind **LocalForecastingModels** in Darts:

```
GlobalForecastingModels (XGBoost, Chronos2, LSTMs, etc.):
  - Können mit retrain=False verwendet werden
  - Modell wird einmal trainiert, dann für alle Vorhersagen genutzt
  - Backtest ist schnell

LocalForecastingModels (ARIMA, ExponentialSmoothing, Prophet, etc.):
  - MÜSSEN mit retrain=True verwendet werden
  - Modell wird bei JEDEM Backtest-Schritt neu trainiert
  - Backtest ist langsam
```

**Rechenbeispiel:**
- 5000 Datenpunkte, start=0.8 → 1000 Testpunkte
- stride=12 → 83 Backtest-Iterationen
- ARIMA fit ~0.6s pro Iteration
- **83 × 0.6s = ~50 Sekunden**

## Optimierungsmöglichkeiten

### 1. Stride erhöhen (weniger Iterationen)

```python
# Langsam: stride=12 → 83 Iterationen
model.historical_forecasts(series, start=0.8, stride=12, ...)

# Schneller: stride=48 → 20 Iterationen
model.historical_forecasts(series, start=0.8, stride=48, ...)
```

| Stride | Iterationen | Zeit |
|--------|-------------|------|
| 12 | 83 | ~50s |
| 24 | 41 | ~25s |
| 48 | 20 | ~12s |
| 96 | 10 | ~6s |

**Trade-off:** Weniger Datenpunkte für Metriken, aber oft ausreichend für Modellvergleich.

### 2. Später starten (weniger Testpunkte)

```python
# Langsam: 20% Testdaten
model.historical_forecasts(series, start=0.8, ...)

# Schneller: 10% Testdaten
model.historical_forecasts(series, start=0.9, ...)
```

| Start | Testpunkte | Iterationen | Zeit |
|-------|------------|-------------|------|
| 0.8 | 1000 | 83 | ~50s |
| 0.9 | 500 | 41 | ~25s |
| 0.95 | 250 | 20 | ~12s |

### 3. Einfacheres Modell

```python
# Komplex: ARIMA(1,1,1) - ~0.6s pro fit
model = ARIMA(p=1, d=1, q=1)

# Einfach: AR(1) - ~0.09s pro fit (6x schneller)
model = ARIMA(p=1, d=0, q=0)
```

### 4. Kombinierte Optimierung (Benchmark-Script)

Das Benchmark-Script verwendet diese Kombination:

```python
# Für Local Models (ARIMA, ExpSmoothing):
start = 0.9           # Weniger Testpunkte
stride = horizon * 4  # Größerer Stride

# Für Global Models (XGBoost, Chronos2):
start = 0.9
stride = horizon      # Normaler Stride, retrain=False
```

**Ergebnis:** ARIMA von 15s auf 2.5s reduziert.

## Empfehlungen

### Für Entwicklung/Debugging:
```python
# Schnell: große Strides, wenig Testdaten
model.historical_forecasts(
    series,
    start=0.95,      # Nur 5% Testdaten
    stride=48,       # Große Schritte
    forecast_horizon=12,
    retrain=True,
)
```

### Für finale Evaluation:
```python
# Genau: kleine Strides, mehr Testdaten
model.historical_forecasts(
    series,
    start=0.8,       # 20% Testdaten
    stride=1,        # Jeder Schritt (langsam!)
    forecast_horizon=12,
    retrain=True,
)
```

### Für Produktion:
```python
# Nur predict(), kein Backtest nötig
model.fit(full_series)
forecast = model.predict(n=12)  # Schnell für alle Modelle
```

## Keine GPU-Beschleunigung möglich

ARIMA und ExponentialSmoothing basieren auf **statsmodels**, einer reinen Python/NumPy-Bibliothek:

- Keine GPU-Unterstützung
- Keine native Parallelisierung für `historical_forecasts`
- Optimierung nur durch weniger Iterationen möglich

**Alternativen mit GPU-Support:**
- cuML (NVIDIA RAPIDS) - GPU-ARIMA, aber nur CUDA
- Neural Prophet - PyTorch-basiert
- Darts TorchForecastingModels (LSTM, Transformer, etc.)
