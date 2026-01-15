# Apps Spezifikation

## Übersicht

Die `apps/` Ordnerstruktur enthält die Benutzeroberflächen für die GW2ML-Anwendung:

```
apps/
├── README.md                    # (veraltet - aktualisieren)
├── api/
│   └── main.py                  # FastAPI Backend
└── streamlit/
    ├── gw2_app.py               # Haupt-Einstiegspunkt mit Navigation
    ├── forecast_app.py          # Forecast Tab
    ├── evaluation_app.py        # Evaluation/Backtest Tab
    └── item_analysis_app.py     # Item Analysis Tab
```

---

## Komponenten

### 1. FastAPI Backend (`apps/api/main.py`)

**Zweck:** REST-API für Training und Forecasting

**Endpoints:**

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/health` | GET | Health-Check |
| `/train` | POST | Trainiert Modelle für gegebene Item-IDs |
| `/forecast` | POST | Generiert Forecast für ein Item |

**Starten:**
```bash
uv run fastapi dev apps/api/main.py --host 0.0.0.0 --port 8000
```

**Request-Schemas:**

```python
# TrainRequest
{
    "item_ids": [19976, 19699],  # Required
    "override_config": {...}     # Optional
}

# ForecastRequest
{
    "item_id": 19976,            # Required
    "override_config": {...}     # Optional
}
```

**Erlaubte Config-Keys:** `data`, `split`, `forecast`, `metric`, `models`

---

### 2. Streamlit Frontend

#### 2.1 Haupt-App (`apps/streamlit/gw2_app.py`)

**Zweck:** Einstiegspunkt mit Tab-Navigation

**Tabs:**
- `Forecast` - Forecasting (ohne Training/Backtest)
- `Evaluation` - Training & Backtests
- `Item Analysis` - Statistische Analysen

**Starten:**
```bash
uv run streamlit run apps/streamlit/gw2_app.py
```

#### 2.2 Forecast Tab (`apps/streamlit/forecast_app.py`)

**Zweck:** Reines Forecasting (ohne Training/Backtest)

**Features:**
- Item-ID Eingabe
- Konfiguration: Zeitbereich, Resampling, Horizon
- Forecast für alle Value Columns (`buy_unit_price`, `sell_unit_price`, Mengen, Demand)
- Visualisierung: Historie + Forecast in durchgehender Timeline
- Profitabilitäts-Indikator für Buy/Sell (Break-even)

#### 2.3 Evaluation Tab (`apps/streamlit/evaluation_app.py`)

**Zweck:** Training & Backtests/Evaluation

**Features:**
- Item-ID Eingabe
- Konfiguration: Zeitbereich, Resampling, Value Column, Horizon
- Modell-/Metrik-Auswahl
- Optionales Retrain vor Evaluation
- Visualisierung: Future Forecast + Historical Backtest
- Metriken-Tabelle

#### 2.4 Item Analysis Tab (`apps/streamlit/item_analysis_app.py`)

**Zweck:** Statistische Analyse von Items

**Features:**
- Multi-Item-Auswahl mit Suche
- History Graph (mit Resampling)
- Distribution Analysis (Seaborn Pairplot)
- ADF Test (Stationarity)

**Caching:**
- 7 Tage TTL
- Disk-Persistenz für große Daten
- Separate Caches für: Items, Series-Batches, ADF-Tests

---

## Durchgeführte Änderungen

### 1. Backtest-Optimierung (Pipeline)

**Problem:** ARIMA und ExponentialSmoothing waren extrem langsam im Backtest (15-50 Sekunden).

**Ursache:** LocalForecastingModels benötigen `retrain=True`, was bei jedem Backtest-Schritt ein komplettes Retraining auslöst.

**Lösung in `src/gw2ml/pipelines/forecast.py`:**

```python
# Vor der Optimierung
backtest_stride = horizon  # z.B. 12

# Nach der Optimierung
is_local_model = model_name in ("ARIMA", "ExponentialSmoothing")
backtest_stride = horizon * 4 if is_local_model else horizon  # z.B. 48 für Local Models
```

**Auswirkung:**
- ARIMA: ~15s → ~2.5s (6x schneller)
- ExpSmoothing: ~7s → ~1s (7x schneller)

### 2. GPU-Optimierung (Modeling)

**Chronos2:** Automatische GPU-Nutzung nach `fit()`
- Device-Erkennung (CUDA, ROCm, MPS, CPU)
- Manuelles Verschieben auf GPU bei `epochs=0`
- Mixed Precision Support

**XGBoost:** GPU-Beschleunigung aktiviert
- `device='cuda'` für CUDA/ROCm
- ~1.3x Speedup bei großen Datensätzen

Siehe `docs/gpu-optimierungen.md` für Details.

---

## Ausstehende Änderungen / TODOs

### Hohe Priorität

1. **`apps/README.md` aktualisieren**
   - Nuxt-Referenz entfernen
   - Streamlit/FastAPI Dokumentation hinzufügen

2. **Error Handling verbessern**
   - GPU-Fehler (MPS/CUDA) besser abfangen
   - Benutzerfreundliche Fehlermeldungen

### Mittlere Priorität

3. **Forecast Tab: Progress Anzeige**
   - Echtzeit-Logs während Training/Forecast
   - Geschätzte Restzeit

4. **Item Analysis: Performance**
   - Batch-Loading für Pairplot optimieren
   - Lazy Loading für große Item-Listen

5. **API: Asynchrone Endpoints**
   - Training als Background-Task
   - Polling-Endpoint für Status

### Niedrige Priorität

6. **Tests für Apps**
   - Unit Tests für Validierung
   - Integration Tests für API

7. **Konfiguration externalisieren**
   - Environment Variables für Ports
   - YAML/JSON Config für Defaults

---

## Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
│  ┌───────────────────┐  ┌─────────────────────────────────┐ │
│  │    gw2_app.py     │  │        Sidebar Navigation      │ │
│  │   (Entry Point)   │  │   - Forecast                   │ │
│  └─────────┬─────────┘  │   - Item Analysis              │ │
│            │            └─────────────────────────────────┘ │
│  ┌─────────▼─────────┐  ┌─────────────────────────────────┐ │
│  │  forecast_app.py  │  │   item_analysis_app.py         │ │
│  │  - Train/Forecast │  │   - History Graph              │ │
│  │  - Visualisierung │  │   - ADF Test                   │ │
│  │  - Metriken       │  │   - Distribution               │ │
│  └─────────┬─────────┘  └──────────────┬──────────────────┘ │
└────────────┼───────────────────────────┼────────────────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    gw2ml Package                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ pipelines/      │  │ modeling/       │  │ data/       │  │
│  │ - train.py      │  │ - arima.py      │  │ - loaders   │  │
│  │ - forecast.py   │  │ - chronos.py    │  │             │  │
│  └─────────────────┘  │ - xgboost.py    │  └─────────────┘  │
│                       │ - exp_smooth.py │                   │
│                       └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  apps/api/main.py                                       ││
│  │  - POST /train    → train_items()                       ││
│  │  - POST /forecast → forecast_item()                     ││
│  │  - GET  /health                                         ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Konfiguration

### Streamlit (`forecast_app.py`)

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `days_back` | 30 | Tage Historiedaten |
| `value_column` | buy_unit_price | Preis-Spalte |
| `horizon` | 12 | Forecast-Schritte |
| `models` | alle | Zu verwendende Modelle |
| `primary_metric` | mape | Hauptmetrik |

### Streamlit (`item_analysis_app.py`)

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `history_days` | 30 | Tage für Analyse |
| `resample_rule` | 1h | Resampling-Intervall |
| `show_history` | false | History Graph |
| `show_distribution` | false | Pairplot |
| `show_adf` | false | ADF Test |

### FastAPI

| Env Variable | Default | Beschreibung |
|--------------|---------|--------------|
| `DB_URL` | - | Datenbank-Verbindung |
| Host/Port | 0.0.0.0:8000 | API-Adresse |

---

## Bekannte Einschränkungen

1. **MPS (Apple Silicon):** Einige Operationen fallen auf CPU zurück
2. **Große Item-Listen:** UI kann bei >10.000 Items langsam werden
3. **Concurrent Requests:** FastAPI hat keine Request-Limitierung
4. **Session State:** Streamlit verliert State bei Page Reload

---

## Referenzen

- [GPU-Optimierungen](./gpu-optimierungen.md)
- [Benchmark-Script](../scripts/benchmark_models.py)
- [Forecast Pipeline](../src/gw2ml/pipelines/forecast.py)
