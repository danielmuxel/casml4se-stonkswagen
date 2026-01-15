# Apps

Dieses Verzeichnis enthält die Benutzeroberflächen der GW2ML-Anwendung.

## Struktur

```
apps/
├── api/
│   └── main.py              # FastAPI Backend (REST-API)
└── streamlit/
    ├── gw2_app.py           # Haupt-Einstiegspunkt
    ├── forecast_app.py      # Training & Forecasting Tab
    └── item_analysis_app.py # Item-Analyse Tab
```

## Schnellstart

**Backend starten:**
```bash
uv run fastapi dev apps/api/main.py --host 0.0.0.0 --port 8000
```

**Frontend starten (anderes Terminal):**
```bash
uv run streamlit run apps/streamlit/gw2_app.py
```

## Dokumentation

Siehe [docs/apps-spezifikation.md](../docs/apps-spezifikation.md) für detaillierte Dokumentation.
