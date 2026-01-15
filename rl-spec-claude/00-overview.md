# GW2 Trading Bot - Reinforcement Learning Spezifikation

## Zielsetzung

Entwicklung eines Reinforcement Learning Agenten für automatisiertes Trading im Guild Wars 2 Trading Post, der auf Basis von 5-Minuten-Tick-Daten aus der Datenbank und GW2 API selbstständig Kauf- und Verkaufsentscheidungen trifft.

## Hauptmerkmale

### Trading Regeln
- **Tick-Frequenz**: Alle 5 Minuten pro Item
- **Kaufregeln**: Buy-Order kann maximal 1 Item kaufen
- **Verkaufsregeln**: Sell-Order kann 1 bis alle verfügbaren Items verkaufen
- **Entscheidungsfreiheit**: Agent entscheidet bei jedem Tick ob und welche Aktion ausgeführt wird

### Bestehende Infrastruktur
Das Projekt verfügt bereits über:
- PostgreSQL Datenbank mit historischen und aktuellen Preisdaten (5-Min Ticks)
- 4 trainierte Forecasting-Modelle (ARIMA, XGBoost, ExponentialSmoothing, Chronos2)
- Backtesting-Framework mit Data-Leakage-Prevention
- FastAPI/Streamlit Serving-Layer
- MLflow Model-Tracking

### Was fehlt
- Trading-Logik und Portfolio-Management
- Order-Execution-System
- RL-Environment und Agent-Implementation
- Reward-Function-Design

## Projektstruktur

Die RL-Spezifikation ist in folgende Dokumente aufgeteilt:

1. **00-overview.md** (dieses Dokument) - Überblick und Zielsetzung
2. **01-rl-environment.md** - RL Environment Design (State, Action, Reward)
3. **02-architecture.md** - Technische Architektur und Integration
4. **03-implementation-plan.md** - Implementierungsplan

## Designprinzipien

1. **Modularität**: RL-Komponenten sollen sauber von bestehenden Forecasting-Modellen getrennt sein
2. **Realismus**: Simulation soll reale GW2 Trading Post Mechaniken abbilden (Gebühren, Latenz, Liquidität)
3. **Evaluierbarkeit**: Backtesting auf historischen Daten mit klaren Metriken
4. **Skalierbarkeit**: System soll für Multiple Items parallel lauffähig sein
5. **Transparenz**: Nachvollziehbare Entscheidungen durch Logging und Visualisierung

## Technologie-Stack (Vorschlag)

- **RL Framework**: Stable-Baselines3 oder RLlib (Ray)
- **Environment**: Gymnasium (ehemals OpenAI Gym)
- **Portfolio-Management**: Custom Python-Klassen
- **Integration**: Erweitert bestehende `src/gw2ml/` Module
- **Evaluation**: Nutzt bestehendes `src/gw2ml/evaluation/backtest.py`

## Erfolgskriterien

1. **Profitabilität**: Agent erzielt positiven ROI auf Test-Set (nach GW2 Trading Post Gebühren)
2. **Stabilität**: Keine katastrophalen Losses (z.B. Max Drawdown < 20%)
3. **Generalisierung**: Funktioniert auf verschiedenen Items und Marktbedingungen
4. **Effizienz**: Training-Zeit akzeptabel für Iteration und Hyperparameter-Tuning
5. **Interpretierbarkeit**: Entscheidungen sind nachvollziehbar und visualisierbar

## Nächste Schritte

Siehe detaillierte Dokumentation in:
- `01-rl-environment.md` für RL-spezifisches Design
- `02-architecture.md` für technische Implementation
- `03-implementation-plan.md` für konkreten Umsetzungsplan
