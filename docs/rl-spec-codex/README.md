# RL Spec (Codex)

Dieses Dokument beschreibt ein erstes Design fuer ein Reinforcement-Learning Modell,
das auf GW2 Tick-Daten (alle 5 Minuten pro Item) Buy/Sell Entscheidungen trifft.
Es ist als Arbeitsgrundlage gedacht und kann spaeter erweitert werden.

## 1. Kontext und vorhandene Bausteine

- Datenzugriff ueber `src/gw2ml/data`:
  - `DatabaseClient` und `database_queries.get_prices(...)` liefern Tick-Daten
    mit Feldern wie `buy_quantity`, `buy_unit_price`, `sell_quantity`,
    `sell_unit_price`, `fetched_at`.
  - `get_prices_snapshot(...)` kann pro Zeitpunkt den letzten Preis je Item liefern.
- ML/Workflow-Orchestrierung ist unter `ml/pipelines/` vorgesehen.
- MLflow Infrastruktur ist unter `infra/MLFlow/` vorhanden.

Diese Bausteine reichen als Fundament fuer Datenzugriff, Pipeline und Logging.

## 2. Zielbild

- Agent entscheidet bei jedem Tick (alle 5 Minuten) pro Item, ob:
  - **HOLD** (keine Aktion)
  - **BUY_1** (genau 1 Stueck kaufen)
  - **SELL_K** (1 bis alle vorhandenen Stuecke verkaufen)
- Aktion wird pro Item pro Tick bewertet.
- Fokus auf profitables Trading unter realistischen Transaktionskosten.

## 3. Environment-Design (Gym-Style)

### 3.1 Episode

- Episode = Zeitfenster eines Items, z.B. 30 Tage Tick-Daten.
- Startzustand:
  - Cash: frei definierbar (z.B. 10_000 Coins)
  - Inventory: 0
- Ende der Episode:
  - Letzter Tick oder max. N Ticks
  - Optionale Zwangsliquidation am letzten Tick

### 3.2 Beobachtung (State)

Empfohlener State (pro Item):

- Marktfeatures (rollende Fenster):
  - `buy_unit_price`, `sell_unit_price`, Spread
  - Returns (1, 3, 12 Ticks)
  - Rolling Volumen (`buy_quantity`, `sell_quantity`)
  - Volatilitaet (z.B. Std. der Returns)
- Agenteninterner Zustand:
  - Cash
  - Inventory (Anzahl Items)
  - Offene Orders (optional)
  - Letzte Aktion (optional)

### 3.3 Aktionen

Minimal:

- `0 = HOLD`
- `1 = BUY_1`
- `2 = SELL_1`
- `3 = SELL_ALL`

Optional erweiterbar:

- Parameterisierte Aktion: `SELL_K` mit `K in [1..inventory]`.
  - Fuer DQN: diskretisieren (z.B. SELL_1, SELL_5, SELL_10, SELL_ALL)
  - Fuer PPO/SAC: separat `action_type` + `sell_qty` als kontinuierlicher Wert

### 3.4 Ausfuehrungsmodell

- Marktorderannahme:
  - Kauf zu `sell_unit_price` (Ask)
  - Verkauf zu `buy_unit_price` (Bid)
- Slippage und Fees als Konfiguration:
  - `fee_buy`, `fee_sell`
  - Optional: Spread/Volumen-basierte Slippage

## 4. Reward-Design

Grundidee: Aenderung des Nettovermoegens pro Tick.

- `net_worth = cash + inventory * mid_price`
- Reward: `delta_net_worth - transaction_fees - risk_penalty`

Moegliche Penalties:

- Inventar-Risiko (zu hohe Inventar-Bestaende)
- Turnover-Kosten (zu viele Trades)
- Drawdown-Penalty

## 5. Daten-Splits und Backtesting

- Zeitbasierte Splits (kein Leakage):
  - Train: aeltester Zeitraum
  - Val: mittlerer Zeitraum
  - Test: juengster Zeitraum
- Evaluationsmetriken:
  - Total Return, Sharpe, Max Drawdown
  - Win/Loss Ratio, Avg. Holding Time
  - Turnover, Fee-Anteil

## 6. Baselines

- Buy-and-Hold pro Item
- Naiver Spread-Capture (buy bei Spread > Schwelle, sell bei Ruecklauf)
- Momentum-Baseline (kauf nach positivem Return, verkauf nach negativem)

## 7. Trainingsstrategie

Start klein:

- Einzel-Item Agent (reduziert Action-Space)
- DQN oder PPO
- Offline Training auf historischen Ticks

Spaeter:

- Multi-Item Agent (Shared Policy)
- Risk-Aware Objective (CVaR, Drawdown-Penalty)

## 8. Pipeline-Integration (Vorschlag)

- `ml/pipelines/rl_train.py`: Training fuer RL-Agent
- `ml/pipelines/rl_eval.py`: Backtesting und Reporting
- `src/gw2ml/rl/`: Environment, Reward, Policies
- Logging nach MLflow (Metrics + Artifacts)

## 9. Offene Fragen (bitte ausfuellen)

- Fee-Struktur der Trades (GW2 Trading Post?)
- Startkapital pro Episode
- Max Inventar pro Item
- Soll es offene Orders geben oder nur Market Orders?
- Multi-Item vs. Single-Item Training
- Welche GW2 API Felder sollen zusaetzlich in den State?

