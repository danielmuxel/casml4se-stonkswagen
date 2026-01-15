# RL Environment Design

## Environment-Architektur

### Gymnasium Interface

```python
import gymnasium as gym
from gymnasium import spaces

class GW2TradingEnv(gym.Env):
    """
    GW2 Trading Post Environment für Single-Item Trading
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, item_id: int, data: pd.DataFrame, initial_capital: float = 10000.0):
        super().__init__()
        self.item_id = item_id
        self.data = data  # Historical price data (5-min ticks)
        self.initial_capital = initial_capital

        # State: [price_history, inventory, capital, order_book_features, forecast]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._state_dim(),),
            dtype=np.float32
        )

        # Action: [no_op, buy_1, sell_1, sell_all]
        self.action_space = spaces.Discrete(4)
```

## State Space (Observation)

### State-Komponenten

Der State setzt sich aus mehreren Komponenten zusammen:

#### 1. Price History Features (Sliding Window)
- **Lookback Window**: Letzte N Ticks (z.B. N=12 für 1 Stunde bei 5-Min Ticks)
- **Features pro Tick**:
  - `buy_unit_price` (Buyer Price)
  - `sell_unit_price` (Seller Price)
  - `spread = sell_unit_price - buy_unit_price`
  - `mid_price = (buy_unit_price + sell_unit_price) / 2`
  - `buy_quantity` (Order Book Depth - Bids)
  - `sell_quantity` (Order Book Depth - Asks)

**Dimensionen**: `12 ticks × 6 features = 72 values`

#### 2. Portfolio State
- `inventory_count`: Anzahl gehaltener Items (0 oder 1 bei buy-limit)
- `available_capital`: Aktuelles Guthaben in Gold
- `inventory_value`: Wert der gehaltenen Items (inventory_count × current_sell_price)
- `total_value`: available_capital + inventory_value
- `roi`: (total_value - initial_capital) / initial_capital

**Dimensionen**: `5 values`

#### 3. Order Book Features (Aggregiert)
- `avg_buy_qty_last_3`: Durchschnittliche Bid-Quantity letzte 3 Ticks
- `avg_sell_qty_last_3`: Durchschnittliche Ask-Quantity letzte 3 Ticks
- `liquidity_imbalance`: (avg_sell_qty - avg_buy_qty) / (avg_sell_qty + avg_buy_qty)
- `spread_percentage`: spread / mid_price

**Dimensionen**: `4 values`

#### 4. Technical Indicators
- `price_momentum_3`: (current_price - price_3_ticks_ago) / price_3_ticks_ago
- `price_momentum_6`: (current_price - price_6_ticks_ago) / price_6_ticks_ago
- `price_volatility`: Standardabweichung der letzten 12 Mid-Prices
- `rsi_14`: Relative Strength Index über 14 Ticks

**Dimensionen**: `4 values`

#### 5. Forecast Features (Optional)
- `forecast_next_3`: Preisprognose für nächste 3 Ticks (aus bestehendem XGBoost/ARIMA)
- `forecast_direction`: +1 (steigend), 0 (stabil), -1 (fallend)
- `forecast_confidence`: Modell-Konfidenz (z.B. aus Backtest-MAPE abgeleitet)

**Dimensionen**: `5 values (3 forecast values + direction + confidence)`

### State Vector Gesamtdimensionen

**Total**: 72 + 5 + 4 + 4 + 5 = **90 Features**

### Normalisierung

Alle Features werden normalisiert:
- Preise: Min-Max Normalisierung auf [0, 1] basierend auf historischen Min/Max
- Quantities: Log-Transformation + Standardisierung
- Prozentuale Werte: Clipping auf [-1, 1]
- Technische Indikatoren: Standardisierung (mean=0, std=1)

## Action Space

### Diskrete Aktionen

Der Agent hat bei jedem Tick 4 mögliche Aktionen:

```python
ACTION_NO_OP = 0      # Keine Aktion (Hold)
ACTION_BUY_1 = 1      # Kaufe 1 Item zum aktuellen Sell-Price
ACTION_SELL_1 = 2     # Verkaufe 1 Item zum aktuellen Buy-Price (wenn inventory >= 1)
ACTION_SELL_ALL = 3   # Verkaufe alle Items zum aktuellen Buy-Price (wenn inventory > 0)
```

### Aktions-Constraints

Die Actions werden durch Environment-Logik gefiltert:

- **BUY_1**: Nur möglich wenn:
  - `available_capital >= (current_sell_price + fees)`
  - `inventory_count < 1` (Buy-Limit)
  - Liquidität vorhanden (`sell_quantity > 0`)

- **SELL_1 / SELL_ALL**: Nur möglich wenn:
  - `inventory_count > 0`
  - Liquidität vorhanden (`buy_quantity > 0`)

Wenn eine ungültige Aktion gewählt wird:
- **Option A**: Ersetze durch NO_OP (keine Strafe)
- **Option B**: Ersetze durch NO_OP + kleine Straf-Reward (-0.01)

### GW2 Trading Post Gebühren

Bei jeder Transaktion fallen Gebühren an:
- **Listing Fee**: 5% des Verkaufspreises (beim Einstellen)
- **Exchange Fee**: 10% des Verkaufspreises (bei erfolgreichem Verkauf)
- **Total Sell Fee**: 15% des Verkaufspreises
- **Buy Fee**: Keine direkte Gebühr

**Implementation**:
```python
def execute_buy(self, quantity: int, price: float):
    cost = quantity * price
    self.capital -= cost
    self.inventory += quantity

def execute_sell(self, quantity: int, price: float):
    revenue_gross = quantity * price
    fee = revenue_gross * 0.15  # 15% GW2 fees
    revenue_net = revenue_gross - fee
    self.capital += revenue_net
    self.inventory -= quantity
```

## Reward Function

### Design-Philosophie

Ziel: Maximiere Profit unter Berücksichtigung von Risiko und Transaktionskosten.

### Reward-Komponenten

Die Reward-Function setzt sich aus mehreren Komponenten zusammen:

#### 1. Realized Profit Reward (Hauptkomponente)

```python
realized_profit_reward = (revenue_net - purchase_cost) / initial_capital
```

Wird nur beim Verkauf ausgegeben (event-basiert).

**Beispiel**:
- Kauf 1 Item @ 100 Gold
- Verkauf 1 Item @ 120 Gold → 120 * 0.85 = 102 Gold (nach Fees)
- Profit: 102 - 100 = 2 Gold
- Reward: 2 / 10000 = 0.0002

#### 2. Unrealized Profit/Loss (Mark-to-Market)

```python
# Bewerte Portfolio-Wert bei jedem Schritt
current_portfolio_value = capital + (inventory * current_sell_price * 0.85)
step_reward = (current_portfolio_value - previous_portfolio_value) / initial_capital
```

**Wichtig**: Berechne sell_price × 0.85, um Fees zu berücksichtigen.

#### 3. Transaction Cost Penalty

```python
transaction_penalty = -0.0001 * action_taken  # Kleine Strafe für jede Aktion außer NO_OP
```

Vermeidet übermäßiges Trading (Overtrading).

#### 4. Holding Time Bonus/Penalty (Optional)

```python
if inventory > 0:
    holding_time += 1
    if holding_time > max_holding_time:  # z.B. 24 Ticks = 2 Stunden
        holding_penalty = -0.0005 * (holding_time - max_holding_time)
```

Verhindert, dass Agent Items zu lange hält.

#### 5. Risk-Adjusted Reward (Sharpe-Like)

```python
# Berechne Volatilität der Returns
returns_std = np.std(recent_returns)
risk_adjusted_reward = step_reward / (returns_std + 1e-6)
```

Optional: Belohne Strategien mit besserer Risk/Reward-Ratio.

### Gesamte Reward Function (Basis-Variante)

```python
def compute_reward(self, action, realized_profit=0):
    # Basis: Unrealized P&L
    current_value = self.capital + (self.inventory * self.current_sell_price * 0.85)
    previous_value = self.previous_portfolio_value
    unrealized_reward = (current_value - previous_value) / self.initial_capital

    # Realized Profit (nur bei Verkauf)
    realized_reward = realized_profit / self.initial_capital if realized_profit > 0 else 0

    # Transaction Cost
    transaction_penalty = -0.0001 if action != ACTION_NO_OP else 0

    # Total Reward
    total_reward = unrealized_reward + realized_reward + transaction_penalty

    self.previous_portfolio_value = current_value
    return total_reward
```

### Alternative Reward Designs

#### Option A: Sparse Reward (Nur bei Episode-Ende)

```python
# Reward nur am Ende der Episode
if done:
    final_value = self.capital + (self.inventory * self.current_sell_price * 0.85)
    total_return = (final_value - self.initial_capital) / self.initial_capital
    reward = total_return * 100  # Skalierung für bessere Lernrate
else:
    reward = 0
```

**Vorteile**: Einfach, klarer Signal
**Nachteile**: Sparse Signal, langsames Lernen

#### Option B: Shaped Reward (Kombination)

```python
# Kombination aus Unrealized P&L + Bonuses
reward = unrealized_reward + realized_reward + transaction_penalty + holding_penalty
```

**Vorteile**: Dichteres Signal, schnelleres Lernen
**Nachteile**: Reward Engineering, Gefahr von Reward Hacking

### Empfehlung für Start

**Basis-Variante** mit Unrealized + Realized Rewards verwenden, später iterieren basierend auf Agent-Verhalten.

## Episode Termination

### Episode-Ende Bedingungen

Episode endet wenn:

1. **Time Limit**: Maximale Anzahl Steps erreicht (z.B. 288 Ticks = 1 Tag)
2. **Bankruptcy**: `capital <= 0` und `inventory = 0`
3. **Data Exhaustion**: Historische Daten zu Ende

### Truncation vs Termination

- **Termination** (done=True): Bankruptcy (ungewolltes Ende)
- **Truncation** (truncated=True): Time Limit erreicht (gewolltes Ende)

Gymnasium unterstützt beide Flags separat.

### Episode-Start

Jede Episode startet:
- An zufälligem Zeitpunkt in historischen Daten (für Varianz)
- Mit `initial_capital` Gold
- Mit `inventory = 0` Items
- Mit zufälligem Seed für Reproducibility

## Multi-Item Extension (Future)

Für paralleles Trading mehrerer Items:

### Approach A: Separate Environments

- Jedes Item hat eigenes Environment
- Separate RL-Policies pro Item
- Einfach, aber keine Cross-Item Strategien

### Approach B: Multi-Discrete Action Space

```python
self.action_space = spaces.MultiDiscrete([4] * num_items)
# Beispiel: [ACTION_NO_OP, ACTION_BUY_1, ACTION_NO_OP, ACTION_SELL_ALL]
#           Item 1: Hold,     Item 2: Buy,       Item 3: Hold,     Item 4: Sell All
```

- Komplexere Policy
- Ermöglicht Portfolio-Diversifikation
- State-Space wächst linear mit num_items

### Empfehlung

Start mit **Single-Item Environment**, später Extension zu Multi-Item wenn Single-Item funktioniert.

## Zusammenfassung

| Komponente | Dimensionen | Details |
|------------|-------------|---------|
| **State Space** | 90 Features | Price History (72) + Portfolio (5) + Order Book (4) + Indicators (4) + Forecast (5) |
| **Action Space** | 4 Discrete | No-Op, Buy 1, Sell 1, Sell All |
| **Reward** | Continuous | Unrealized P&L + Realized Profit - Transaction Costs |
| **Episode Length** | ~288 Steps | 1 Tag bei 5-Min Ticks |
| **Fees** | 15% Sell | 5% Listing + 10% Exchange (GW2 Standard) |
