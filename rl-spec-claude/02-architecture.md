# Technische Architektur

## System-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                     GW2 Trading RL System                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐            ┌────────▼────────┐
        │  Data Layer    │            │   RL Layer      │
        │  (Existing)    │            │   (New)         │
        └───────┬────────┘            └────────┬────────┘
                │                               │
    ┌───────────┼───────────┐       ┌──────────┼──────────┐
    │           │           │       │          │          │
┌───▼──┐   ┌───▼──┐   ┌───▼──┐ ┌──▼───┐  ┌───▼───┐  ┌──▼────┐
│ DB   │   │Loader│   │Models│ │ Env  │  │ Agent │  │Trainer│
│Query │   │      │   │      │ │      │  │       │  │       │
└──────┘   └──────┘   └──────┘ └──────┘  └───────┘  └───────┘
                                    │          │          │
                            ┌───────┴──────────┴──────────┘
                            │
                    ┌───────▼──────┐
                    │ Evaluation   │
                    │ & Backtesting│
                    └──────────────┘
```

## Modul-Struktur

### Neue Verzeichnisse

```
src/gw2ml/
├── rl/                          # Neuer RL-Bereich
│   ├── __init__.py
│   ├── environments/            # Gymnasium Environments
│   │   ├── __init__.py
│   │   ├── base_trading_env.py      # Basis Trading Environment
│   │   ├── gw2_trading_env.py       # GW2-spezifische Implementation
│   │   └── portfolio_manager.py     # Portfolio State Management
│   ├── agents/                  # RL Agent Implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract Agent Interface
│   │   ├── ppo_agent.py            # Stable-Baselines3 PPO Wrapper
│   │   ├── dqn_agent.py            # Stable-Baselines3 DQN Wrapper
│   │   └── random_agent.py         # Baseline für Evaluation
│   ├── features/                # RL-spezifische Feature Engineering
│   │   ├── __init__.py
│   │   ├── state_builder.py        # State Vector Construction
│   │   ├── technical_indicators.py  # RSI, Momentum, etc.
│   │   └── forecast_integration.py  # Nutzt bestehende Forecasts
│   ├── rewards/                 # Reward Function Implementations
│   │   ├── __init__.py
│   │   ├── base_reward.py          # Abstract Reward Interface
│   │   ├── pnl_reward.py           # P&L-basierte Rewards
│   │   └── risk_adjusted_reward.py  # Sharpe-like Rewards
│   └── training/                # Training Pipelines
│       ├── __init__.py
│       ├── train_agent.py          # Training Loop
│       ├── callbacks.py            # Training Callbacks (Logging, Checkpointing)
│       └── hyperparameters.py      # Hyperparameter Configs
├── evaluation/
│   └── rl_backtest.py          # Erweiterung des bestehenden Backtesting
└── pipelines/
    └── rl_pipeline.py          # High-Level RL Training Pipeline
```

### Integration mit bestehendem Code

```
Existing                        New RL Components
─────────                       ─────────────────
src/gw2ml/data/                 src/gw2ml/rl/environments/
  ├── loaders.py      ──────►      └── gw2_trading_env.py
  └── database_queries.py               (nutzt load_gw2_series())

src/gw2ml/modeling/             src/gw2ml/rl/features/
  ├── xgboost.py      ──────►      └── forecast_integration.py
  └── registry.py                       (lädt trainierte Modelle)

src/gw2ml/evaluation/           src/gw2ml/evaluation/
  └── backtest.py     ──────►      └── rl_backtest.py
                                        (erweitert BacktestResult)
```

## Komponenten-Details

### 1. GW2TradingEnv (Gymnasium Environment)

**File**: `src/gw2ml/rl/environments/gw2_trading_env.py`

```python
class GW2TradingEnv(gym.Env):
    """
    Gymnasium-kompatibles Environment für GW2 Trading
    """

    def __init__(
        self,
        item_id: int,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        lookback_window: int = 12,
        reward_function: BaseReward = None,
        forecast_model: Optional[BaseModel] = None,
        include_forecast: bool = False,
    ):
        """
        Args:
            item_id: GW2 Item ID
            data: DataFrame mit Columns [timestamp, buy_unit_price, sell_unit_price,
                                         buy_quantity, sell_quantity]
            initial_capital: Startkapital in Gold
            lookback_window: Anzahl historische Ticks im State
            reward_function: Reward Function Instanz
            forecast_model: Optional trainiertes Forecasting-Modell
            include_forecast: Ob Forecast-Features im State enthalten sein sollen
        """

    def reset(self, seed=None, options=None):
        """Start neue Episode an zufälligem Zeitpunkt"""
        # Initialize portfolio, state, step counter
        return observation, info

    def step(self, action):
        """Execute action und advance environment"""
        # 1. Validate action
        # 2. Execute trade (if valid)
        # 3. Update portfolio state
        # 4. Advance to next tick
        # 5. Compute reward
        # 6. Build next observation
        # 7. Check termination
        return observation, reward, terminated, truncated, info

    def _build_observation(self):
        """Konstruiere State Vector aus aktueller Situation"""
        # Nutzt StateBuilder aus rl/features/state_builder.py

    def _execute_action(self, action):
        """Führe Trading-Aktion aus"""
        # Nutzt PortfolioManager aus rl/environments/portfolio_manager.py

    def _compute_reward(self, action, trade_result):
        """Berechne Reward für aktuellen Step"""
        # Delegiert an self.reward_function

    def render(self, mode='human'):
        """Optional: Visualisiere aktuellen State"""
```

**Key Dependencies**:
- `src/gw2ml/data/loaders.py::load_gw2_series()` für Daten-Loading
- `src/gw2ml/rl/features/state_builder.py::StateBuilder` für State Construction
- `src/gw2ml/rl/environments/portfolio_manager.py::PortfolioManager` für Portfolio State
- `src/gw2ml/rl/rewards/base_reward.py::BaseReward` für Reward Computation

### 2. PortfolioManager

**File**: `src/gw2ml/rl/environments/portfolio_manager.py`

```python
@dataclass
class PortfolioState:
    """Snapshot des Portfolio-Zustands"""
    capital: float
    inventory: int
    item_id: int
    trades: List[Trade]  # Trade History
    total_fees_paid: float

    @property
    def total_value(self, current_sell_price: float) -> float:
        """Portfolio-Gesamtwert (mark-to-market)"""
        return self.capital + (self.inventory * current_sell_price * 0.85)

    @property
    def roi(self, initial_capital: float) -> float:
        """Return on Investment"""
        return (self.total_value - initial_capital) / initial_capital


@dataclass
class Trade:
    """Einzelner Trade Record"""
    timestamp: pd.Timestamp
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    fees: float
    capital_after: float
    inventory_after: int


class PortfolioManager:
    """Verwaltet Portfolio-State und Trade-Execution"""

    def __init__(self, initial_capital: float, item_id: int):
        self.state = PortfolioState(
            capital=initial_capital,
            inventory=0,
            item_id=item_id,
            trades=[],
            total_fees_paid=0.0
        )
        self.initial_capital = initial_capital

    def execute_buy(self, quantity: int, price: float, timestamp: pd.Timestamp):
        """Führe Kauforder aus"""
        cost = quantity * price
        if cost > self.state.capital:
            raise InsufficientCapitalError(...)
        if self.state.inventory + quantity > 1:  # Buy limit
            raise InventoryLimitError(...)

        self.state.capital -= cost
        self.state.inventory += quantity
        self.state.trades.append(Trade(
            timestamp=timestamp,
            action="BUY",
            quantity=quantity,
            price=price,
            fees=0.0,  # Keine Kauf-Gebühren bei GW2
            capital_after=self.state.capital,
            inventory_after=self.state.inventory
        ))

    def execute_sell(self, quantity: int, price: float, timestamp: pd.Timestamp):
        """Führe Verkaufsorder aus"""
        if quantity > self.state.inventory:
            raise InsufficientInventoryError(...)

        revenue_gross = quantity * price
        fees = revenue_gross * 0.15  # 15% GW2 Trading Post Fees
        revenue_net = revenue_gross - fees

        self.state.capital += revenue_net
        self.state.inventory -= quantity
        self.state.total_fees_paid += fees
        self.state.trades.append(Trade(
            timestamp=timestamp,
            action="SELL",
            quantity=quantity,
            price=price,
            fees=fees,
            capital_after=self.state.capital,
            inventory_after=self.state.inventory
        ))

    def can_buy(self, quantity: int, price: float) -> bool:
        """Check ob Buy-Order möglich ist"""
        return (self.state.capital >= quantity * price and
                self.state.inventory + quantity <= 1)

    def can_sell(self, quantity: int) -> bool:
        """Check ob Sell-Order möglich ist"""
        return self.state.inventory >= quantity

    def get_summary(self) -> Dict[str, Any]:
        """Portfolio-Statistiken"""
        return {
            "capital": self.state.capital,
            "inventory": self.state.inventory,
            "total_trades": len(self.state.trades),
            "total_fees_paid": self.state.total_fees_paid,
            "roi": self.state.roi(self.initial_capital),
        }
```

### 3. StateBuilder

**File**: `src/gw2ml/rl/features/state_builder.py`

```python
class StateBuilder:
    """Konstruiert State Vector für RL Agent"""

    def __init__(
        self,
        lookback_window: int = 12,
        include_forecast: bool = False,
        forecast_horizon: int = 3,
    ):
        self.lookback_window = lookback_window
        self.include_forecast = include_forecast
        self.forecast_horizon = forecast_horizon

        # Feature Scalers (fit on historical data)
        self.price_scaler = MinMaxScaler()
        self.qty_scaler = StandardScaler()

    def fit(self, data: pd.DataFrame):
        """Fit Scalers auf historischen Daten"""
        self.price_scaler.fit(data[['buy_unit_price', 'sell_unit_price']])
        self.qty_scaler.fit(np.log1p(data[['buy_quantity', 'sell_quantity']]))

    def build_state(
        self,
        current_idx: int,
        data: pd.DataFrame,
        portfolio_state: PortfolioState,
        forecast_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Konstruiere State Vector

        Returns:
            np.ndarray of shape (state_dim,)
        """
        features = []

        # 1. Price History Features (72 values)
        price_history = self._extract_price_history(current_idx, data)
        features.extend(price_history)

        # 2. Portfolio State (5 values)
        portfolio_features = self._extract_portfolio_features(
            portfolio_state, data.iloc[current_idx]
        )
        features.extend(portfolio_features)

        # 3. Order Book Features (4 values)
        orderbook_features = self._extract_orderbook_features(current_idx, data)
        features.extend(orderbook_features)

        # 4. Technical Indicators (4 values)
        technical_features = self._extract_technical_indicators(current_idx, data)
        features.extend(technical_features)

        # 5. Forecast Features (5 values, optional)
        if self.include_forecast and forecast_values is not None:
            forecast_features = self._extract_forecast_features(forecast_values)
            features.extend(forecast_features)
        elif self.include_forecast:
            features.extend([0.0] * 5)  # Placeholder

        return np.array(features, dtype=np.float32)

    def _extract_price_history(self, current_idx: int, data: pd.DataFrame) -> List[float]:
        """Extract lookback window price features"""
        start_idx = max(0, current_idx - self.lookback_window + 1)
        window = data.iloc[start_idx:current_idx+1]

        # Pad if insufficient history
        if len(window) < self.lookback_window:
            window = pd.concat([
                pd.DataFrame(np.repeat(window.iloc[[0]].values,
                                      self.lookback_window - len(window), axis=0),
                            columns=window.columns),
                window
            ])

        features = []
        for _, row in window.iterrows():
            buy_price = row['buy_unit_price']
            sell_price = row['sell_unit_price']
            spread = sell_price - buy_price
            mid_price = (buy_price + sell_price) / 2
            buy_qty = np.log1p(row['buy_quantity'])
            sell_qty = np.log1p(row['sell_quantity'])

            # Normalize
            buy_price_norm = self.price_scaler.transform([[buy_price, sell_price]])[0][0]
            sell_price_norm = self.price_scaler.transform([[buy_price, sell_price]])[0][1]
            spread_norm = (spread / mid_price) if mid_price > 0 else 0

            features.extend([
                buy_price_norm, sell_price_norm, spread_norm,
                mid_price / 10000,  # Scale by typical item price
                buy_qty / 10,  # Rough normalization
                sell_qty / 10
            ])

        return features

    def _extract_portfolio_features(
        self, portfolio: PortfolioState, current_tick: pd.Series
    ) -> List[float]:
        """Extract portfolio state features"""
        current_sell_price = current_tick['sell_unit_price']
        inventory_value = portfolio.inventory * current_sell_price * 0.85
        total_value = portfolio.capital + inventory_value
        roi = (total_value - portfolio.initial_capital) / portfolio.initial_capital

        return [
            float(portfolio.inventory),  # 0 or 1
            portfolio.capital / 10000,   # Normalize by typical capital
            inventory_value / 10000,
            total_value / 10000,
            roi  # Already normalized (-1 to inf)
        ]

    # ... weitere Helper-Methoden für OrderBook, Technicals, Forecasts
```

### 4. Reward Functions

**File**: `src/gw2ml/rl/rewards/pnl_reward.py`

```python
class PnLReward(BaseReward):
    """P&L-basierte Reward Function"""

    def __init__(
        self,
        transaction_penalty: float = 0.0001,
        holding_penalty_threshold: int = 24,  # 2 Stunden
        holding_penalty_rate: float = 0.0005,
    ):
        self.transaction_penalty = transaction_penalty
        self.holding_penalty_threshold = holding_penalty_threshold
        self.holding_penalty_rate = holding_penalty_rate

        # State tracking
        self.previous_portfolio_value = None
        self.holding_time = 0

    def reset(self, initial_capital: float):
        """Reset für neue Episode"""
        self.previous_portfolio_value = initial_capital
        self.holding_time = 0

    def compute(
        self,
        action: int,
        portfolio_state: PortfolioState,
        current_sell_price: float,
        realized_profit: float = 0.0,
    ) -> float:
        """
        Berechne Reward für aktuellen Step

        Args:
            action: Ausgeführte Aktion (0-3)
            portfolio_state: Aktueller Portfolio-State
            current_sell_price: Aktueller Verkaufspreis
            realized_profit: Realisierter Profit bei Verkauf (default: 0)

        Returns:
            float: Reward value
        """
        # 1. Unrealized P&L
        current_value = portfolio_state.total_value(current_sell_price)
        unrealized_reward = (current_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # 2. Realized Profit Reward
        realized_reward = realized_profit / self.previous_portfolio_value if realized_profit > 0 else 0

        # 3. Transaction Cost Penalty
        trans_penalty = -self.transaction_penalty if action != ACTION_NO_OP else 0

        # 4. Holding Penalty
        if portfolio_state.inventory > 0:
            self.holding_time += 1
            if self.holding_time > self.holding_penalty_threshold:
                holding_penalty = -self.holding_penalty_rate * (
                    self.holding_time - self.holding_penalty_threshold
                )
            else:
                holding_penalty = 0
        else:
            self.holding_time = 0
            holding_penalty = 0

        # Total Reward
        total_reward = unrealized_reward + realized_reward + trans_penalty + holding_penalty

        # Update state
        self.previous_portfolio_value = current_value

        return total_reward
```

### 5. RL Agent Wrapper

**File**: `src/gw2ml/rl/agents/ppo_agent.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

class PPOAgent(BaseAgent):
    """Stable-Baselines3 PPO Agent Wrapper"""

    def __init__(
        self,
        env: gym.Env,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        **kwargs
    ):
        self.model = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            **kwargs
        )

    def train(
        self,
        total_timesteps: int,
        checkpoint_dir: str = "models/rl_checkpoints",
        eval_env: Optional[gym.Env] = None,
    ):
        """Train agent"""
        callbacks = [
            CheckpointCallback(
                save_freq=10000,
                save_path=checkpoint_dir,
                name_prefix="ppo_agent"
            )
        ]

        if eval_env is not None:
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=f"{checkpoint_dir}/best_model",
                log_path=f"{checkpoint_dir}/eval_logs",
                eval_freq=5000,
                deterministic=True
            ))

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )

    def predict(self, observation, deterministic=True):
        """Predict action"""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save model"""
        self.model.save(path)

    def load(self, path: str):
        """Load model"""
        self.model = PPO.load(path)
```

### 6. Training Pipeline

**File**: `src/gw2ml/pipelines/rl_pipeline.py`

```python
def train_rl_agent(
    item_id: int,
    days_back: int = 90,
    train_timesteps: int = 100000,
    algorithm: str = "PPO",
    config_override: Optional[Dict] = None,
):
    """
    High-Level RL Training Pipeline

    Args:
        item_id: GW2 Item ID
        days_back: Anzahl Tage historische Daten
        train_timesteps: Anzahl Training Steps
        algorithm: "PPO" oder "DQN"
        config_override: Optional Config-Overrides

    Returns:
        Trained Agent + Evaluation Results
    """
    # 1. Load Data
    gw2_series = load_gw2_series(
        item_id=item_id,
        days_back=days_back,
        value_column="buy_unit_price",
        fill_missing_dates=True
    )

    # 2. Split Data (Train/Val/Test)
    train_data, val_data, test_data = gw2_series.split_days(
        train_days=60, val_days=15  # Rest is test
    )

    # 3. Create Environments
    train_env = GW2TradingEnv(
        item_id=item_id,
        data=train_data.pd_dataframe(),
        initial_capital=10000.0,
        reward_function=PnLReward()
    )

    val_env = GW2TradingEnv(
        item_id=item_id,
        data=val_data.pd_dataframe(),
        initial_capital=10000.0,
        reward_function=PnLReward()
    )

    # 4. Load Forecast Model (optional)
    forecast_model = load_trained_model(item_id)  # From existing pipeline

    # 5. Create Agent
    if algorithm == "PPO":
        agent = PPOAgent(train_env, **config_override or {})
    elif algorithm == "DQN":
        agent = DQNAgent(train_env, **config_override or {})
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 6. Train
    agent.train(
        total_timesteps=train_timesteps,
        eval_env=val_env,
        checkpoint_dir=f"models/rl/{item_id}"
    )

    # 7. Evaluate on Test Set
    test_results = evaluate_rl_agent(
        agent=agent,
        env=test_env,
        num_episodes=10
    )

    # 8. Save Results
    save_rl_results(item_id, agent, test_results)

    return agent, test_results
```

## Abhängigkeiten

### Neue Dependencies (zu `pyproject.toml` hinzufügen)

```toml
[project.dependencies]
# Existing...
gymnasium = "^0.29.1"  # Nachfolger von gym
stable-baselines3 = "^2.3.0"  # RL Algorithms
sb3-contrib = "^2.3.0"  # Additional algorithms (TQC, etc.)
tensorboard = "^2.17.0"  # Training Monitoring
```

### Optional Dependencies

```toml
[project.optional-dependencies]
rl = [
    "ray[rllib]>=2.40.0",  # Falls RLlib statt SB3 gewünscht
    "wandb>=0.15.0",  # Experiment Tracking
]
```

## Konfiguration

### RL Config File

**File**: `src/gw2ml/pipelines/rl_config.py`

```python
DEFAULT_RL_CONFIG = {
    "environment": {
        "initial_capital": 10000.0,
        "lookback_window": 12,
        "include_forecast": False,
        "reward_function": "PnLReward",
        "reward_params": {
            "transaction_penalty": 0.0001,
            "holding_penalty_threshold": 24,
            "holding_penalty_rate": 0.0005,
        }
    },
    "agent": {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
    },
    "training": {
        "total_timesteps": 100000,
        "checkpoint_freq": 10000,
        "eval_freq": 5000,
    },
    "data": {
        "days_back": 90,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        # test_ratio = 0.15 (implizit)
    }
}
```

## Evaluation & Backtesting

### RL Backtest Extension

**File**: `src/gw2ml/evaluation/rl_backtest.py`

```python
@dataclass
class RLBacktestResult:
    """Ergebnis eines RL Backtests"""
    agent_name: str
    item_id: int
    item_name: str

    # Portfolio Metrics
    initial_capital: float
    final_capital: float
    final_inventory: int
    final_total_value: float
    total_return: float  # ROI
    total_return_pct: float

    # Trading Metrics
    num_trades: int
    num_buys: int
    num_sells: int
    total_fees_paid: float
    win_rate: float  # % profitable trades

    # Risk Metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float

    # Episode Stats
    num_episodes: int
    avg_episode_length: float
    avg_episode_return: float

    # Trade History
    trades: List[Trade]
    portfolio_value_history: pd.Series  # Time series of portfolio value


def evaluate_rl_agent(
    agent: BaseAgent,
    env: GW2TradingEnv,
    num_episodes: int = 10,
    deterministic: bool = True,
) -> RLBacktestResult:
    """
    Evaluiere RL Agent auf Test Environment

    Args:
        agent: Trainierter RL Agent
        env: Test Environment
        num_episodes: Anzahl Test-Episoden
        deterministic: Deterministic policy (True für Evaluation)

    Returns:
        RLBacktestResult mit Metriken
    """
    episode_returns = []
    episode_lengths = []
    all_trades = []
    portfolio_values = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

            # Track portfolio value
            portfolio_values.append({
                'timestamp': info['timestamp'],
                'value': info['portfolio_value']
            })

        episode_returns.append(episode_return)
        episode_lengths.append(info['step'])
        all_trades.extend(info['trades'])

    # Compute Metrics
    portfolio_df = pd.DataFrame(portfolio_values)
    metrics = compute_rl_metrics(portfolio_df, all_trades, env.initial_capital)

    return RLBacktestResult(
        agent_name=agent.__class__.__name__,
        item_id=env.item_id,
        item_name=env.item_name,
        num_episodes=num_episodes,
        avg_episode_return=np.mean(episode_returns),
        avg_episode_length=np.mean(episode_lengths),
        **metrics
    )
```

## Integration mit Existing Pipeline

### Unified Training Script

**File**: `scripts/train_unified.py`

```python
"""
Unified Training Script für Forecasting + RL

Usage:
    # Train nur Forecasting-Modelle
    python scripts/train_unified.py --mode forecast --items 19702 19721

    # Train nur RL Agent
    python scripts/train_unified.py --mode rl --items 19702 --algorithm PPO

    # Train beides
    python scripts/train_unified.py --mode both --items 19702
"""

def main(args):
    if args.mode in ["forecast", "both"]:
        # Nutzt bestehendes train_items()
        train_items(args.items, config_override=args.forecast_config)

    if args.mode in ["rl", "both"]:
        for item_id in args.items:
            train_rl_agent(
                item_id=item_id,
                algorithm=args.algorithm,
                config_override=args.rl_config
            )
```

## Monitoring & Logging

### TensorBoard Integration

```python
# In Training Loop
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"runs/rl_{item_id}")

# Log Metrics
writer.add_scalar("train/episode_return", episode_return, episode)
writer.add_scalar("train/portfolio_value", final_value, episode)
writer.add_scalar("eval/sharpe_ratio", sharpe, eval_step)
```

### MLflow Integration (optional)

```python
import mlflow

with mlflow.start_run(run_name=f"rl_{item_id}_{algorithm}"):
    mlflow.log_params({
        "item_id": item_id,
        "algorithm": algorithm,
        "learning_rate": lr,
        ...
    })

    # Train...

    mlflow.log_metrics({
        "final_return": final_return,
        "sharpe_ratio": sharpe,
        ...
    })

    mlflow.log_artifact(f"models/rl/{item_id}")
```

## Zusammenfassung

Die Architektur integriert sich nahtlos in die bestehende Codebase:

1. **Modular**: Neue `rl/` Module unabhängig von bestehendem Code
2. **Wiederverwendbar**: Nutzt `data/loaders.py`, `modeling/`, `evaluation/backtest.py`
3. **Erweiterbar**: Leicht neue Agents, Rewards, Features hinzufügen
4. **Evaluierbar**: Einheitliches Backtest-Framework für Forecasts und RL
5. **Wartbar**: Klare Trennung von Concerns, Type Hints, Docstrings
