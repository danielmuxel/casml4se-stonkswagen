# Implementierungsplan

## Phasen-Übersicht

Die Implementation erfolgt in 5 Phasen mit jeweils klaren Deliverables und Validierung.

```
Phase 1: Foundation       (~ 2-3 Tage Dev-Time)
    ↓
Phase 2: Environment      (~ 3-4 Tage Dev-Time)
    ↓
Phase 3: Agent & Training (~ 2-3 Tage Dev-Time)
    ↓
Phase 4: Evaluation       (~ 2-3 Tage Dev-Time)
    ↓
Phase 5: Optimization     (~ 3-5 Tage Dev-Time)
```

---

## Phase 1: Foundation Setup

### Ziele
- Projekt-Dependencies installieren
- Basis-Struktur erstellen
- Unit Tests Setup

### Tasks

#### 1.1 Dependencies installieren

```bash
# Add to pyproject.toml
uv add gymnasium stable-baselines3 tensorboard

# Optional
uv add --group dev pytest-env pytest-mock
```

#### 1.2 Module-Struktur erstellen

```bash
mkdir -p src/gw2ml/rl/{environments,agents,features,rewards,training}
touch src/gw2ml/rl/__init__.py
touch src/gw2ml/rl/environments/__init__.py
touch src/gw2ml/rl/agents/__init__.py
touch src/gw2ml/rl/features/__init__.py
touch src/gw2ml/rl/rewards/__init__.py
touch src/gw2ml/rl/training/__init__.py
```

#### 1.3 Base Classes implementieren

**Files zu erstellen**:
- `src/gw2ml/rl/rewards/base_reward.py`
- `src/gw2ml/rl/agents/base_agent.py`

**base_reward.py**:
```python
from abc import ABC, abstractmethod

class BaseReward(ABC):
    """Abstract base class for reward functions"""

    @abstractmethod
    def reset(self, initial_capital: float):
        """Reset internal state for new episode"""
        pass

    @abstractmethod
    def compute(
        self,
        action: int,
        portfolio_state,
        current_sell_price: float,
        realized_profit: float = 0.0,
    ) -> float:
        """Compute reward for current step"""
        pass
```

**base_agent.py**:
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for RL agents"""

    @abstractmethod
    def train(self, total_timesteps: int, **kwargs):
        """Train the agent"""
        pass

    @abstractmethod
    def predict(self, observation, deterministic=True):
        """Predict action given observation"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent to disk"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load agent from disk"""
        pass
```

#### 1.4 Constants & Config

**File**: `src/gw2ml/rl/constants.py`

```python
# Action Space
ACTION_NO_OP = 0
ACTION_BUY_1 = 1
ACTION_SELL_1 = 2
ACTION_SELL_ALL = 3

ACTION_NAMES = {
    ACTION_NO_OP: "NO_OP",
    ACTION_BUY_1: "BUY_1",
    ACTION_SELL_1: "SELL_1",
    ACTION_SELL_ALL: "SELL_ALL",
}

# GW2 Trading Post Fees
GW2_LISTING_FEE = 0.05
GW2_EXCHANGE_FEE = 0.10
GW2_TOTAL_SELL_FEE = 0.15

# Default Hyperparameters
DEFAULT_LOOKBACK_WINDOW = 12
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_MAX_INVENTORY = 1
```

### Validierung Phase 1
- [ ] Dependencies installiert
- [ ] Module-Struktur erstellt
- [ ] Base Classes implementiert
- [ ] Constants definiert
- [ ] `pytest` läuft durch

---

## Phase 2: Environment Implementation

### Ziele
- PortfolioManager implementieren
- StateBuilder implementieren
- GW2TradingEnv implementieren
- Unit Tests für Environment

### Tasks

#### 2.1 PortfolioManager

**File**: `src/gw2ml/rl/environments/portfolio_manager.py`

Siehe Architektur-Dokument für vollständige Implementation.

**Key Features**:
- `PortfolioState` dataclass
- `Trade` dataclass
- `execute_buy()` / `execute_sell()` mit GW2 Fees
- `can_buy()` / `can_sell()` Constraints
- `get_summary()` Statistiken

#### 2.2 StateBuilder

**File**: `src/gw2ml/rl/features/state_builder.py`

**Step-by-Step**:

1. Implementiere `__init__()` mit Lookback-Window Parameter
2. Implementiere `fit()` für Feature Scalers
3. Implementiere `_extract_price_history()`
4. Implementiere `_extract_portfolio_features()`
5. Implementiere `_extract_orderbook_features()`
6. Implementiere `_extract_technical_indicators()`
7. Implementiere `build_state()` als Orchestrator

**Tests**:
```python
def test_state_builder_dimensions():
    builder = StateBuilder(lookback_window=12)
    state = builder.build_state(...)
    assert state.shape == (90,)  # Expected dimension

def test_state_normalization():
    builder = StateBuilder()
    builder.fit(historical_data)
    state = builder.build_state(...)
    assert np.all(state >= -5) and np.all(state <= 5)  # Reasonable bounds
```

#### 2.3 Technical Indicators

**File**: `src/gw2ml/rl/features/technical_indicators.py`

Implementiere Utility-Functions:

```python
def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute Relative Strength Index"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    return 100 - (100 / (1 + rs))

def compute_momentum(prices: np.ndarray, period: int) -> float:
    """Compute price momentum"""
    if len(prices) < period + 1:
        return 0.0
    return (prices[-1] - prices[-period-1]) / prices[-period-1]

def compute_volatility(prices: np.ndarray, window: int = 12) -> float:
    """Compute price volatility (std dev)"""
    if len(prices) < window:
        return 0.0
    return np.std(prices[-window:])
```

#### 2.4 Reward Functions

**File**: `src/gw2ml/rl/rewards/pnl_reward.py`

Implementiere `PnLReward` wie in Architektur-Dokument beschrieben.

**Tests**:
```python
def test_pnl_reward_profitable_trade():
    reward_fn = PnLReward()
    reward_fn.reset(initial_capital=10000)

    # Simulate buy
    portfolio.execute_buy(quantity=1, price=100)
    r1 = reward_fn.compute(ACTION_BUY_1, portfolio, current_sell_price=100)

    # Simulate profitable sell
    portfolio.execute_sell(quantity=1, price=120)
    r2 = reward_fn.compute(ACTION_SELL_1, portfolio, current_sell_price=120)

    assert r2 > 0  # Should be positive for profitable trade
```

#### 2.5 GW2TradingEnv

**File**: `src/gw2ml/rl/environments/gw2_trading_env.py`

**Implementation Steps**:

1. Implementiere `__init__()`
2. Implementiere `reset()` - Random episode start
3. Implementiere `_build_observation()` - Nutzt StateBuilder
4. Implementiere `_validate_action()` - Check Constraints
5. Implementiere `_execute_action()` - Nutzt PortfolioManager
6. Implementiere `_compute_reward()` - Nutzt RewardFunction
7. Implementiere `step()` - Orchestrator
8. Implementiere `_check_termination()` - Bankruptcy, Time Limit
9. Optional: Implementiere `render()` für Debugging

**Tests**:
```python
def test_env_initialization():
    env = GW2TradingEnv(item_id=19702, data=sample_data)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape

def test_env_invalid_action_handling():
    env = GW2TradingEnv(item_id=19702, data=sample_data)
    env.reset()

    # Try to sell without inventory
    obs, reward, terminated, truncated, info = env.step(ACTION_SELL_1)
    assert info['action_taken'] == ACTION_NO_OP  # Should be converted to NO_OP

def test_env_episode_termination():
    env = GW2TradingEnv(item_id=19702, data=sample_data, max_steps=10)
    env.reset()

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)

    assert truncated == True  # Time limit reached
```

### Validierung Phase 2
- [ ] PortfolioManager funktioniert (Unit Tests)
- [ ] StateBuilder baut korrekten State Vector
- [ ] GW2TradingEnv erfüllt Gymnasium Interface
- [ ] Environment kann min. 1 Episode durchlaufen
- [ ] Actions werden korrekt validiert und ausgeführt
- [ ] Rewards werden korrekt berechnet

---

## Phase 3: Agent & Training

### Ziele
- PPO Agent Wrapper implementieren
- Training Pipeline implementieren
- Erste Trainings durchführen
- TensorBoard Logging

### Tasks

#### 3.1 Random Agent (Baseline)

**File**: `src/gw2ml/rl/agents/random_agent.py`

```python
class RandomAgent(BaseAgent):
    """Random action baseline for comparison"""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        return self.action_space.sample(), None

    def train(self, total_timesteps: int, **kwargs):
        pass  # No training for random agent

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
```

#### 3.2 PPO Agent

**File**: `src/gw2ml/rl/agents/ppo_agent.py`

Siehe Architektur-Dokument für vollständige Implementation.

**Key Features**:
- Stable-Baselines3 PPO Wrapper
- Custom Training Callbacks (Checkpointing, Eval)
- Save/Load Functionality

#### 3.3 Training Config

**File**: `src/gw2ml/rl/training/hyperparameters.py`

```python
PPO_CONFIGS = {
    "default": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    "aggressive": {
        "learning_rate": 5e-4,
        "n_steps": 1024,
        "batch_size": 32,
        "gamma": 0.95,
    },
    "conservative": {
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "gamma": 0.99,
    }
}
```

#### 3.4 Training Pipeline

**File**: `src/gw2ml/pipelines/rl_pipeline.py`

Siehe Architektur-Dokument für vollständige Implementation.

**Entry Point**:
```python
def train_rl_agent(
    item_id: int,
    days_back: int = 90,
    train_timesteps: int = 100000,
    algorithm: str = "PPO",
    config_override: Optional[Dict] = None,
) -> Tuple[BaseAgent, Dict]:
    """High-level RL training pipeline"""
    ...
```

#### 3.5 Training Script

**File**: `scripts/train_rl.py`

```python
"""
Train RL Agent for GW2 Trading

Usage:
    python scripts/train_rl.py --item-id 19702 --timesteps 100000
    python scripts/train_rl.py --item-id 19721 --algorithm PPO --config aggressive
"""

import argparse
from src.gw2ml.pipelines.rl_pipeline import train_rl_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-id", type=int, required=True)
    parser.add_argument("--days-back", type=int, default=90)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--config", type=str, default="default")
    args = parser.parse_args()

    agent, results = train_rl_agent(
        item_id=args.item_id,
        days_back=args.days_back,
        train_timesteps=args.timesteps,
        algorithm=args.algorithm,
        config_name=args.config,
    )

    print(f"\n=== Training Complete ===")
    print(f"Final Return: {results['final_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Model saved to: models/rl/{args.item_id}/")

if __name__ == "__main__":
    main()
```

#### 3.6 Callbacks

**File**: `src/gw2ml/rl/training/callbacks.py`

```python
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """Custom callback for detailed TensorBoard logging"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log custom metrics
        if "episode" in self.locals["infos"][0]:
            info = self.locals["infos"][0]["episode"]
            self.logger.record("rollout/ep_reward", info["r"])
            self.logger.record("rollout/ep_length", info["l"])

        return True
```

### Validierung Phase 3
- [ ] Random Agent läuft als Baseline
- [ ] PPO Agent trainiert ohne Crashes
- [ ] Training speichert Checkpoints
- [ ] TensorBoard zeigt Metriken an
- [ ] Trained Agent kann geladen und weiterverwendet werden

**Erste Trainings-Runs**:
```bash
# Baseline: Random Agent
python scripts/train_rl.py --item-id 19702 --timesteps 10000 --algorithm Random

# PPO Training
python scripts/train_rl.py --item-id 19702 --timesteps 100000 --algorithm PPO

# TensorBoard
tensorboard --logdir runs/
```

---

## Phase 4: Evaluation & Backtesting

### Ziele
- RL Backtest Framework implementieren
- Metriken berechnen (Sharpe, Drawdown, etc.)
- Visualisierungen erstellen
- Vergleich mit Buy-and-Hold Baseline

### Tasks

#### 4.1 Metrics Computation

**File**: `src/gw2ml/evaluation/rl_metrics.py`

```python
def compute_returns(portfolio_values: pd.Series) -> pd.Series:
    """Compute period-over-period returns"""
    return portfolio_values.pct_change().fillna(0)

def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe Ratio"""
    if len(returns) < 2:
        return 0.0
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    return (mean_return - risk_free_rate) / std_return * np.sqrt(len(returns))

def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute Sortino Ratio (only downside deviation)"""
    if len(returns) < 2:
        return 0.0
    mean_return = returns.mean()
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    return (mean_return - risk_free_rate) / downside_std * np.sqrt(len(returns))

def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Compute Maximum Drawdown"""
    cumulative_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    return drawdown.min()

def compute_win_rate(trades: List[Trade]) -> float:
    """Compute percentage of profitable trades"""
    if len(trades) == 0:
        return 0.0

    # Match buys with sells
    profitable_trades = 0
    total_trades = 0

    buy_stack = []
    for trade in trades:
        if trade.action == "BUY":
            buy_stack.append(trade)
        elif trade.action == "SELL":
            if buy_stack:
                buy_trade = buy_stack.pop(0)
                # Calculate profit (including fees)
                revenue = trade.quantity * trade.price * (1 - 0.15)  # After fees
                cost = buy_trade.quantity * buy_trade.price
                if revenue > cost:
                    profitable_trades += 1
                total_trades += 1

    return profitable_trades / total_trades if total_trades > 0 else 0.0
```

#### 4.2 RL Backtest

**File**: `src/gw2ml/evaluation/rl_backtest.py`

Siehe Architektur-Dokument für vollständige Implementation.

**Key Functions**:
- `evaluate_rl_agent()` - Multi-Episode Evaluation
- `compute_rl_metrics()` - Aggregiere Metriken
- `compare_with_baseline()` - Vergleich mit Buy-and-Hold

#### 4.3 Visualizations

**File**: `src/gw2ml/evaluation/rl_plots.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_portfolio_value(
    portfolio_values: pd.Series,
    baseline_values: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
):
    """Plot portfolio value over time with optional baseline"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(portfolio_values.index, portfolio_values.values, label="RL Agent", linewidth=2)

    if baseline_values is not None:
        ax.plot(baseline_values.index, baseline_values.values,
                label="Buy & Hold", linestyle="--", linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio Value (Gold)")
    ax.set_title("Portfolio Value Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_action_distribution(trades: List[Trade], save_path: Optional[str] = None):
    """Plot distribution of actions taken"""
    actions = [trade.action for trade in trades]
    action_counts = pd.Series(actions).value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    action_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_drawdown(portfolio_values: pd.Series, save_path: Optional[str] = None):
    """Plot drawdown over time"""
    cumulative_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, color='red', linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.set_title("Portfolio Drawdown")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

#### 4.4 Evaluation Script

**File**: `scripts/evaluate_rl.py`

```python
"""
Evaluate trained RL Agent

Usage:
    python scripts/evaluate_rl.py --item-id 19702 --model-path models/rl/19702/best_model.zip
"""

import argparse
from src.gw2ml.pipelines.rl_pipeline import load_rl_agent
from src.gw2ml.evaluation.rl_backtest import evaluate_rl_agent, compare_with_baseline
from src.gw2ml.evaluation.rl_plots import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-id", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    args = parser.parse_args()

    # Load Agent
    agent = load_rl_agent(args.model_path)

    # Load Test Environment
    test_env = create_test_env(args.item_id)

    # Evaluate
    results = evaluate_rl_agent(
        agent=agent,
        env=test_env,
        num_episodes=args.num_episodes,
        deterministic=True
    )

    # Compare with Baseline
    baseline_results = compare_with_baseline(test_env)

    # Print Results
    print("\n=== Evaluation Results ===")
    print(f"Total Return: {results.total_return_pct:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"\nBaseline Return: {baseline_results.total_return_pct:.2%}")

    # Generate Plots
    plot_portfolio_value(results.portfolio_value_history,
                         baseline_results.portfolio_value_history,
                         save_path=f"plots/rl_{args.item_id}_portfolio.png")

    plot_action_distribution(results.trades,
                            save_path=f"plots/rl_{args.item_id}_actions.png")

    plot_drawdown(results.portfolio_value_history,
                  save_path=f"plots/rl_{args.item_id}_drawdown.png")

if __name__ == "__main__":
    main()
```

### Validierung Phase 4
- [ ] Evaluation läuft auf Test-Set
- [ ] Metriken werden korrekt berechnet
- [ ] Plots werden generiert
- [ ] Vergleich mit Baseline funktioniert
- [ ] Ergebnisse sind interpretierbar

---

## Phase 5: Optimization & Extensions

### Ziele
- Hyperparameter Tuning
- Alternative Algorithmen testen (DQN, A2C)
- Forecast-Integration
- Multi-Item Trading (optional)
- Production Deployment (optional)

### Tasks

#### 5.1 Hyperparameter Tuning

**File**: `scripts/tune_rl.py`

```python
"""
Hyperparameter Tuning mit Optuna

Usage:
    python scripts/tune_rl.py --item-id 19702 --n-trials 50
"""

import optuna
from optuna.pruners import MedianPruner
from src.gw2ml.pipelines.rl_pipeline import train_rl_agent

def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)

    config = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "ent_coef": ent_coef,
    }

    # Train agent
    agent, results = train_rl_agent(
        item_id=args.item_id,
        train_timesteps=50000,  # Shorter for tuning
        config_override=config
    )

    # Return metric to optimize (e.g., Sharpe Ratio)
    return results["sharpe_ratio"]

def main():
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=3600)

    print("\n=== Best Hyperparameters ===")
    print(study.best_params)
    print(f"Best Sharpe Ratio: {study.best_value:.2f}")

if __name__ == "__main__":
    main()
```

#### 5.2 DQN Agent

**File**: `src/gw2ml/rl/agents/dqn_agent.py`

```python
from stable_baselines3 import DQN

class DQNAgent(BaseAgent):
    """Deep Q-Network Agent Wrapper"""

    def __init__(
        self,
        env: gym.Env,
        policy: str = "MlpPolicy",
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        gamma: float = 0.99,
        **kwargs
    ):
        self.model = DQN(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            verbose=1,
            **kwargs
        )

    # ... rest similar to PPOAgent
```

#### 5.3 Forecast Integration

**File**: `src/gw2ml/rl/features/forecast_integration.py`

```python
from src.gw2ml.pipelines.forecast import forecast_item
from src.gw2ml.modeling.registry import get_model

class ForecastFeatureExtractor:
    """Integrate existing forecast models as RL features"""

    def __init__(self, item_id: int, forecast_horizon: int = 3):
        self.item_id = item_id
        self.forecast_horizon = forecast_horizon

        # Load trained forecast model
        self.forecast_model = self._load_forecast_model()

    def _load_forecast_model(self):
        """Load best trained model for this item"""
        model_path = f"models/{self.item_id}/best_model.pkl"
        # Load model using existing infrastructure
        return get_model("XGBoost").load(model_path)

    def get_forecast_features(self, current_data: pd.DataFrame) -> np.ndarray:
        """
        Generate forecast features for current state

        Returns:
            np.ndarray: [forecast_t+1, forecast_t+2, forecast_t+3, direction, confidence]
        """
        # Generate forecast using trained model
        forecast = self.forecast_model.predict(n=self.forecast_horizon)

        # Extract forecast values
        forecast_values = forecast.values().flatten()

        # Compute direction
        current_price = current_data.iloc[-1]['buy_unit_price']
        avg_forecast = forecast_values.mean()
        direction = 1 if avg_forecast > current_price else (-1 if avg_forecast < current_price else 0)

        # Compute confidence (simplified)
        confidence = 1.0 / (1.0 + np.std(forecast_values))

        return np.concatenate([forecast_values, [direction, confidence]])
```

**Update StateBuilder** um Forecasts zu inkludieren:

```python
# In StateBuilder.__init__
self.forecast_extractor = ForecastFeatureExtractor(item_id) if include_forecast else None

# In build_state()
if self.include_forecast and self.forecast_extractor:
    forecast_features = self.forecast_extractor.get_forecast_features(data.iloc[:current_idx+1])
    features.extend(forecast_features)
```

#### 5.4 Multi-Item Environment (Optional)

**File**: `src/gw2ml/rl/environments/multi_item_env.py`

```python
class MultiItemTradingEnv(gym.Env):
    """Environment for trading multiple items simultaneously"""

    def __init__(
        self,
        item_ids: List[int],
        data_dict: Dict[int, pd.DataFrame],
        initial_capital: float = 10000.0,
        max_inventory_per_item: int = 1,
    ):
        self.item_ids = item_ids
        self.num_items = len(item_ids)

        # Multi-discrete action space: one action per item
        self.action_space = spaces.MultiDiscrete([4] * self.num_items)

        # State: concatenated states for all items + portfolio
        single_item_state_dim = 90  # From Phase 2
        portfolio_state_dim = 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(single_item_state_dim * self.num_items + portfolio_state_dim,),
            dtype=np.float32
        )

        # Portfolio manager for multiple items
        self.portfolio = MultiItemPortfolioManager(initial_capital, item_ids)

    def step(self, actions: np.ndarray):
        """
        Execute actions for all items

        Args:
            actions: np.ndarray of shape (num_items,) with action per item
        """
        # Execute each action
        for item_idx, action in enumerate(actions):
            item_id = self.item_ids[item_idx]
            # ... execute action for this item

        # Compute aggregate reward across all items
        reward = self._compute_multi_item_reward()

        # Build next observation (all items)
        observation = self._build_observation()

        return observation, reward, terminated, truncated, info
```

#### 5.5 Streamlit Dashboard (Optional)

**File**: `apps/streamlit/rl_dashboard.py`

```python
"""
Streamlit Dashboard für RL Agent Monitoring

Usage:
    streamlit run apps/streamlit/rl_dashboard.py
"""

import streamlit as st
import pandas as pd
from src.gw2ml.pipelines.rl_pipeline import load_rl_agent
from src.gw2ml.evaluation.rl_backtest import evaluate_rl_agent

st.title("GW2 Trading RL Agent Dashboard")

# Sidebar
item_id = st.sidebar.selectbox("Select Item", [19702, 19721, ...])
model_path = st.sidebar.text_input("Model Path", f"models/rl/{item_id}/best_model.zip")

if st.sidebar.button("Load & Evaluate"):
    with st.spinner("Loading agent..."):
        agent = load_rl_agent(model_path)
        test_env = create_test_env(item_id)
        results = evaluate_rl_agent(agent, test_env, num_episodes=10)

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{results.total_return_pct:.2%}")
    col2.metric("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
    col3.metric("Win Rate", f"{results.win_rate:.2%}")

    # Plot Portfolio Value
    st.subheader("Portfolio Value Over Time")
    st.line_chart(results.portfolio_value_history)

    # Trade History Table
    st.subheader("Trade History")
    trades_df = pd.DataFrame([t.__dict__ for t in results.trades])
    st.dataframe(trades_df)

    # Action Distribution
    st.subheader("Action Distribution")
    action_counts = trades_df['action'].value_counts()
    st.bar_chart(action_counts)
```

### Validierung Phase 5
- [ ] Hyperparameter Tuning liefert bessere Results
- [ ] Alternative Algorithmen getestet (DQN, A2C)
- [ ] Forecast-Integration verbessert Performance
- [ ] Multi-Item Trading funktioniert (optional)
- [ ] Dashboard deployed (optional)

---

## Testing Strategy

### Unit Tests

```bash
tests/
├── test_rl_environment.py        # Environment Tests
├── test_portfolio_manager.py     # Portfolio Logic Tests
├── test_state_builder.py         # State Construction Tests
├── test_reward_functions.py      # Reward Computation Tests
└── test_agents.py                # Agent Wrapper Tests
```

**Beispiel Test**:
```python
# tests/test_portfolio_manager.py
import pytest
from src.gw2ml.rl.environments.portfolio_manager import PortfolioManager

def test_execute_buy():
    pm = PortfolioManager(initial_capital=10000, item_id=19702)
    pm.execute_buy(quantity=1, price=100, timestamp=pd.Timestamp.now())

    assert pm.state.capital == 9900
    assert pm.state.inventory == 1
    assert len(pm.state.trades) == 1

def test_execute_sell_with_fees():
    pm = PortfolioManager(initial_capital=10000, item_id=19702)
    pm.execute_buy(quantity=1, price=100, timestamp=pd.Timestamp.now())
    pm.execute_sell(quantity=1, price=120, timestamp=pd.Timestamp.now())

    # 120 * 0.85 = 102 (after 15% fees)
    # 9900 (after buy) + 102 = 10002
    assert pm.state.capital == 10002
    assert pm.state.inventory == 0
    assert pm.state.total_fees_paid == 18  # 120 * 0.15

def test_cannot_buy_without_capital():
    pm = PortfolioManager(initial_capital=10, item_id=19702)
    with pytest.raises(InsufficientCapitalError):
        pm.execute_buy(quantity=1, price=100, timestamp=pd.Timestamp.now())
```

### Integration Tests

```python
# tests/integration/test_training_pipeline.py
def test_full_training_pipeline():
    """Test end-to-end training pipeline"""
    agent, results = train_rl_agent(
        item_id=19702,
        days_back=30,  # Small dataset for test
        train_timesteps=1000,  # Quick training
        algorithm="PPO"
    )

    assert agent is not None
    assert "final_return" in results
    assert os.path.exists(f"models/rl/19702/best_model.zip")
```

---

## Monitoring & Debugging

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/rl_training_{item_id}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# In Training
logger.info(f"Episode {episode}: Return={episode_return:.2f}, Length={episode_length}")
```

### Debug Mode

```python
# Enable environment rendering for debugging
env = GW2TradingEnv(..., render_mode='human')

# Step through manually
obs, info = env.reset()
for _ in range(10):
    action = agent.predict(obs)[0]
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Print state
    print(f"Action: {ACTION_NAMES[action]}, Reward: {reward:.4f}")
```

---

## Checkliste Gesamte Implementation

### Phase 1: Foundation
- [ ] Dependencies installiert
- [ ] Module-Struktur erstellt
- [ ] Base Classes implementiert
- [ ] Constants definiert

### Phase 2: Environment
- [ ] PortfolioManager implementiert und getestet
- [ ] StateBuilder implementiert und getestet
- [ ] Technical Indicators implementiert
- [ ] Reward Functions implementiert
- [ ] GW2TradingEnv implementiert und getestet
- [ ] Environment kann Episode durchlaufen

### Phase 3: Training
- [ ] Random Agent als Baseline
- [ ] PPO Agent implementiert
- [ ] Training Pipeline implementiert
- [ ] Training Script funktioniert
- [ ] TensorBoard Logging funktioniert

### Phase 4: Evaluation
- [ ] RL Metrics implementiert
- [ ] Backtest Framework implementiert
- [ ] Visualisierungen implementiert
- [ ] Evaluation Script funktioniert
- [ ] Vergleich mit Baseline möglich

### Phase 5: Optimization
- [ ] Hyperparameter Tuning durchgeführt
- [ ] Alternative Algorithmen getestet
- [ ] Forecast-Integration funktioniert
- [ ] Multi-Item Extension (optional)
- [ ] Dashboard deployed (optional)

---

## Nächste Schritte

Nach Abschluss aller Phasen:

1. **Produktionsreife**:
   - Robustness Testing auf verschiedenen Items
   - Error Handling & Logging verbessern
   - CI/CD Pipeline aufsetzen

2. **Live Trading** (optional):
   - GW2 API Integration für echte Orders
   - Paper Trading Mode für Validation
   - Risk Management Layer

3. **Research Extensions**:
   - Model-based RL (World Models)
   - Multi-agent RL (Market Simulation)
   - Transfer Learning zwischen Items

4. **Publikation**:
   - Dokumentation vervollständigen
   - Benchmarks veröffentlichen
   - Blog Post / Paper schreiben
