"""
Online Stochastic Volatility Filtering Example

This example demonstrates online filtering for a stochastic volatility model
applied to M&S stock price data. The model uses:

- State: Log-volatility X_t following an Ornstein-Uhlenbeck process
- Observation: Log-returns Y_t ~ N(0, exp(X_t))
- Continuous time dynamics to handle irregular observations
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import jax
from jax import Array, random, numpy as jnp
from jax.scipy.stats import norm
import yfinance as yf
import numpy as np

from cuthbert import filter
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import systematic


class ObservationData(NamedTuple):
    """Model inputs for stochastic volatility filtering."""

    time: Array  # Current observation time (days since origin)
    time_prev: Array  # Previous observation time (days since origin)
    log_return: Array  # Log return Y_t = log(p_t / p_{t-1})


def download_stock_data(
    ticker: str = "MKS.L", start_date: str = "2020-01-01", end_date: str | None = None
) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (default: "MKS.L" for Marks & Spencer)
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format (defaults to today)

    Returns:
        DataFrame with columns: Date, Close, and computed log_returns
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    if len(data) == 0:
        raise ValueError(f"No data found for ticker {ticker}")

    # Calculate log returns
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))

    # Remove first row (NaN log return)
    data = data.iloc[1:].reset_index()

    # Convert dates to days since first observation
    origin_date = data["Date"].iloc[0]
    data["days_since_origin"] = (data["Date"] - origin_date).dt.days

    return data


def create_model_inputs(
    data: pd.DataFrame,
) -> ObservationData:
    """
    Create model inputs from stock data.

    Args:
        data: DataFrame with columns: days_since_origin, log_return

    Returns:
        ObservationData
    """
    times = jnp.array(data["days_since_origin"].values)
    log_returns = jnp.array(data["log_return"].values)

    # Create previous times (first observation has no previous)
    times_prev = jnp.concatenate([jnp.array([0]), times[:-1]])

    obs_data = ObservationData(
        time=times,
        time_prev=times_prev,
        log_return=log_returns,
    )

    return obs_data


# Download stock price data
print("Downloading M&S stock price data...")
data = download_stock_data(ticker="MKS.L", start_date="2020-01-01")

print(f"Downloaded {len(data)} observations")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

# Create model inputs
obs_data = create_model_inputs(data)

# Model parameters
mu = 0.0  # Mean log-volatility
rho = 0.95  # Persistence (close to 1 for volatility clustering)
sigma = 0.3  # Volatility of volatility
init_mean = 0.0  # Initial mean
init_std = 1.0  # Initial std

# Particle filter parameters
n_particles = 1000
ess_threshold = 0.5

# Convert discrete AR parameter to continuous OU mean reversion speed
# For discrete: X_t = μ + ρ(X_{t-1} - μ) + U_t
# For continuous: dX_t = -θ(X_t - μ)dt + σ dW_t
# The relationship is: e^{-θ*Δt} = ρ, so θ = -log(ρ) for unit time step
theta = -jnp.log(rho)  # Mean reversion speed


def init_sample(key: Array, model_inputs: ObservationData) -> Array:
    """Sample initial log-volatility X_0."""
    return init_mean + init_std * random.normal(key, ())


def propagate_sample(key: Array, state: Array, model_inputs: ObservationData) -> Array:
    """
    Propagate log-volatility through OU process.

    For OU process: dX_t = -θ(X_t - μ)dt + σ dW_t
    Over time interval Δt, the solution is:
    X_{t+Δt} | X_t ~ N(μ + (X_t - μ)*exp(-θ*Δt), σ²*(1 - exp(-2*θ*Δt))/(2*θ))
    """
    dt = model_inputs.time - model_inputs.time_prev

    # Mean: mean reversion towards μ
    mean = mu + (state - mu) * jnp.exp(-theta * dt)

    # Variance: derived from OU process solution
    # X_{t+Δt} | X_t ~ N(μ + (X_t - μ)*exp(-θ*Δt), σ²*(1 - exp(-2*θ*Δt))/(2*θ))
    var = (sigma**2) * (1 - jnp.exp(-2 * theta * dt)) / (2 * theta)
    std = jnp.sqrt(var)
    return mean + std * random.normal(key, ())


def log_potential(
    state_prev: Array, state: Array, model_inputs: ObservationData
) -> Array:
    """
    Compute log observation likelihood.

    Observation model: Y_t | X_t ~ N(0, exp(X_t))
    If log_return is NaN, return 0 (uniform potential => filter just propagates)
    """
    # Check if observation is missing (NaN)
    is_missing = jnp.isnan(model_inputs.log_return)

    # Log likelihood: log p(Y_t | X_t) = log N(Y_t; 0, exp(X_t))
    log_vol = state
    vol = jnp.exp(0.5 * log_vol)  # exp(X_t/2) = sqrt(exp(X_t))
    log_pot = norm.logpdf(model_inputs.log_return, 0.0, vol)

    # Missing observation = uniform potential => filter just propagates
    return jnp.where(is_missing, 0.0, log_pot)


# Build particle filter
print("\nBuilding stochastic volatility filter...")
pf = particle_filter.build_filter(
    init_sample=init_sample,
    propagate_sample=propagate_sample,
    log_potential=log_potential,
    n_filter_particles=n_particles,
    resampling_fn=systematic.resampling,
    ess_threshold=ess_threshold,
)

# Use offline filtering for most of the data, then online prediction for last few points
print("Running offline filtering on historical data...")
key = random.PRNGKey(42)

# Split data: use most for offline filtering, keep last few for online prediction demo
n_online = 3  # Number of points to demonstrate online prediction
n_total = len(obs_data.time)
n_offline = n_total - n_online

# Offline filtering on historical data
obs_data_offline = ObservationData(
    time=obs_data.time[: n_offline + 1],
    time_prev=obs_data.time_prev[: n_offline + 1],
    log_return=obs_data.log_return[: n_offline + 1],
)

filter_states = filter(pf, obs_data_offline, key=key)
print(f"Filtered {n_offline} time steps offline")

# Get final state from offline filtering
final_state = jax.tree.map(lambda x: x[-1], filter_states)

# Now do online prediction and evaluation on the last few points
print(f"\nOnline prediction and evaluation on last {n_online} points:")
results = []
state = final_state

# Online loop: predict ahead, then update with actual observation
for i, t in enumerate(range(n_offline, n_total)):
    # Extract filtered volatility at current time
    particles = state.particles
    log_weights = state.log_weights
    weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
    filtered_log_vol = jnp.sum(weights * particles)
    filtered_vol = jnp.exp(0.5 * filtered_log_vol)

    # Predict one step ahead using filter API with NaN observation
    if t + 1 < n_total:
        pred_key, key = random.split(key, 2)
        pred_model_inputs = ObservationData(
            time=obs_data.time[t + 1],
            time_prev=obs_data.time[t],
            log_return=jnp.array(jnp.nan),  # NaN = prediction step
        )

        # Use filter to propagate (predict)
        prepare_pred = pf.filter_prepare(pred_model_inputs, key=pred_key)
        predicted_state = pf.filter_combine(state, prepare_pred)

        # Extract predicted volatility and VaR
        pred_particles = predicted_state.particles
        pred_log_weights = predicted_state.log_weights
        pred_weights = jnp.exp(pred_log_weights - jax.nn.logsumexp(pred_log_weights))
        predicted_log_vol = jnp.sum(pred_weights * pred_particles)
        predicted_vol = jnp.exp(0.5 * predicted_log_vol)

        # Compute VaR from predictive distribution
        pred_vols = jnp.exp(0.5 * pred_particles)
        n_samples = 500
        key1, key2 = random.split(pred_key, 2)
        indices = random.categorical(key1, pred_log_weights, shape=(n_samples,))
        resampled_vols = pred_vols[indices]
        pred_returns = random.normal(key2, (n_samples,)) * resampled_vols
        VaR_95 = -jnp.quantile(pred_returns, 0.05)

        actual_return = obs_data.log_return[t + 1]

        results.append(
            {
                "time": t,
                "date": data["Date"].iloc[t],
                "pred_date": data["Date"].iloc[t + 1],
                "filtered_vol": filtered_vol,
                "predicted_vol": predicted_vol,
                "VaR_95": VaR_95,
                "actual_return": actual_return,
                "violated": actual_return < -VaR_95,
            }
        )

        print(f"\nTime {t} ({data['Date'].iloc[t].strftime('%Y-%m-%d')}):")
        print(f"  Filtered volatility: {filtered_vol:.4f}")
        print(f"  Predicted volatility: {predicted_vol:.4f}")
        print(f"  VaR (95%): {VaR_95:.4f}")
        print(f"  Actual return: {actual_return:.4f}")
        print(f"  VaR violation: {actual_return < -VaR_95}")

    # Update filter with actual observation at time t
    model_inputs_t = ObservationData(
        time=obs_data.time[t],
        time_prev=obs_data.time_prev[t],
        log_return=obs_data.log_return[t],
    )
    prepare_state = pf.filter_prepare(model_inputs_t, key=key)
    state = pf.filter_combine(state, prepare_state)

print(f"\nFinal log-likelihood: {state.log_likelihood:.2f}")

# Simple, clear visualization
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Volatility predictions
ax1 = axes[0]
for i, r in enumerate(results):
    ax1.plot(
        [r["date"], r["pred_date"]],
        [r["filtered_vol"], r["predicted_vol"]],
        "o-",
        color=f"C{i}",
        linewidth=2,
        markersize=8,
        label=f"Time {r['time']}",
    )
    ax1.scatter(r["date"], r["filtered_vol"], s=100, color=f"C{i}", zorder=5)
    ax1.scatter(
        r["pred_date"], r["predicted_vol"], s=100, color=f"C{i}", marker="s", zorder=5
    )

ax1.set_ylabel("Volatility", fontsize=12)
ax1.set_title("Online Volatility Prediction", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: VaR and actual returns
ax2 = axes[1]
for i, r in enumerate(results):
    color = "red" if r["violated"] else "green"
    ax2.barh(
        i,
        r["VaR_95"],
        left=-r["VaR_95"],
        color=color,
        alpha=0.3,
        label="VaR band" if i == 0 else "",
    )
    ax2.scatter(
        r["actual_return"], i, s=150, color=color, marker="x", linewidth=3, zorder=5
    )
    ax2.text(
        r["actual_return"] + 0.005
        if r["actual_return"] > 0
        else r["actual_return"] - 0.005,
        i,
        f"{r['actual_return']:.3f}",
        va="center",
        ha="left" if r["actual_return"] > 0 else "right",
        fontsize=10,
        fontweight="bold",
    )

ax2.set_yticks(range(len(results)))
ax2.set_yticklabels([f"Time {r['time']}" for r in results])
ax2.set_xlabel("Log Return", fontsize=12)
ax2.set_title("Predicted VaR vs Actual Returns", fontsize=14, fontweight="bold")
ax2.axvline(0, color="black", linestyle="-", alpha=0.3)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(
    "docs/assets/online_stoch_vol_predictions.png", dpi=150, bbox_inches="tight"
)
print("\nVisualization saved to docs/assets/online_stoch_vol_predictions.png")
plt.close()
