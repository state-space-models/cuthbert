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
from jax import Array, random, numpy as jnp, tree
from jax.scipy.stats import norm
import numpy as np
import yfinance as yf

from cuthbert import filter
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import systematic


class ObservationData(NamedTuple):
    """Model inputs for stochastic volatility filtering."""

    time: Array  # Current observation time (days since origin)
    time_prev: Array  # Previous observation time (days since origin)
    log_return: Array  # Log return Y_t = log(p_t / p_{t-1})


def download_stock_data(
    ticker: str = "MKS.L", start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (default: "MKS.L" for Marks & Spencer)
        start_date: Start date in "YYYY-MM-DD" format
            (defaults to 3 years before end date)
        end_date: End date in "YYYY-MM-DD" format (defaults to today)

    Returns:
        DataFrame with columns: Date, Close, and computed log_returns
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    if start_date is None:
        # end_date - 3 years
        start_date = str(pd.Timestamp(end_date) - pd.Timedelta(days=365 * 3))[:10]

    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

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
data = download_stock_data(ticker="MKS.L", start_date="2020-01-01")

# Create model inputs
obs_data = create_model_inputs(data)

previous_data = tree.map(lambda x: x[:-1], obs_data)
new_data = tree.map(lambda x: x[-1], obs_data)

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


# Run the particle filter on previous data
key, previous_key = random.split(random.key(0))
previous_states = filter(pf, previous_data, key=key)
filter_state = tree.map(lambda x: x[-1], previous_states)

# Predict
predict_model_inputs = ObservationData(
    time=new_data.time,
    time_prev=new_data.time_prev,
    log_return=jnp.array(jnp.nan),
)
key, predict_key, predict_ys_key = random.split(key)
predict_state = pf.filter_combine(
    filter_state, pf.filter_prepare(predict_model_inputs, key=predict_key)
)
predict_ys = jnp.exp(predict_state.particles / 2) * random.normal(
    predict_ys_key, (n_particles,)
)


## TODO: plot predicted xs and ys distributions


key, filter_key = random.split(key)
new_filter_state = pf.filter_combine(
    filter_state, pf.filter_prepare(new_data, key=filter_key)
)

# TODO: plot new filter state distribution
