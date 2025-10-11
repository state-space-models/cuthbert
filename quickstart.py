from typing import NamedTuple
from jax import random, jit, numpy as jnp, Array
from jax.scipy.stats import norm
from jax.nn import sigmoid
import pandas as pd
import matplotlib.pyplot as plt

from cuthbert import filter
from cuthbert.gaussian import taylor
from cuthbertlib.types import LogDensity, LogConditionalDensity


### Load data
def load_international_football_data(
    start_date: str = "1872-11-30",
    end_date: str | None = None,
    origin_date: str | None = None,
    min_matches: int = 300,
) -> tuple[pd.DataFrame, dict[int, str], dict[str, int]]:
    """
    Load international football match result data.

    Sourced with gratitude from the very handy:
    https://github.com/martj42/international_results

    Requires internet connection to read the data.

    Args:
        start_date: The start date of the data to load.
            Defaults to the apparent start of international football "1872-11-30".
            Required in "YYYY-MM-DD" format.
        end_date: The end date of the data to load. Defaults to today's date
            Required in "YYYY-MM-DD" format.
        origin_date: The date to use as the zero point the output timestamps. Defaults
            to start_date. Required in "YYYY-MM-DD" format.
        min_matches: The minimum number of matches a team must have to be included.
            Defaults to 300.

    Returns:
        A tuple of match times, match team indices,
            match results (0 for draw, 1 for home win, 2 for away win),
            teams id to name dictionary, and teams name to id dictionary.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    if origin_date is None:
        origin_date = start_date

    origin_timestamp = pd.to_datetime(origin_date)

    data_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    data_all = pd.read_csv(data_url)

    # Process time data into days since origin date
    data_all["date"] = pd.to_datetime(data_all["date"])
    data_all["timestamp_days"] = (data_all["date"] - origin_timestamp).dt.days
    data_all = data_all[
        (data_all["date"] >= start_date) & (data_all["date"] <= end_date)
    ]

    # Filter teams with fewer than min_matches
    home_counts: pd.Series = data_all["home_team"].value_counts()
    away_counts: pd.Series = data_all["away_team"].value_counts()
    total_counts = home_counts.add(away_counts, fill_value=0)
    valid_teams = set(total_counts[total_counts >= min_matches].index)
    data_all = data_all[
        data_all["home_team"].isin(list(valid_teams))
        & data_all["away_team"].isin(list(valid_teams))
    ]

    # Build team dictionaries and IDs
    teams_arr = sorted(valid_teams)
    teams_name_to_id_dict = {a: i for i, a in enumerate(teams_arr)}
    teams_id_to_name_dict = {i: a for i, a in enumerate(teams_arr)}
    data_all["home_team_id"] = data_all["home_team"].apply(
        lambda s: teams_name_to_id_dict[s]
    )
    data_all["away_team_id"] = data_all["away_team"].apply(
        lambda s: teams_name_to_id_dict[s]
    )

    return data_all, teams_id_to_name_dict, teams_name_to_id_dict


football_data, teams_id_to_name_dict, teams_name_to_id_dict = (
    load_international_football_data()
)

print(football_data.tail())
print("Num teams:", len(teams_id_to_name_dict))
print("Num matches:", len(football_data))


### Define model_inputs
# Extract data needed for filtering into JAX arrays
match_times = jnp.array(football_data["timestamp_days"])
match_team_indices = jnp.array(football_data[["home_team_id", "away_team_id"]])
home_goals = jnp.array(football_data["home_score"])
away_goals = jnp.array(football_data["away_score"])
match_results = jnp.where(
    home_goals > away_goals, 1, jnp.where(home_goals < away_goals, 2, 0)
)  # 0 for draw, 1 for home win, 2 for away win


# Model inputs
class MatchData(NamedTuple):
    time: Array  # float with shape (,)
    time_prev: Array  # float with shape (,)
    team_indices: Array  # int with shape (2,)
    result: Array  # {0, 1, 2} with shape (,)


match_times_prev = jnp.concatenate([jnp.array([-1]), match_times[:-1]])

# Load into NamedTuple
match_data = MatchData(match_times, match_times_prev, match_team_indices, match_results)


### Define (taylor) model
num_teams = len(teams_id_to_name_dict)
# Params from https://doi.org/10.1093/jrsssc/qlae035
init_sd = 0.5**0.5
tau = 0.05
epsilon = 0.3


def get_init_log_density(model_inputs: MatchData) -> tuple[LogDensity, Array]:
    def init_log_density(x):
        return norm.logpdf(x, 0, init_sd).sum()

    return init_log_density, jnp.zeros(num_teams)


def get_dynamics_log_density(
    state: taylor.LinearizedKalmanFilterState, model_inputs: MatchData
) -> tuple[LogConditionalDensity, Array, Array]:
    def dynamics_log_density(x_prev, x):
        return norm.logpdf(
            x,
            x_prev,
            jnp.sqrt((tau**2) * (model_inputs.time - model_inputs.time_prev))
            + 1e-8,  # Add small nugget to avoid numerical issues
        ).sum()

    return dynamics_log_density, jnp.zeros(num_teams), jnp.zeros(num_teams)


def get_observation_func(
    state: taylor.LinearizedKalmanFilterState, model_inputs: MatchData
) -> tuple[taylor.LogPotential, Array]:
    def log_potential(x):
        x_home = x[model_inputs.team_indices[0]]
        x_away = x[model_inputs.team_indices[1]]

        prob_home_win = sigmoid(x_home - x_away - epsilon)
        prob_away_win = 1 - sigmoid(x_home - x_away + epsilon)
        prob_draw = 1 - prob_home_win - prob_away_win

        return jnp.log(
            jnp.array([prob_draw, prob_home_win, prob_away_win])[model_inputs.result]
        )

    return log_potential, state.mean


### Build model
football_filter = taylor.build_filter(
    get_init_log_density, get_dynamics_log_density, get_observation_func
)

########### Little test/debugging
from jax import tree, hessian
from cuthbertlib.linearize import linearize_taylor

state = football_filter.init_prepare(tree.map(lambda x: x[0], match_data))
print(state.mean)
print(state.chol_cov @ state.chol_cov.T)
# Not right, cov should not be zeros on the diagonal not touched by team_indices, it should still be init_sd**2
model_inputs = tree.map(lambda x: x[0], match_data)
observation_log_potential, linearization_point = get_observation_func(
    state, model_inputs
)
d, chol_R = linearize_taylor(observation_log_potential, linearization_point)
print(d)
print(chol_R)

# These should be non-zero and the others should be nans ?
print(d[model_inputs.team_indices])
print(chol_R[model_inputs.team_indices, model_inputs.team_indices])


# Mini log pot
def log_potential(x):
    x_home = x[0]
    x_away = x[1]

    prob_home_win = sigmoid(x_home - x_away - epsilon)
    prob_away_win = 1 - sigmoid(x_home - x_away + epsilon)
    prob_draw = 1 - prob_home_win - prob_away_win

    return jnp.log(jnp.array([prob_draw, prob_home_win, prob_away_win])[0])


d_mini, chol_R_mini = linearize_taylor(log_potential, jnp.array([0.0, 0.0]))
prec_mini = -hessian(log_potential)(jnp.array([0.0, 0.0]))


## Things to fix/check
# 1. linearize_taylor/symmetric_inv_sqrt can handle sparse (prec) matrices
# 2. prec_mini/d_mini, chol_R_mini and the final state.chol_cov make sense
