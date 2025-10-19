from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
from jax import Array, vmap
from jax import numpy as jnp
from jax.nn import sigmoid
from jax.scipy.stats import norm

from cuthbert import filter, smoother
from cuthbert.gaussian import taylor
from cuthbertlib.types import LogConditionalDensity, LogDensity


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
    time: Array  # float with shape (,) at each time step
    time_prev: Array  # float with shape (,) at each time step
    team_indices: Array  # int with shape (2,) at each time step
    result: Array  # {0, 1, 2} with shape (,) at each time step


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
            + 1e-8,  # Add small nugget to avoid numerical issues when x = x_prev
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

        prob_array = jnp.array([prob_draw, prob_home_win, prob_away_win])
        return jnp.log(prob_array[model_inputs.result])

    return log_potential, state.mean


### Build model
football_filter = taylor.build_filter(
    get_init_log_density,
    get_dynamics_log_density,
    get_observation_func,
    ignore_nan_dims=True,
)

### Run the filter - this takes about 90 seconds on my laptop
filter_states = filter(football_filter, match_data)


### Ok who are the best teams right now?
mean = filter_states.mean[-1]
top_team_inds = jnp.argsort(mean)[-20:]
top_team_names = [teams_id_to_name_dict[int(i)] for i in top_team_inds]
top_team_means = mean[top_team_inds]
cov = filter_states.chol_cov[-1] @ filter_states.chol_cov[-1].T
top_team_stds = jnp.sqrt(jnp.diag(cov) ** 2)[top_team_inds]

plt.barh(top_team_names, top_team_means, xerr=top_team_stds, color="limegreen")
last_match_date = football_data["date"].max().strftime("%Y-%m-%d")
plt.xlabel(f"Skill Rating {last_match_date}")
plt.show()
plt.tight_layout()
plt.savefig("docs/assets/international_football_latest_skill_rating.png", dpi=300)


### Build and run the smoother (historical skill ratings) - this takes about 60 seconds on my laptop
football_smoother = taylor.build_smoother(
    get_dynamics_log_density,
)
smoother_states = smoother(football_smoother, filter_states, match_data)

### Ok how have these teams performed historically?
time_ind_start = -10000
top_teams_over_time_inds = jnp.argsort(mean)[-10:][::-1]
top_team_names_over_time = [
    teams_id_to_name_dict[int(i)] for i in top_teams_over_time_inds
]
match_dates_over_time = football_data["date"][time_ind_start:]
top_team_means_over_time = smoother_states.mean[
    time_ind_start:, top_teams_over_time_inds
]
all_covs_diag = vmap(lambda x: jnp.diag(x @ x.T))(
    smoother_states.chol_cov[time_ind_start:]
)
top_team_stds_over_time = jnp.sqrt(all_covs_diag[:, top_teams_over_time_inds])

interesting_dates = {
    "Spain 1\nNetherlands 0": "2010-07-11",
    "Germany 1\nArgentina 0": "2014-07-13",
    "France 4\nCroatia 2": "2018-07-15",
    "Argentina 3(pens)\nFrance 3": "2022-12-18",
}


plt.plot(
    match_dates_over_time,
    top_team_means_over_time[:],
    label=top_team_names_over_time,
    alpha=0.6,
)

for name, date in interesting_dates.items():
    date = pd.to_datetime(date)
    # Add name as little annotation at the date, vertical orientation
    ylim_top = plt.ylim()[1]
    plt.annotate(
        name,
        (date, ylim_top - 0.01),  # type: ignore
        rotation=90,
        fontsize=6,
        fontweight="bold",
        va="top",
        ha="right",
    )

plt.legend(top_team_names_over_time, loc="lower right", fontsize=9)
plt.ylabel("Skill Rating")
plt.show()
plt.tight_layout()
plt.savefig("docs/assets/international_football_historical_skill_rating.png", dpi=300)


### TODO: Can the figures be updated in the docs automatically on every-re-run?
