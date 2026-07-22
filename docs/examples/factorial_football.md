# Modelling international football with `cuthbert.factorial`

We'll walk through an example of ranking international football teams over
time using a factorial Kalman filter and a (probabilistic) Elo-style model.

## Imports

```{.python #factorial-football-imports}
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from jax import Array, tree
from jax import numpy as jnp
from jax.nn import sigmoid
from jax.scipy.stats import norm

from cuthbert import factorial
from cuthbert.gaussian import taylor
from cuthbertlib.types import LogConditionalDensity, LogDensity
```

Nothing too surprising there I hope. We'll be using the [`taylor`](api_cuthbert/gaussian/taylor.md)
module which will let us generate Gaussian approximations to the filtering and smoothing
distributions whilst handling the discrete nature of the observations.


## Load data

We're going to need historical data from international football matches including
the dates of the matches, which teams played, and the result (draw, home win, away win).
Luckily, there's a very handy dataset of international football match results available on GitHub:
[github.com/martj42/international_results](https://github.com/martj42/international_results),
thanks Mart!.

Expand the code block below to see the data loading code (or just trust me on it).

??? quote "Code to download international football data into a `pandas` DataFrame"
    ```{.python #factorial-football-load-data}
    def load_international_football_data(
        start_date: str = "1872-11-30",
        end_date: str | None = None,
        origin_date: str | None = None,
        min_matches: int = 0,
    ) -> tuple[pd.DataFrame, dict[int, str], dict[str, int]]:
        """Load international football match result data.

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

        # Timestamp of the previous match for home and away team in each match
        # Extract previous timestamps for home and away teams
        num_matches = len(data_all)
        match_positions = np.arange(num_matches)
        timestamps = data_all["timestamp_days"].to_numpy()
        team_ids = np.concatenate(
            [
                data_all["home_team_id"].to_numpy(),
                data_all["away_team_id"].to_numpy(),
            ]
        )
        match_positions_by_team = np.concatenate([match_positions, match_positions])
        timestamps_by_team = np.concatenate([timestamps, timestamps])
        is_home_team = np.concatenate(
            [np.ones(num_matches, dtype=bool), np.zeros(num_matches, dtype=bool)]
        )
        order = np.lexsort((match_positions_by_team, timestamps_by_team, team_ids))
        previous_timestamps = np.zeros(2 * num_matches, dtype=timestamps.dtype)
        same_team_as_previous = team_ids[order][1:] == team_ids[order][:-1]
        previous_timestamps[order[1:]] = np.where(
            same_team_as_previous,
            timestamps_by_team[order[:-1]],
            0,
        )
        data_all["home_timestamp_previous"] = previous_timestamps[is_home_team]
        data_all["away_timestamp_previous"] = previous_timestamps[~is_home_team]


        return data_all, teams_id_to_name_dict, teams_name_to_id_dict
    ```

We'll now load the data and convert it into JAX arrays - the format expected by
`cuthbert` (we'll filter out very old matches).

```{.python #factorial-football-load-data-jax}
football_data, teams_id_to_name_dict, teams_name_to_id_dict = (
    load_international_football_data(start_date="1990-01-01", min_matches=300)
)

print(football_data.tail())
print("Num teams:", len(teams_id_to_name_dict))
print("Num matches:", len(football_data))

# Extract data needed for filtering into JAX arrays
match_times = jnp.array(football_data["timestamp_days"])
match_team_indices = jnp.array(football_data[["home_team_id", "away_team_id"]])
home_goals = jnp.array(football_data["home_score"])
away_goals = jnp.array(football_data["away_score"])
home_times_prev = jnp.array(football_data["home_timestamp_previous"])
away_times_prev = jnp.array(football_data["away_timestamp_previous"])
match_results = jnp.where(
    home_goals > away_goals, 1, jnp.where(home_goals < away_goals, 2, 0)
)  # 0 for draw, 1 for home win, 2 for away win
```

`cuthbert` convention is to not include an observation at the initial time step.
So we add dummy values to the start of the data
```{.python #factorial-football-no-initial-obs}
match_times = jnp.concatenate([jnp.array([0]), match_times])
match_team_indices = jnp.concatenate([jnp.array([[-1, -1]]), match_team_indices])
home_goals = jnp.concatenate([jnp.array([-1]), home_goals])
away_goals = jnp.concatenate([jnp.array([-1]), away_goals])
home_times_prev = jnp.concatenate([jnp.array([-1]), home_times_prev])
away_times_prev = jnp.concatenate([jnp.array([-1]), away_times_prev])
match_results = jnp.concatenate([jnp.array([-1]), match_results])
```


I said `cuthbert` expects JAX arrays, but more specifically and more generally,
it expects [`pytrees`](https://docs.jax.dev/en/latest/working-with-pytrees.html) with
`jax.Array` leaves (we call this an `ArrayTree`). Basically this allows us to
use clearer Python structures as long as the underlying data is a JAX array.

Here we'll use a [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)
to store all the information we'll need at each filtering step. Note that this includes
the time of the current match but also the time of the previous match.


```{.python #factorial-football-model-inputs}
# Model inputs
class MatchData(NamedTuple):
    time: Array  # float with shape (,) at each time step
    home_time_prev: Array  # float with shape (,) at each time step
    away_time_prev: Array  # float with shape (,) at each time step
    team_indices: Array  # int with shape (2,) at each time step
    result: Array  # {0, 1, 2} with shape (,) at each time step for {draw, home win, away win}

# Load into NamedTuple
match_data = MatchData(
    match_times, home_times_prev, away_times_prev, match_team_indices, match_results
)
```


## Define the state-space model

Now that we've got the data in a format we like, we can define the state-space model.

We'll use the model from [Duffield et al](https://doi.org/10.1093/jrsssc/qlae035)
which is an Elo-style probabilistic state-space model for temporal result data.

$$
\begin{aligned}
p(x_0^i) &= \mathcal{N}(x_0^i \mid 0, \sigma_0^2) \\
p(x_t^i | x_{t-1}^i) &= \mathcal{N}(x_t \mid x_{t-1}, \tau^2 (t - t^i_{prev})) \\
p(y_t | x_t^h, x_t^a) &=
\begin{cases}
\sigma(x_t^{h} - x_t^{a} + \epsilon) - \sigma(x_t^{h} - x_t^{a} - \epsilon) & y_t = \text{draw}, \\
\sigma(x_t^{h} - x_t^{a} - \epsilon) & y_t = h, \\
\sigma(x_t^{h} - x_t^{a} + \epsilon) & y_t = a,
\end{cases}
\end{aligned}
$$

where $\sigma(x) = (1 + \exp(-x))^{-1}$ is the sigmoid function and $h, a$ denote the
home and away team indices (although this simple model doesn't have a notion of home
advantage and many matches are played at neutral venues).

Here we'll just fix the static hyperparameters $(\sigma_0, \tau, \epsilon)$ to the values
from the paper (although these could also be learnt from the data - see [next steps](#next-steps)).


```{.python #factorial-football-state-space-model}

num_teams = len(teams_id_to_name_dict)

# Params from https://doi.org/10.1093/jrsssc/qlae035
init_sd = 0.5**0.5
tau = 0.05
epsilon = 0.3


def get_init_log_density(model_inputs: MatchData) -> tuple[LogDensity, Array]:
    def init_log_density(x):
        return norm.logpdf(x, 0, init_sd).sum()

    return init_log_density, jnp.zeros((num_teams, 1))


def get_dynamics_log_density(
    state: taylor.LinearizedKalmanFilterState, model_inputs: MatchData
) -> tuple[LogConditionalDensity, Array, Array]:

    def dynamics_log_density(x_prev, x):
        timestamps_prev = jnp.array(
            [model_inputs.home_time_prev, model_inputs.away_time_prev]
        )
        time_diff = model_inputs.time - timestamps_prev
        time_diff = jnp.where(
            time_diff < 1e-3, 1e-3, time_diff
        )  # Ensure non-negative time differences
        return norm.logpdf(x, x_prev, jnp.sqrt((tau**2) * time_diff)).sum()

    return dynamics_log_density, jnp.zeros(2), jnp.zeros(2)


def get_observation_func(
    state: taylor.LinearizedKalmanFilterState, model_inputs: MatchData
) -> tuple[taylor.LogPotential, Array]:
    def log_potential(x):
        x_home = x[0]
        x_away = x[1]

        prob_home_win = sigmoid(x_home - x_away - epsilon)
        prob_away_win = 1 - sigmoid(x_home - x_away + epsilon)
        prob_draw = 1 - prob_home_win - prob_away_win

        prob_array = jnp.array([prob_draw, prob_home_win, prob_away_win])
        return jnp.log(prob_array[model_inputs.result])

    return log_potential, state.mean
```

So what have we done here? We've defined the initial distribution, the dynamics, and the observation model
by simply writing their log densities as JAX functions.

Since the `taylor` method uses automatic differentiation to convert these into
conditional Gaussian parameters, we also needed to specify the linearization point to
use (the initial and dynamics distributions are Gaussian so we can actually use any
linearization point we like and `taylor` will exactly recover the Gaussian parameters,
the observation model is non-Gaussian so we tell `cuthbert` to linearize around the
current mean). The linearization point is specified in the additional output of the
`get_` functions - see the [`taylor` documentation](api_cuthbert/gaussian/taylor.md)
for more details.


## Build the filter

Now that we've defined the model, we can construct the `cuthbert` [filter object][cuthbert.inference.Filter].

```{.python #factorial-football-build-filter}
football_filter = taylor.build_filter(
    get_init_log_density,
    get_dynamics_log_density,
    get_observation_func,
)
```

Because this is a factorial model, we'll also need to build a `factorializer` to
extract the relevant factors (teams) for matches they are involved in.

```{.python #factorial-football-build-factorializer}
factorializer = factorial.gaussian.build_factorializer(
    get_factorial_indices=lambda model_inputs: model_inputs.team_indices
)
```



## Run the filter

We'll use [`cuthbert.factorial.filter`][cuthbert.factorial.filtering.filter] to easily run offline filtering on our data.

```{.python #factorial-football-run-filter}
init_match_data = tree.map(lambda x: x[0], match_data)
filter_match_data = tree.map(lambda x: x[1:], match_data)
init_state = football_filter.init_prepare(init_match_data)
init_state = factorializer.factorialize_init_state(init_state, init_match_data)
local_filter_states, final_factorial_state = factorial.filter(
    football_filter, factorializer, filter_match_data, init_state
)
```

Filtering done! So what have we got?
`local_filter_states` is an ArrayTree containing
the mean and variance of the skill of the two teams involved at each time step
(`local_filter_states.mean.shape = (num_time_steps, 2)`).
`final_factorial_state` is an ArrayTree containing the mean and variance of the skill of
all teams at their most recent match timestamp (`final_factorial_state.mean.shape = (num_teams,)`).

??? "Online filtering"
    `cuthbert.factorial.filter` assumes that all data is passed at once. If you are in an
    online setting where you want to filter as you go, you can use
    ```python
    # Filter next time point as new data arrives
    local_state = factorializer.extract_and_join(factorial_state, match_data)
    local_filter_state = football_filter.filter_combine(
        local_state, football_filter.filter_prepare(match_data)
    )
    factorial_state = factorializer.marginalize_and_insert(
        local_filter_state, factorial_state, match_data
    )
    ```

## Synchronize the factorial state

We've run offline filtering. But one of the quirks with factorial models is that the factorial state encodes the filtering distributions of all teams only
at their most recent match. If we want to update them all to be at the current time we have to run a synchronization step. In `cuthbert` we do this by running a separate filter across factors.


```{.python #factorial-football-sync}
# Model inputs
class DynamicsOnlyData(NamedTuple):
    current_time: Array  # float with shape (,) at each time step
    time_prev: Array  # float with shape (,) at each time step
    team_index: Array  # int with shape (,) at each time step


timestamps = jnp.array(football_data["timestamp_days"].to_numpy())
most_recent_timestamp_by_team = jnp.zeros(num_teams)
most_recent_timestamp_by_team = most_recent_timestamp_by_team.at[
    jnp.array(football_data["home_team_id"].to_numpy())
].max(timestamps)
most_recent_timestamp_by_team = most_recent_timestamp_by_team.at[
    jnp.array(football_data["away_team_id"].to_numpy())
].max(timestamps)

# Load into NamedTuple
sync_data = DynamicsOnlyData(
    current_time=jnp.broadcast_to(timestamps.max(), (num_teams,)),
    time_prev=most_recent_timestamp_by_team,
    team_index=jnp.arange(num_teams), 
)


def get_dynamics_log_density_sync(
    state: taylor.LinearizedKalmanFilterState, model_inputs: DynamicsOnlyData
) -> tuple[LogConditionalDensity, Array, Array]:
    time_diff = model_inputs.current_time - model_inputs.time_prev

    def dynamics_log_density(x_prev, x):
        return norm.logpdf(x, x_prev, jnp.sqrt((tau**2) * time_diff)).sum()
    
    lin_point = jnp.where(time_diff < 0.5, jnp.array([jnp.nan]), jnp.zeros(1))
    return dynamics_log_density, lin_point, lin_point


single_team_filter = taylor.build_filter(
    get_init_log_density,
    get_dynamics_log_density_sync,
    get_observation_func=lambda state, model_inputs: (
        lambda x: jnp.zeros(1),  # No observations
        jnp.zeros(1),
    ),
)

sync_factorial_state = factorial.synchronize(
    single_team_filter, factorializer, sync_data, final_factorial_state
)
```

## Ok so who are the best teams right now?

Now that we've filtered the data, we can extract the mean and covariance of the
filtered distribution which we can get from `filter_states.mean` and
`filter_states.chol_cov`.


??? quote "Code to extract and plot the latest filtered distribution"
    ```{.python #factorial-football-extract-filtered-distribution}
    mean = sync_factorial_state.mean[..., 0]
    top_team_inds = jnp.argsort(mean)[-20:]
    top_team_names = [teams_id_to_name_dict[int(i)] for i in top_team_inds]
    top_team_means = mean[top_team_inds]
    stds = jnp.abs(sync_factorial_state.chol_cov[..., 0, 0])
    top_team_stds = stds[top_team_inds]

    plt.figure()
    plt.barh(top_team_names, top_team_means, xerr=top_team_stds, color="limegreen")
    last_match_date = football_data["date"].max().strftime("%Y-%m-%d")
    plt.xlabel(f"Skill Rating {last_match_date}")
    plt.tight_layout()
    plt.savefig("docs/assets/international_football_latest_skill_rating.png", dpi=300)
    plt.close()
    ```

![Best teams right now](assets/international_football_latest_skill_rating.png)



TODO: Add factorial smoother, bit more complex but uses the same filter as synchronize
 

## Build and run the smoother

The filtering distribution gives us live estimates with uncertainty. However,
for historical evaluation we want to use smoothing so that information is passed
backwards too.

With `cuthbert` this is just as easy as filtering.

```{.python #factorial-football-build-smoother}
factor_states = factorial.serial_to_factorial(factorializer.extract, local_states, match_team_indices[1:], init_state)


football_smoother = taylor.build_smoother(get_dynamics_log_density)
smoother_states = cuthbert.factorial.smoother(football_smoother, filter_states, match_data)
```


## Ok so who are the best teams historically?

??? quote "Code to extract and plot the historical smoothed distribution"

    ```{.python #factorial-football-extract-historical-distribution}
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

    plt.figure()
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
    plt.tight_layout()
    plt.savefig("docs/assets/international_football_historical_skill_rating.png", dpi=300)
    plt.close()
    ```

![Best teams historically](assets/international_football_historical_skill_rating.png)


## Key Takeaways

- **Flexible model specification**: `cuthbert.gaussian.taylor` allows you to define
  state-space models using simple log-density functions, making it easy to work
  with complex, non-linear models like the Elo-style ranking model used here.
- **Filtering for online inference**: `cuthbert.filter` can be used to offline
  filtering on a full dataset, `filter_prepare` and `filter_combine` can be used
  to perform online filtering as new data arrives.
- **Smoothing for historical analysis**: While filtering provides online estimates,
  smoothing gives more accurate historical estimates by incorporating future
  information.


## Next Steps

- **Parameter learning**: We could learn the hyperparameters from the data using
    gradient descent, expectation maximization or Bayesian sampling that all use
    filtering and smoothing internally. Check out the [parameter estimation example](examples/parameter_estimation_em.md) for more details.
- **Factorial state-space models**: The technique here is actually inefficient for this
    model because it treats all teams as a high-dimensional correlated state. A more
    efficient approach would be to use a factorial state-space model where each team's
    skill is assumed to evolve independently (aside from pairwsie interactions at matches).
    See [Duffield et al](https://doi.org/10.1093/jrsssc/qlae035) for more details, and
    `cuthbert` support coming soon!
- **More examples!**: Check out the other [examples](examples/index.md) for more
    techniques including exact Kalman inference, sequential Monte Carlo, interfacing
    with probabilistic programming languages, and more.


<!--- entangled-tangle-block
```{.python file=examples_scripts/factorial_football.py}
<<factorial-football-imports>>
<<factorial-football-load-data>>
<<factorial-football-load-data-jax>>
<<factorial-football-no-initial-obs>>
<<factorial-football-model-inputs>>
<<factorial-football-state-space-model>>
<<factorial-football-build-filter>>
<<factorial-football-build-factorializer>>
<<factorial-football-run-filter>>
<<factorial-football-sync>>
<<factorial-football-extract-filtered-distribution>>
```
-->
