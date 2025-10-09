import pandas as pd
from jax import random, jit, numpy as jnp, Array
import matplotlib.pyplot as plt

from cuthbert import filter
from cuthbert.gaussian import taylor


def load_international_football_data(
    start_date: str = "1872-11-30",
    end_date: str | None = None,
    origin_date: str | None = None,
    min_matches: int = 300,
) -> tuple[Array, Array, Array, dict, dict]:
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
        origin_date: The date to use as the zero point the output timestamps, defaults
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
    data_all["Timestamp"] = pd.to_datetime(data_all["date"])
    data_all["TimestampDays"] = (data_all["Timestamp"] - origin_timestamp).dt.days
    data_all = data_all[
        (data_all["Timestamp"] >= start_date) & (data_all["Timestamp"] <= end_date)
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
    data_all["HomeTeamID"] = data_all["home_team"].apply(
        lambda s: teams_name_to_id_dict[s]
    )
    data_all["AwayTeamID"] = data_all["away_team"].apply(
        lambda s: teams_name_to_id_dict[s]
    )

    # Extract output data into JAX arrays
    match_times = jnp.array(data_all["TimestampDays"])
    match_team_indices = jnp.array(data_all[["HomeTeamID", "AwayTeamID"]])
    home_goals = jnp.array(data_all["home_score"])
    away_goals = jnp.array(data_all["away_score"])
    match_results = jnp.where(
        home_goals > away_goals, 1, jnp.where(home_goals < away_goals, 2, 0)
    )
    return (
        match_times,
        match_team_indices,
        match_results,
        teams_id_to_name_dict,
        teams_name_to_id_dict,
    )
