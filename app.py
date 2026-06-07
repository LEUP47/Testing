"""Streamlit app for FIFA World Cup 2026 match outcome prediction.

The app downloads public international football results, engineers ELO and
recent-form features, trains a scikit-learn classifier, and predicts match
outcomes from the perspective of the first selected team.
"""

from __future__ import annotations

import math
import unicodedata
from io import StringIO
from textwrap import dedent
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import altair as alt
import certifi
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REQUESTED_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international-football-results-from-1872-to-2017/master/results.csv"
)
MAINTAINED_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
WORLD_CUP_2026_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/"
    "master/2026/worldcup.json"
)
RESULTS_URLS = (REQUESTED_RESULTS_URL, MAINTAINED_RESULTS_URL)
CACHE_TTL_SECONDS = 12 * 60 * 60
TRAINING_START_DATE = pd.Timestamp("2000-01-01")

FEATURE_COLUMNS = [
    "team1_elo_rating",
    "team2_elo_rating",
    "elo_difference",
    "team1_form",
    "team2_form",
    "is_neutral",
]

GROUP_ORDER = tuple(f"Group {letter}" for letter in "ABCDEFGHIJKL")
SIMULATION_COLUMNS = [
    "reached_r32",
    "reached_qf",
    "reached_sf",
    "reached_final",
    "won_tournament",
]
BRACKET_SEED_ORDER = (
    0,
    31,
    15,
    16,
    7,
    24,
    8,
    23,
    3,
    28,
    12,
    19,
    4,
    27,
    11,
    20,
    1,
    30,
    14,
    17,
    6,
    25,
    9,
    22,
    2,
    29,
    13,
    18,
    5,
    26,
    10,
    21,
)

HOST_NATIONS = {"Canada", "Mexico", "USA", "United States"}
HOST_GROUNDS = {
    "Canada": ("Toronto", "Vancouver"),
    "Mexico": ("Guadalajara", "Mexico City", "Monterrey", "Zapopan"),
    "USA": (
        "Atlanta",
        "Boston",
        "Dallas",
        "East Rutherford",
        "Houston",
        "Kansas City",
        "Los Angeles",
        "Miami",
        "New York/New Jersey",
        "Philadelphia",
        "San Francisco Bay Area",
        "Seattle",
    ),
}
DEFAULT_ELO = 1500.0
ELO_K_FACTOR = 24.0


WORLD_CUP_2026_TEAMS: Tuple[str, ...] = (
    "Algeria",
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Cabo Verde",
    "Canada",
    "Colombia",
    "Congo DR",
    "Croatia",
    "Curaçao",
    "Czechia",
    "Ecuador",
    "Egypt",
    "England",
    "France",
    "Germany",
    "Ghana",
    "Haiti",
    "IR Iran",
    "Iraq",
    "Japan",
    "Jordan",
    "Korea Republic",
    "Mexico",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Panama",
    "Paraguay",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Scotland",
    "Senegal",
    "South Africa",
    "Spain",
    "Sweden",
    "Switzerland",
    "Tunisia",
    "Türkiye",
    "USA",
    "Uruguay",
    "Uzbekistan",
    "Côte d'Ivoire",
)

TEAM_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "Bosnia and Herzegovina": ("Bosnia & Herzegovina",),
    "Cabo Verde": ("Cape Verde",),
    "Congo DR": ("DR Congo", "Democratic Republic of the Congo", "Zaire"),
    "Curaçao": ("Curacao",),
    "Czechia": ("Czech Republic", "Czechoslovakia"),
    "IR Iran": ("Iran",),
    "Korea Republic": ("South Korea",),
    "Türkiye": ("Turkey",),
    "USA": ("United States",),
    "United States": ("USA", "United States"),
    "Côte d'Ivoire": ("Ivory Coast", "Cote d'Ivoire"),
}

BASELINE_STRENGTHS: Mapping[str, float] = {
    "Argentina": 1865.0,
    "France": 1840.0,
    "Spain": 1815.0,
    "England": 1805.0,
    "Brazil": 1800.0,
    "Portugal": 1785.0,
    "Netherlands": 1765.0,
    "Belgium": 1750.0,
    "Germany": 1740.0,
    "Croatia": 1720.0,
    "Uruguay": 1710.0,
    "Morocco": 1705.0,
    "Colombia": 1700.0,
    "Switzerland": 1685.0,
    "USA": 1660.0,
    "United States": 1660.0,
    "Mexico": 1655.0,
    "Japan": 1650.0,
    "Senegal": 1645.0,
    "Austria": 1640.0,
    "Denmark": 1635.0,
    "Ecuador": 1630.0,
    "IR Iran": 1625.0,
    "Sweden": 1620.0,
    "Norway": 1615.0,
    "Korea Republic": 1605.0,
    "Australia": 1600.0,
    "Côte d'Ivoire": 1595.0,
    "Qatar": 1585.0,
    "Egypt": 1580.0,
    "Türkiye": 1575.0,
    "Canada": 1570.0,
    "Saudi Arabia": 1565.0,
    "Paraguay": 1560.0,
    "Tunisia": 1555.0,
    "Algeria": 1550.0,
    "Scotland": 1545.0,
    "South Africa": 1535.0,
    "Uzbekistan": 1530.0,
    "Ghana": 1525.0,
    "Panama": 1520.0,
    "Iraq": 1515.0,
    "Jordan": 1505.0,
    "Bosnia and Herzegovina": 1500.0,
    "Czechia": 1495.0,
    "Congo DR": 1490.0,
    "New Zealand": 1485.0,
    "Haiti": 1465.0,
    "Cabo Verde": 1460.0,
    "Curaçao": 1450.0,
}


ROSTER_METRICS: Mapping[str, Tuple[float, float]] = {
    "Argentina": (1100.0, 0.65),
    "France": (1250.0, 0.75),
    "England": (1400.0, 0.80),
    "Brazil": (1050.0, 0.70),
    "Spain": (1000.0, 0.60),
    "Portugal": (950.0, 0.65),
    "Netherlands": (750.0, 0.55),
    "Germany": (850.0, 0.70),
    "USA": (350.0, 0.40),
    "United States": (350.0, 0.40),
    "Mexico": (220.0, 0.35),
    "Canada": (180.0, 0.30),
    "Japan": (300.0, 0.45),
    "Morocco": (380.0, 0.50),
    "Czechia": (150.0, 0.30),
}
ROSTER_METRICS_NOTE = (
    "Squad values should be refreshed from national-team squad market values. "
    "Fatigue should be computed as likely starters above 3,500 club minutes "
    "divided by 11."
)


@dataclass(frozen=True)
class TeamState:
    """Latest model-ready state for one national team."""

    elo: float
    form: float


def normalize_team_name(name: str) -> str:
    """Normalize a team name for accent-insensitive matching."""

    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(char for char in normalized if not unicodedata.combining(char))
    return ascii_name.casefold().replace(".", "").replace("-", " ").strip()


def build_team_lookup(available_teams: Iterable[str]) -> Dict[str, str]:
    """Create a normalized lookup table from known names and aliases."""

    lookup = {normalize_team_name(team): team for team in available_teams}
    for display_name, aliases in TEAM_ALIASES.items():
        for alias in (display_name, *aliases):
            normalized = normalize_team_name(alias)
            if normalized in lookup:
                lookup[normalize_team_name(display_name)] = lookup[normalized]
                break
    return lookup


def build_display_lookup() -> Dict[str, str]:
    """Create a normalized lookup table from aliases to World Cup UI names."""

    lookup = {normalize_team_name(team): team for team in WORLD_CUP_2026_TEAMS}
    for display_name, aliases in TEAM_ALIASES.items():
        if display_name in WORLD_CUP_2026_TEAMS:
            for alias in aliases:
                lookup[normalize_team_name(alias)] = display_name
    return lookup


def resolve_display_name(name: str, display_lookup: Mapping[str, str]) -> Optional[str]:
    """Resolve schedule team names to the app's World Cup team labels."""

    return display_lookup.get(normalize_team_name(name))


def build_custom_team_options(available_teams: Iterable[str]) -> Tuple[str, ...]:
    """Build custom-matchup choices from World Cup teams and historical teams."""

    return tuple(sorted(set(WORLD_CUP_2026_TEAMS).union(available_teams)))


def read_csv_from_public_urls(
    urls: Tuple[str, ...],
    parse_dates: List[str],
) -> Tuple[pd.DataFrame, str]:
    """Read a CSV from the first reachable public URL with SSL fallbacks."""

    errors: List[str] = []
    for url in urls:
        try:
            response = fetch_public_url(url)
            csv_frame = pd.read_csv(StringIO(response.text), parse_dates=parse_dates)
            return csv_frame, url
        except Exception as exc:
            errors.append(f"{url} ({exc})")

    joined_errors = "; ".join(errors)
    raise ValueError(f"Unable to fetch CSV data from public URLs: {joined_errors}")


def fetch_public_url(url: str) -> requests.Response:
    """Fetch a public data URL with a certifi-backed SSL context."""

    headers = {"User-Agent": "world-cup-2026-streamlit-app/1.0"}
    try:
        response = requests.get(
            url,
            timeout=30,
            headers=headers,
            verify=certifi.where(),
        )
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        response = requests.get(
            url,
            timeout=30,
            headers=headers,
            verify=False,
        )
        response.raise_for_status()
        return response


@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    show_spinner="Downloading 2026 World Cup fixture data...",
)
def load_world_cup_2026_schedule() -> pd.DataFrame:
    """Fetch and deeply unpack the 2026 World Cup fixture list."""

    response = fetch_public_url(WORLD_CUP_2026_URL)
    payload = response.json()

    all_matches: List[Dict[str, Optional[str]]] = []
    if "rounds" in payload:
        for round_data in payload["rounds"]:
            group_name = round_data.get("name", "Unknown Round")
            for match in round_data.get("matches", []):
                all_matches.append(
                    {
                        "date": match.get("date"),
                        "group": group_name,
                        "team1": match.get("team1"),
                        "team2": match.get("team2"),
                        "ground": match.get("venue", match.get("ground")),
                    }
                )
    elif "matches" in payload:
        for match in payload["matches"]:
            all_matches.append(
                {
                    "date": match.get("date"),
                    "group": match.get("group", match.get("round", "Unknown Round")),
                    "team1": match.get("team1"),
                    "team2": match.get("team2"),
                    "ground": match.get("ground", match.get("venue")),
                }
            )

    schedule = pd.DataFrame(all_matches)
    if schedule.empty:
        return pd.DataFrame(columns=["date", "group", "team1", "team2", "ground"])

    schedule["ground"] = schedule["ground"].fillna("Unknown Stadium")
    schedule["date"] = pd.to_datetime(schedule["date"])
    return schedule


@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    show_spinner="Downloading international match data...",
)
def load_match_data() -> Tuple[pd.DataFrame, str]:
    """Fetch historical match data from public GitHub URLs."""

    results, results_source = read_csv_from_public_urls(
        RESULTS_URLS,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "country",
        "neutral",
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Results data is missing required columns: {missing}")

    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["neutral"] = results["neutral"].astype(bool)
    results = results[results["date"] >= TRAINING_START_DATE]
    results = results.sort_values("date").reset_index(drop=True)

    return results, results_source


def match_points(goals_for: int, goals_against: int) -> int:
    """Return football table points for a single match result."""

    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def target_from_scores(team1_score: int, team2_score: int) -> int:
    """Map Team 1 result to target class: 0 loss, 1 draw, 2 win."""

    if team1_score > team2_score:
        return 2
    if team1_score == team2_score:
        return 1
    return 0


def expected_result(team_rating: float, opponent_rating: float) -> float:
    """Calculate standard ELO expected score."""

    return 1.0 / (1.0 + math.pow(10.0, (opponent_rating - team_rating) / 400.0))


def update_elo(
    team_rating: float,
    opponent_rating: float,
    team_score: int,
    opponent_score: int,
) -> float:
    """Update one team's ELO rating from a match result."""

    if team_score > opponent_score:
        actual = 1.0
    elif team_score == opponent_score:
        actual = 0.5
    else:
        actual = 0.0

    margin_multiplier = 1.0 + math.log1p(abs(team_score - opponent_score))
    expected = expected_result(team_rating, opponent_rating)
    return team_rating + ELO_K_FACTOR * margin_multiplier * (actual - expected)


def calculate_form(recent_points: Deque[int]) -> float:
    """Return points accumulated over the latest five matches."""

    return float(sum(recent_points))


def engineer_training_features(
    results: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, TeamState]]:
    """Engineer pre-match ELO and recent-form features for model training."""

    elo_ratings: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    recent_form: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=5))
    rows: List[Dict[str, float | int | bool]] = []

    for match in results.itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        home_score = int(match.home_score)
        away_score = int(match.away_score)

        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]
        home_form = calculate_form(recent_form[home_team])
        away_form = calculate_form(recent_form[away_team])

        rows.append(
            {
                "team1_elo_rating": home_elo,
                "team2_elo_rating": away_elo,
                "elo_difference": home_elo - away_elo,
                "team1_form": home_form,
                "team2_form": away_form,
                "is_neutral": bool(match.neutral),
                "target": target_from_scores(home_score, away_score),
            }
        )

        next_home_elo = update_elo(home_elo, away_elo, home_score, away_score)
        next_away_elo = update_elo(away_elo, home_elo, away_score, home_score)
        elo_ratings[home_team] = next_home_elo
        elo_ratings[away_team] = next_away_elo

        recent_form[home_team].append(match_points(home_score, away_score))
        recent_form[away_team].append(match_points(away_score, home_score))

    features = pd.DataFrame(rows)
    team_states = {
        team: TeamState(elo=rating, form=calculate_form(recent_form[team]))
        for team, rating in elo_ratings.items()
    }

    return features, team_states


@st.cache_resource(show_spinner="Training match outcome model...")
def train_model(
    results: pd.DataFrame,
) -> Tuple[Pipeline, Dict[str, TeamState], Dict[str, float]]:
    """Train and cache the scikit-learn prediction pipeline."""

    feature_frame, team_states = engineer_training_features(results)
    x = feature_frame[FEATURE_COLUMNS]
    y = feature_frame["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    n_estimators=180,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        "matches": float(len(feature_frame)),
        "teams": float(len(team_states)),
    }

    return model, team_states, metrics


def get_team_state(
    display_name: str,
    team_states: Mapping[str, TeamState],
    team_lookup: Mapping[str, str],
) -> TeamState:
    """Resolve team state, anchoring baseline strength plus player metrics."""

    normalized_display = normalize_team_name(display_name)
    dataset_name = team_lookup.get(normalized_display)

    hist_elo = DEFAULT_ELO
    hist_form = 7.0

    if dataset_name and dataset_name in team_states:
        hist_elo = team_states[dataset_name].elo
        hist_form = team_states[dataset_name].form
    else:
        for alias in TEAM_ALIASES.get(display_name, ()):
            normalized_alias = normalize_team_name(alias)
            alt_name = team_lookup.get(normalized_alias)
            if alt_name and alt_name in team_states:
                hist_elo = team_states[alt_name].elo
                hist_form = team_states[alt_name].form
                break

    baseline_elo = BASELINE_STRENGTHS.get(display_name, DEFAULT_ELO)
    calibrated_elo = (baseline_elo * 0.70) + (hist_elo * 0.30)
    squad_value, fatigue_idx = ROSTER_METRICS.get(display_name, (50.0, 0.20))
    roster_modifier = (math.log10(squad_value) * 25.0) - (fatigue_idx * 40.0)
    final_elo = calibrated_elo + roster_modifier

    return TeamState(elo=final_elo, form=hist_form)


def build_prediction_frame(
    team1: str,
    team2: str,
    team_states: Mapping[str, TeamState],
    team_lookup: Mapping[str, str],
    ground: Optional[str] = None,
) -> pd.DataFrame:
    """Create one model input row for a selected 2026 World Cup matchup."""

    team1_state = get_team_state(team1, team_states, team_lookup)
    team2_state = get_team_state(team2, team_states, team_lookup)
    is_neutral = infer_neutrality(team1, ground)

    return pd.DataFrame(
        [
            {
                "team1_elo_rating": team1_state.elo,
                "team2_elo_rating": team2_state.elo,
                "elo_difference": team1_state.elo - team2_state.elo,
                "team1_form": team1_state.form,
                "team2_form": team2_state.form,
                "is_neutral": is_neutral,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def infer_neutrality(team1: str, ground: Optional[str] = None) -> bool:
    """Infer whether a World Cup match is neutral from Team 1's perspective."""

    if team1 not in HOST_NATIONS:
        return True
    if not ground:
        return False

    host_key = "USA" if team1 in {"USA", "United States"} else team1
    return not any(location in ground for location in HOST_GROUNDS.get(host_key, ()))


def probability_by_class(
    model: Pipeline,
    prediction_frame: pd.DataFrame,
) -> Dict[int, float]:
    """Return class probabilities keyed by target label."""

    classifier = model.named_steps["classifier"]
    classes = classifier.classes_
    probabilities = model.predict_proba(prediction_frame)[0]
    return {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }


def prepare_group_stage_matches(
    schedule: pd.DataFrame,
    display_lookup: Mapping[str, str],
) -> pd.DataFrame:
    """Resolve and filter the 2026 schedule down to group-stage matches."""

    resolved_schedule = schedule.copy()
    resolved_schedule["team1_resolved"] = resolved_schedule["team1"].apply(
        lambda name: resolve_display_name(str(name), display_lookup)
    )
    resolved_schedule["team2_resolved"] = resolved_schedule["team2"].apply(
        lambda name: resolve_display_name(str(name), display_lookup)
    )
    group_matches = resolved_schedule[
        resolved_schedule["group"].isin(GROUP_ORDER)
        & resolved_schedule["team1_resolved"].notna()
        & resolved_schedule["team2_resolved"].notna()
    ].copy()

    group_matches["team1"] = group_matches["team1_resolved"]
    group_matches["team2"] = group_matches["team2_resolved"]
    return group_matches[["date", "group", "team1", "team2", "ground"]].reset_index(
        drop=True
    )


def serialize_tournament_team_states(
    team_states: Mapping[str, TeamState],
    team_lookup: Mapping[str, str],
) -> Tuple[Tuple[str, float, float], ...]:
    """Resolve all World Cup teams into hashable state records."""

    records = []
    for team in WORLD_CUP_2026_TEAMS:
        state = get_team_state(team, team_states, team_lookup)
        records.append((team, float(state.elo), float(state.form)))
    return tuple(records)


def deserialize_team_states(
    records: Tuple[Tuple[str, float, float], ...],
) -> Dict[str, TeamState]:
    """Convert hashable team state records back to a TeamState mapping."""

    return {team: TeamState(elo=elo, form=form) for team, elo, form in records}


def build_probability_cache(
    model: Pipeline,
    team_states: Mapping[str, TeamState],
) -> Tuple[Dict[Tuple[str, str, bool], np.ndarray], np.ndarray]:
    """Precompute ordered-pair probabilities for fast simulation sampling."""

    keys: List[Tuple[str, str, bool]] = []
    rows: List[Dict[str, float | bool]] = []

    for is_neutral in (False, True):
        for team1, team1_state in team_states.items():
            for team2, team2_state in team_states.items():
                if team1 == team2:
                    continue
                keys.append((team1, team2, is_neutral))
                rows.append(
                    {
                        "team1_elo_rating": team1_state.elo,
                        "team2_elo_rating": team2_state.elo,
                        "elo_difference": team1_state.elo - team2_state.elo,
                        "team1_form": team1_state.form,
                        "team2_form": team2_state.form,
                        "is_neutral": is_neutral,
                    }
                )

    probability_frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(probability_frame)
    class_labels = np.asarray(model.named_steps["classifier"].classes_, dtype=int)
    probability_cache = {
        key: probabilities[index].astype(float) for index, key in enumerate(keys)
    }
    return probability_cache, class_labels


def sample_match_result(
    team1: str,
    team2: str,
    is_neutral: bool,
    probability_cache: Mapping[Tuple[str, str, bool], np.ndarray],
    class_labels: np.ndarray,
) -> int:
    """Sample one match outcome from the model's class probability array."""

    probabilities = probability_cache[(team1, team2, is_neutral)]
    return int(np.random.choice(class_labels, p=probabilities))


def rank_group(
    group_name: str,
    standings: Mapping[str, Dict[str, float] | List[float]],
) -> List[Dict[str, float | str | int]]:
    """Sort one group's standings by points and ELO tie-breaker."""

    ranked_rows = []
    for team, values in standings.items():
        if isinstance(values, dict):
            points = values["points"]
            elo = values["elo"]
        else:
            points = values[0]
            elo = values[1]
        ranked_rows.append(
            {
                "group": group_name,
                "team": team,
                "points": int(points),
                "elo": float(elo),
            }
        )
    return sorted(
        ranked_rows,
        key=lambda row: (-int(row["points"]), -float(row["elo"]), str(row["team"])),
    )


def build_group_simulation_inputs(
    group_matches: pd.DataFrame,
    team_states: Mapping[str, TeamState],
) -> Tuple[
    Tuple[Tuple[str, str, str, bool], ...],
    Dict[str, Dict[str, float]],
]:
    """Precompute immutable group fixture records and standings templates."""

    group_records: List[Tuple[str, str, str, bool]] = []
    standings_template: Dict[str, Dict[str, float]] = {}

    for match in group_matches.itertuples(index=False):
        group_name = str(match.group)
        team1 = str(match.team1)
        team2 = str(match.team2)
        is_neutral = infer_neutrality(team1, str(match.ground))
        group_records.append((group_name, team1, team2, is_neutral))

        standings_template.setdefault(group_name, {})
        standings_template[group_name].setdefault(team1, team_states[team1].elo)
        standings_template[group_name].setdefault(team2, team_states[team2].elo)

    return tuple(group_records), standings_template


def simulate_group_phase(
    group_records: Tuple[Tuple[str, str, str, bool], ...],
    standings_template: Mapping[str, Mapping[str, float]],
    probability_cache: Mapping[Tuple[str, str, bool], np.ndarray],
    class_labels: np.ndarray,
) -> List[Dict[str, float | str | int]]:
    """Simulate the 72 group matches and return 32 qualified teams."""

    standings = {
        group_name: {team: [0.0, elo] for team, elo in group_teams.items()}
        for group_name, group_teams in standings_template.items()
    }

    for group_name, team1, team2, is_neutral in group_records:
        result = sample_match_result(
            team1,
            team2,
            is_neutral,
            probability_cache,
            class_labels,
        )

        if result == 2:
            standings[group_name][team1][0] += 3.0
        elif result == 1:
            standings[group_name][team1][0] += 1.0
            standings[group_name][team2][0] += 1.0
        else:
            standings[group_name][team2][0] += 3.0

    qualified: List[Dict[str, float | str | int]] = []
    third_place_candidates: List[Dict[str, float | str | int]] = []
    ordered_groups = [group for group in GROUP_ORDER if group in standings]

    for group_name in ordered_groups:
        ranked_group = rank_group(group_name, standings[group_name])
        for position, row in enumerate(ranked_group, start=1):
            row["group_position"] = position

        qualified.extend(ranked_group[:2])
        third_place_candidates.append(ranked_group[2])

    wildcard_teams = sorted(
        third_place_candidates,
        key=lambda row: (-int(row["points"]), -float(row["elo"]), str(row["team"])),
    )[:8]
    qualified.extend(wildcard_teams)

    if len(qualified) != 32:
        raise ValueError(f"Expected 32 knockout qualifiers, found {len(qualified)}")

    return qualified


def seed_knockout_bracket(
    qualified_teams: List[Dict[str, float | str | int]],
) -> List[str]:
    """Create a deterministic 32-team knockout bracket from qualifiers."""

    ordered_qualifiers = sorted(
        qualified_teams,
        key=lambda row: (
            int(row["group_position"]),
            -int(row["points"]),
            -float(row["elo"]),
            str(row["team"]),
        ),
    )
    return [str(ordered_qualifiers[index]["team"]) for index in BRACKET_SEED_ORDER]


def resolve_knockout_match(
    team1: str,
    team2: str,
    team_states: Mapping[str, TeamState],
    probability_cache: Mapping[Tuple[str, str, bool], np.ndarray],
    class_labels: np.ndarray,
) -> str:
    """Resolve a single-elimination match, including draw tie-breakers."""

    result = sample_match_result(
        team1,
        team2,
        True,
        probability_cache,
        class_labels,
    )
    if result == 2:
        return team1
    if result == 0:
        return team2

    elo_difference = team_states[team1].elo - team_states[team2].elo
    team1_tiebreak_probability = float(np.clip(0.5 + elo_difference / 1200, 0.35, 0.65))
    return str(
        np.random.choice(
            [team1, team2],
            p=[team1_tiebreak_probability, 1.0 - team1_tiebreak_probability],
        )
    )


def play_knockout_round(
    teams: List[str],
    team_states: Mapping[str, TeamState],
    probability_cache: Mapping[Tuple[str, str, bool], np.ndarray],
    class_labels: np.ndarray,
) -> List[str]:
    """Play one knockout round and return advancing teams in bracket order."""

    if len(teams) % 2 != 0:
        raise ValueError("Knockout rounds require an even number of teams")

    winners = []
    for index in range(0, len(teams), 2):
        winners.append(
            resolve_knockout_match(
                teams[index],
                teams[index + 1],
                team_states,
                probability_cache,
                class_labels,
            )
        )
    return winners


@st.cache_data(show_spinner="Running tournament simulations...")
def run_monte_carlo_simulation(
    iterations: int,
    group_matches: pd.DataFrame,
    team_state_records: Tuple[Tuple[str, float, float], ...],
    _model: Pipeline,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Run repeated World Cup simulations and aggregate stage probabilities."""

    if iterations < 1:
        raise ValueError("Simulation iterations must be at least 1")

    np.random.seed(random_seed)
    team_states = deserialize_team_states(team_state_records)
    probability_cache, class_labels = build_probability_cache(_model, team_states)
    group_records, standings_template = build_group_simulation_inputs(
        group_matches,
        team_states,
    )
    counters: Dict[str, Dict[str, int]] = {
        team: {column: 0 for column in SIMULATION_COLUMNS}
        for team in WORLD_CUP_2026_TEAMS
    }

    for _ in range(iterations):
        qualified_teams = simulate_group_phase(
            group_records,
            standings_template,
            probability_cache,
            class_labels,
        )
        round_of_32 = seed_knockout_bracket(qualified_teams)
        for team in round_of_32:
            counters[team]["reached_r32"] += 1

        round_of_16 = play_knockout_round(
            round_of_32,
            team_states,
            probability_cache,
            class_labels,
        )
        quarterfinalists = play_knockout_round(
            round_of_16,
            team_states,
            probability_cache,
            class_labels,
        )
        for team in quarterfinalists:
            counters[team]["reached_qf"] += 1

        semifinalists = play_knockout_round(
            quarterfinalists,
            team_states,
            probability_cache,
            class_labels,
        )
        for team in semifinalists:
            counters[team]["reached_sf"] += 1

        finalists = play_knockout_round(
            semifinalists,
            team_states,
            probability_cache,
            class_labels,
        )
        for team in finalists:
            counters[team]["reached_final"] += 1

        champion = play_knockout_round(
            finalists,
            team_states,
            probability_cache,
            class_labels,
        )[0]
        counters[champion]["won_tournament"] += 1

    leaderboard = pd.DataFrame(
        [
            {
                "team": team,
                **{
                    column: counters[team][column] / iterations * 100.0
                    for column in SIMULATION_COLUMNS
                },
            }
            for team in WORLD_CUP_2026_TEAMS
        ]
    )
    return leaderboard.sort_values(
        ["won_tournament", "reached_final", "reached_sf"],
        ascending=False,
    ).reset_index(drop=True)


def simulate_single_tournament(
    group_matches: pd.DataFrame,
    team_state_records: Tuple[Tuple[str, float, float], ...],
    _model: Pipeline,
    random_seed: int,
) -> Dict[str, List[str]]:
    """Simulate one tournament run and return concrete bracket paths."""

    np.random.seed(random_seed)
    team_states = deserialize_team_states(team_state_records)
    probability_cache, class_labels = build_probability_cache(_model, team_states)
    group_records, standings_template = build_group_simulation_inputs(
        group_matches,
        team_states,
    )

    qualified_teams = simulate_group_phase(
        group_records,
        standings_template,
        probability_cache,
        class_labels,
    )

    round_of_32 = seed_knockout_bracket(qualified_teams)
    round_of_16 = play_knockout_round(
        round_of_32,
        team_states,
        probability_cache,
        class_labels,
    )
    quarterfinals = play_knockout_round(
        round_of_16,
        team_states,
        probability_cache,
        class_labels,
    )
    semifinals = play_knockout_round(
        quarterfinals,
        team_states,
        probability_cache,
        class_labels,
    )
    finalists = play_knockout_round(
        semifinals,
        team_states,
        probability_cache,
        class_labels,
    )
    champion = play_knockout_round(
        finalists,
        team_states,
        probability_cache,
        class_labels,
    )[0]

    return {
        "Round of 32": round_of_32,
        "Round of 16": round_of_16,
        "Quarterfinals": quarterfinals,
        "Semifinals": semifinals,
        "Finalists": finalists,
        "Champion": [champion],
    }


def render_simulation_leaderboard(simulation_results: pd.DataFrame) -> None:
    """Render probability leaderboard and top-winner chart."""

    st.dataframe(
        simulation_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "team": st.column_config.TextColumn("Team", width="medium"),
            "reached_r32": st.column_config.ProgressColumn(
                "Group survival",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "reached_qf": st.column_config.ProgressColumn(
                "Quarterfinals",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "reached_sf": st.column_config.ProgressColumn(
                "Semifinals",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "reached_final": st.column_config.ProgressColumn(
                "Final",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "won_tournament": st.column_config.ProgressColumn(
                "Win World Cup",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    top_winners = simulation_results.nlargest(10, "won_tournament")
    color_scale = alt.Scale(
        range=[
            "#4fc3f7", "#29b6f6", "#039be5", "#0288d1", "#0277bd",
            "#01579b", "#1a237e", "#311b92", "#4a148c", "#880e4f",
        ]
    )
    chart = (
        alt.Chart(top_winners)
        .mark_bar(
            cornerRadiusTopRight=6,
            cornerRadiusBottomRight=6,
        )
        .encode(
            x=alt.X(
                "won_tournament:Q",
                title="Tournament win probability (%)",
                axis=alt.Axis(grid=True, gridColor="#1e293b", labelColor="#94a3b8", titleColor="#cbd5e1"),
            ),
            y=alt.Y(
                "team:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelColor="#e2e8f0", labelFontSize=13),
            ),
            color=alt.Color(
                "won_tournament:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("won_tournament:Q", title="Win %", format=".1f"),
                alt.Tooltip("reached_final:Q", title="Final %", format=".1f"),
                alt.Tooltip("reached_sf:Q", title="Semifinal %", format=".1f"),
            ],
        )
        .properties(
            height=380,
            background="#0f172a",
            title=alt.TitleParams(
                text="🏆 Top 10 — World Cup Win Probability",
                color="#f1f5f9",
                fontSize=16,
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def render_plotly_bracket(bracket_data: Dict[str, List[str]]) -> None:
    """Render a symmetrical Plotly knockout bracket tree."""

    round_of_32 = bracket_data["Round of 32"]
    round_of_16 = bracket_data["Round of 16"]
    quarterfinals = bracket_data["Quarterfinals"]
    semifinals = bracket_data["Semifinals"]
    finalists = bracket_data["Finalists"]
    champion = bracket_data["Champion"][0]

    left_rounds = [
        ("R32", 0.0, round_of_32[:16]),
        ("R16", 1.0, round_of_16[:8]),
        ("QF", 2.0, quarterfinals[:4]),
        ("SF", 3.0, semifinals[:2]),
        ("Finalist", 4.0, finalists[:1]),
    ]
    right_rounds = [
        ("R32", 10.0, round_of_32[16:]),
        ("R16", 9.0, round_of_16[8:]),
        ("QF", 8.0, quarterfinals[4:]),
        ("SF", 7.0, semifinals[2:]),
        ("Finalist", 6.0, finalists[1:]),
    ]

    y_positions = {
        16: list(np.linspace(4, 116, 16)),
        8: list(np.linspace(8, 112, 8)),
        4: list(np.linspace(20, 100, 4)),
        2: [40.0, 80.0],
        1: [60.0],
    }
    node_width = 0.78
    node_height = 3.6
    shapes = []
    annotations = []
    hover_x: List[float] = []
    hover_y: List[float] = []
    hover_text: List[str] = []

    def node_style(round_label: str) -> Tuple[str, str, int, str]:
        """Return (fill_color, border_color, font_size, font_color)."""
        if round_label == "Champion":
            return "#0d1f0a", "#ffd700", 13, "#ffffff"
        if round_label == "Finalist":
            return "#1e3a5f", "#63b3ed", 11, "#e2e8f0"
        if round_label == "SF":
            return "#1a2744", "#4299e1", 10, "#e2e8f0"
        if round_label == "QF":
            return "#152035", "#2b6cb0", 10, "#cbd5e1"
        return "#0f172a", "#334155", 9, "#94a3b8"

    def add_node(team: str, x_value: float, y_value: float, round_label: str) -> None:
        fill_color, border_color, font_size, font_color = node_style(round_label)
        width = 1.1 if round_label == "Champion" else node_width
        height = 5.0 if round_label == "Champion" else node_height
        border_width = 2.5 if round_label == "Champion" else 1.3
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": x_value - width / 2,
                "x1": x_value + width / 2,
                "y0": y_value - height / 2,
                "y1": y_value + height / 2,
                "line": {"color": border_color, "width": border_width},
                "fillcolor": fill_color,
                "layer": "below",
            }
        )
        label = f"🏆 <b>{team}</b>" if round_label == "Champion" else f"<b>{team}</b>"
        annotations.append(
            {
                "x": x_value,
                "y": y_value,
                "xref": "x",
                "yref": "y",
                "text": label,
                "showarrow": False,
                "font": {"size": font_size, "color": font_color},
                "align": "center",
            }
        )
        hover_x.append(x_value)
        hover_y.append(y_value)
        hover_text.append(f"{round_label}: {team}")

    def add_connector(
        figure: go.Figure,
        x_from: float,
        y_one: float,
        y_two: float,
        x_to: float,
        y_winner: float,
    ) -> None:
        direction = 1 if x_to > x_from else -1
        start_x = x_from + direction * (node_width / 2)
        end_x = x_to - direction * (node_width / 2)
        elbow_x = (start_x + end_x) / 2
        figure.add_trace(
            go.Scatter(
                x=[
                    start_x,
                    elbow_x,
                    None,
                    start_x,
                    elbow_x,
                    None,
                    elbow_x,
                    elbow_x,
                    None,
                    elbow_x,
                    end_x,
                ],
                y=[
                    y_one,
                    y_one,
                    None,
                    y_two,
                    y_two,
                    None,
                    y_one,
                    y_two,
                    None,
                    y_winner,
                    y_winner,
                ],
                mode="lines",
                line={"color": "#334155", "width": 1.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure()
    side_coordinates: Dict[str, List[Tuple[float, float]]] = {}

    for side_name, rounds in (("left", left_rounds), ("right", right_rounds)):
        for round_label, x_value, teams in rounds:
            coordinates = [(x_value, y) for y in y_positions[len(teams)]]
            side_coordinates[f"{side_name}_{round_label}"] = coordinates
            for team, (_, y_value) in zip(teams, coordinates):
                add_node(team, x_value, y_value, round_label)

    add_node(champion, 5.0, 60.0, "Champion")

    round_pairs = [
        ("R32", "R16"),
        ("R16", "QF"),
        ("QF", "SF"),
        ("SF", "Finalist"),
    ]
    for side_name in ("left", "right"):
        for source_round, target_round in round_pairs:
            source_coordinates = side_coordinates[f"{side_name}_{source_round}"]
            target_coordinates = side_coordinates[f"{side_name}_{target_round}"]
            for index, (_, target_y) in enumerate(target_coordinates):
                x_from = source_coordinates[index * 2][0]
                y_one = source_coordinates[index * 2][1]
                y_two = source_coordinates[index * 2 + 1][1]
                x_to = target_coordinates[index][0]
                add_connector(fig, x_from, y_one, y_two, x_to, target_y)

    for finalist_x, finalist_y in [(4.0, 60.0), (6.0, 60.0)]:
        fig.add_trace(
            go.Scatter(
                x=[finalist_x, 5.0],
                y=[finalist_y, 60.0],
                mode="lines",
                line={"color": "#ffd700", "width": 2.2},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker={"size": 18, "color": "rgba(0,0,0,0)"},
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": "🏆 Single-Seed Knockout Bracket — FIFA World Cup 2026",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#f1f5f9", "family": "Inter, sans-serif"},
        },
        height=960,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        shapes=shapes,
        annotations=annotations,
        xaxis={
            "range": [-0.75, 10.75],
            "tickmode": "array",
            "tickvals": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ticktext": [
                "R32",
                "R16",
                "QF",
                "SF",
                "Final",
                "🏆",
                "Final",
                "SF",
                "QF",
                "R16",
                "R32",
            ],
            "showgrid": False,
            "zeroline": False,
            "fixedrange": False,
            "tickfont": {"color": "#94a3b8", "size": 11, "family": "Inter, sans-serif"},
        },
        yaxis={
            "range": [-4, 124],
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "fixedrange": False,
        },
    )

    st.markdown("<hr style='border-color:rgba(255,215,0,0.15);margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='color:#f1f5f9;font-family:Inter,sans-serif;font-weight:800;font-size:18px;'>🗺️ Interactive Knockout Bracket</h3>"
        "<p style='color:#64748b;font-size:13px;margin-top:2px;font-family:Inter,sans-serif;'>A single tournament path generated from the selected random seed.</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_probability_card(label: str, probability: float) -> None:
    """Render a visually rich probability card."""

    pct = probability * 100
    bar_pct = int(min(max(pct, 0.0), 100.0))
    if pct >= 45:
        gradient = "linear-gradient(135deg, #1a3a1a 0%, #064e3b 100%)"
        bar_color = "#10b981"
        text_color = "#6ee7b7"
    elif pct >= 30:
        gradient = "linear-gradient(135deg, #1c2a3a 0%, #1e3a5f 100%)"
        bar_color = "#3b82f6"
        text_color = "#93c5fd"
    else:
        gradient = "linear-gradient(135deg, #1e1a2e 0%, #2d1b4e 100%)"
        bar_color = "#8b5cf6"
        text_color = "#c4b5fd"

    st.markdown(
        f"""
        <div style="
            background:{gradient};
            border-radius:16px;
            padding:20px 22px 16px 22px;
            margin-bottom:4px;
            border:1px solid rgba(255,255,255,0.08);
            box-shadow:0 4px 24px rgba(0,0,0,0.4);
        ">
            <div style="color:#94a3b8;font-size:12px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;font-family:'Inter',sans-serif;">{label}</div>
            <div style="color:{text_color};font-size:36px;font-weight:800;line-height:1;font-family:'Inter',sans-serif;">{pct:.1f}<span style="font-size:18px;font-weight:400;">%</span></div>
            <div style="margin-top:12px;background:rgba(255,255,255,0.08);border-radius:999px;height:6px;overflow:hidden;">
                <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:999px;transition:width 0.6s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def poisson_probability(k: int, lamb: float) -> float:
    """Calculate Poisson probability natively."""

    return (math.pow(lamb, k) * math.exp(-lamb)) / math.factorial(k)


def render_score_probability_matrix(
    team1: str,
    team2: str,
    team1_elo: float,
    team2_elo: float,
    is_neutral: bool,
) -> None:
    """Render exact scoreline probabilities from independent Poisson models."""

    exp_score1 = 1.0 / (1.0 + math.pow(10.0, (team2_elo - team1_elo) / 400.0))
    lambda1 = 2.7 * exp_score1
    lambda2 = 2.7 * (1.0 - exp_score1)

    if not is_neutral:
        lambda1 += 0.25
        lambda2 = max(0.1, lambda2 - 0.15)

    max_goals = 6
    matrix = np.zeros((max_goals, max_goals))
    for team1_goals in range(max_goals):
        for team2_goals in range(max_goals):
            p1 = poisson_probability(team1_goals, lambda1)
            p2 = poisson_probability(team2_goals, lambda2)
            matrix[team1_goals, team2_goals] = p1 * p2 * 100.0

    score_rankings = []
    for team1_goals in range(max_goals):
        for team2_goals in range(max_goals):
            score_rankings.append(
                (matrix[team1_goals, team2_goals], f"{team1_goals} - {team2_goals}")
            )
    score_rankings.sort(key=lambda item: item[0], reverse=True)

    st.markdown(
        "<hr style='border-color:rgba(255,215,0,0.15);margin:24px 0;'>",
        unsafe_allow_html=True,
    )
    header_html = f"""
    <h3 style="
        color:#f1f5f9;font-family:Inter,sans-serif;
        font-weight:800;font-size:18px;
    ">Exact Scoreline Probabilities</h3>
    <p style="
        color:#64748b;font-size:13px;margin-top:2px;
        font-family:Inter,sans-serif;
    ">
        Calculated via independent Poisson goal models. Estimated expected
        goals: {team1} ({lambda1:.2f}) vs {team2} ({lambda2:.2f})
    </p>
    """
    st.markdown(header_html.replace("\n", " "), unsafe_allow_html=True)

    chart_col, stats_col = st.columns([2, 1])
    with chart_col:
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[f"{goals} Goals" for goals in range(max_goals)],
                y=[f"{goals} Goals" for goals in range(max_goals)],
                colorscale="Viridis",
                hovertemplate=(
                    f"<b>{team1}</b>: %{{y}}<br>"
                    f"<b>{team2}</b>: %{{x}}<br>"
                    "Probability: %{z:.2f}%<extra></extra>"
                ),
                showscale=True,
                colorbar={"title": "%"},
            )
        )
        fig.update_layout(
            height=360,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font={"color": "#94a3b8", "family": "Inter, sans-serif"},
            xaxis={
                "title": f"{team2} goals",
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                "title": f"{team1} goals",
                "showgrid": False,
                "zeroline": False,
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with stats_col:
        st.markdown(
            """
            <div style="
                background:rgba(255,255,255,0.02);
                padding:16px;border-radius:12px;
                border:1px solid rgba(255,255,255,0.05);
                height:100%;
            ">
            <div style="
                font-size:11px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.05em;
                margin-bottom:12px;font-family:Inter,sans-serif;
            ">Top Score Predictions</div>
            """.replace(
                "\n", " "
            ),
            unsafe_allow_html=True,
        )
        for probability, score_str in score_rankings[:5]:
            row_html = f"""
            <div style="
                display:flex;justify-content:space-between;
                align-items:center;margin-bottom:8px;font-size:14px;
                font-family:Inter,sans-serif;
            ">
                <span style="
                    color:#cbd5e1;font-weight:600;min-width:60px;
                ">{score_str}</span>
                <span style="
                    color:#fbbf24;font-weight:700;margin-left:auto;
                ">{probability:.1f}%</span>
            </div>
            """
            st.markdown(row_html.replace("\n", " "), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def inject_styles() -> None:
    """Inject global CSS styles for the FIFA World Cup 2026 theme."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── Global reset & base ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        .stApp {
            background: linear-gradient(160deg, #050d1f 0%, #0a1628 40%, #0d1f3c 70%, #071020 100%) !important;
            background-attachment: fixed !important;
        }

        /* ── Decorative stadium grid overlay ── */
        .stApp::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,215,0,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,215,0,0.015) 1px, transparent 1px);
            background-size: 60px 60px;
            pointer-events: none;
            z-index: 0;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020817 0%, #0a1628 100%) !important;
            border-right: 1px solid rgba(255,215,0,0.15) !important;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stMetric {
            background: rgba(255,215,0,0.04);
            border: 1px solid rgba(255,215,0,0.12);
            border-radius: 10px;
            padding: 8px 14px;
            margin-bottom: 8px !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #fbbf24 !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            color: #94a3b8 !important;
            font-size: 13px !important;
        }

        /* ── Main content text ── */
        .stMarkdown p, .stMarkdown li {
            color: #cbd5e1 !important;
        }
        label, .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #94a3b8 !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
        }

        /* ── Selectboxes & inputs ── */
        .stSelectbox > div > div, .stNumberInput > div > div > input {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(99, 179, 237, 0.3) !important;
            color: #e2e8f0 !important;
            border-radius: 10px !important;
        }

        /* ── Primary button ── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%) !important;
            color: #0a0a0a !important;
            font-weight: 800 !important;
            font-size: 15px !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 14px 28px !important;
            letter-spacing: 0.04em !important;
            box-shadow: 0 4px 20px rgba(251,191,36,0.4) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(251,191,36,0.6) !important;
        }
        .stButton > button:not([kind="primary"]) {
            background: rgba(30, 58, 138, 0.5) !important;
            color: #93c5fd !important;
            border: 1px solid rgba(99, 179, 237, 0.3) !important;
            border-radius: 10px !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(15, 23, 42, 0.6) !important;
            border-radius: 12px !important;
            padding: 4px !important;
            border: 1px solid rgba(255,215,0,0.1) !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            border-radius: 8px !important;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(251,191,36,0.15) !important;
            color: #fbbf24 !important;
        }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,215,0,0.1) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(99,179,237,0.2) !important;
            border-radius: 10px !important;
            color: #93c5fd !important;
        }

        /* ── Info / Warning / Error boxes ── */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
        }

        /* ── Slider ── */
        [data-testid="stSlider"] .st-bd {
            background: rgba(251,191,36,0.3) !important;
        }

        /* ── Metrics (main area) ── */
        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 14px !important;
            padding: 16px !important;
        }
        [data-testid="stMetricValue"] {
            color: #fbbf24 !important;
            font-weight: 800 !important;
            font-size: 24px !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }
        [data-testid="stMetricDelta"] {
            color: #10b981 !important;
            font-weight: 600 !important;
        }

        /* ── Radio buttons ── */
        .stRadio [data-baseweb="radio"] span {
            color: #cbd5e1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header() -> None:
    """Render a premium hero header for the app."""

    hero_html = dedent(
        """
        <div style="
            background: linear-gradient(135deg, #050d1f 0%, #0d2137 50%, #071020 100%);
            border: 1px solid rgba(255,215,0,0.2);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 40px rgba(0,0,0,0.6);
        ">
            <!-- Glowing orb decoration -->
            <div style="
                position:absolute;top:-40px;right:-40px;
                width:200px;height:200px;
                background:radial-gradient(circle, rgba(255,215,0,0.12) 0%, transparent 70%);
                border-radius:50%;
                pointer-events:none;
            "></div>
            <div style="
                position:absolute;bottom:-60px;left:20%;
                width:300px;height:300px;
                background:radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%);
                border-radius:50%;
                pointer-events:none;
            "></div>

            <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
                <div style="
                    font-size:48px;
                    filter:drop-shadow(0 0 12px rgba(255,215,0,0.6));
                ">⚽</div>
                <div>
                    <div style="
                        font-size:11px;font-weight:700;
                        letter-spacing:0.2em;text-transform:uppercase;
                        color:#fbbf24;margin-bottom:4px;
                        font-family:'Inter',sans-serif;
                    ">FIFA</div>
                    <h1 style="
                        margin:0;padding:0;
                        font-size:clamp(22px, 3.5vw, 36px);
                        font-weight:900;
                        line-height:1.1;
                        font-family:'Inter',sans-serif;
                        background:linear-gradient(135deg,#ffffff 0%,#fbbf24 60%,#f59e0b 100%);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;
                        background-clip:text;
                    ">World Cup 2026<br>Match Predictor</h1>
                </div>
            </div>
            <p style="
                margin:0;color:#94a3b8;
                font-size:14px;line-height:1.6;
                max-width:600px;
                font-family:'Inter',sans-serif;
            ">
                Powered by a <strong style="color:#93c5fd;">Gradient Boosting</strong> model trained on
                historical international results — featuring ELO ratings, recent form, and venue-context features.
                Simulate full tournaments with Monte Carlo analysis.
            </p>

            <div style="
                display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;
            ">
                <span style="
                    background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.3);
                    color:#fbbf24;padding:4px 12px;border-radius:999px;
                    font-size:11px;font-weight:600;letter-spacing:0.06em;
                    font-family:'Inter',sans-serif;
                ">🇨🇦 Canada &nbsp;·&nbsp; 🇲🇽 Mexico &nbsp;·&nbsp; 🇺🇸 USA</span>
                <span style="
                    background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);
                    color:#93c5fd;padding:4px 12px;border-radius:999px;
                    font-size:11px;font-weight:600;letter-spacing:0.06em;
                    font-family:'Inter',sans-serif;
                ">48 Teams &nbsp;·&nbsp; 104 Matches</span>
                <span style="
                    background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);
                    color:#6ee7b7;padding:4px 12px;border-radius:999px;
                    font-size:11px;font-weight:600;letter-spacing:0.06em;
                    font-family:'Inter',sans-serif;
                ">Live Data Engine</span>
            </div>
        </div>
        """
    )
    st.markdown(hero_html.replace("\n", " "), unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title="FIFA World Cup 2026 Match Predictor",
        page_icon="⚽",
        layout="wide",
    )

    inject_styles()
    render_hero_header()

    try:
        results, results_source = load_match_data()
        world_cup_schedule = load_world_cup_2026_schedule()
        model, team_states, metrics = train_model(results)
    except Exception as exc:
        st.error(f"Unable to load data or train the model: {exc}")
        st.stop()

    available_teams = set(results["home_team"]).union(results["away_team"])
    team_lookup = build_team_lookup(available_teams)
    custom_team_options = build_custom_team_options(available_teams)

    display_lookup = build_display_lookup()
    group_stage_matches = prepare_group_stage_matches(
        world_cup_schedule,
        display_lookup,
    )
    app_view = st.sidebar.radio(
        "🧭 Vista",
        ("Match Predictor", "Tournament Simulation Engine"),
        label_visibility="collapsed",
    )

    if app_view == "Match Predictor":
        st.markdown(
            "<h2 style='color:#f1f5f9;font-family:Inter,sans-serif;font-weight:800;"
            "font-size:20px;margin-bottom:4px;'>⚽ Match Predictor</h2>",
            unsafe_allow_html=True,
        )
        prediction_mode = st.radio(
            "Prediction mode",
            ("World Cup fixture", "Custom matchup"),
            horizontal=True,
        )

        selected_ground: Optional[str] = None
        if prediction_mode == "World Cup fixture" and not group_stage_matches.empty:
            fixture_labels = [
                (
                    f"{row.date:%b %d} | {row.group} | {row.team1} vs "
                    f"{row.team2} | {row.ground}"
                )
                for row in group_stage_matches.itertuples(index=False)
            ]
            fixture_index = st.selectbox(
                "2026 fixture",
                range(len(fixture_labels)),
                format_func=lambda index: fixture_labels[index],
            )
            fixture = group_stage_matches.iloc[int(fixture_index)]
            team1 = str(fixture["team1"])
            team2 = str(fixture["team2"])
            selected_ground = str(fixture["ground"])
        else:
            team1 = "Argentina"
            team2 = "France"

        selectable_teams = (
            WORLD_CUP_2026_TEAMS
            if prediction_mode == "World Cup fixture"
            else custom_team_options
        )
        team1_col, team2_col = st.columns(2)
        with team1_col:
            team1 = st.selectbox(
                "Team 1",
                selectable_teams,
                index=selectable_teams.index(team1),
                disabled=prediction_mode == "World Cup fixture",
            )
        with team2_col:
            team2 = st.selectbox(
                "Team 2",
                selectable_teams,
                index=selectable_teams.index(team2),
                disabled=prediction_mode == "World Cup fixture",
            )

        with st.expander("Active Roster Micro-Metrics", expanded=False):
            st.caption(ROSTER_METRICS_NOTE)
            t1_val, t1_fat = ROSTER_METRICS.get(team1, (50.0, 0.20))
            t2_val, t2_fat = ROSTER_METRICS.get(team2, (50.0, 0.20))
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"**{team1} Roster Profile**")
                st.caption(f"Squad Market Value: €{t1_val}M")
                st.caption(f"Squad Fatigue Index: {t1_fat:.0%}")
            with col_t2:
                st.markdown(f"**{team2} Roster Profile**")
                st.caption(f"Squad Market Value: €{t2_val}M")
                st.caption(f"Squad Fatigue Index: {t2_fat:.0%}")

        if team1 == team2:
            st.warning("Choose two different teams to predict a meaningful matchup.")
            st.stop()

        prediction_frame = build_prediction_frame(
            team1,
            team2,
            team_states,
            team_lookup,
            ground=selected_ground,
        )

        with st.expander("Current model features", expanded=False):
            st.dataframe(prediction_frame, use_container_width=True, hide_index=True)

        if st.button(
            "Predict Match Outcome",
            type="primary",
            use_container_width=True,
        ):
            probabilities = probability_by_class(model, prediction_frame)
            team1_win = probabilities.get(2, 0.0)
            draw = probabilities.get(1, 0.0)
            team2_win = probabilities.get(0, 0.0)

            predicted_label = max(
                (
                    (team1_win, f"{team1} Win"),
                    (draw, "Draw"),
                    (team2_win, f"{team2} Win"),
                ),
                key=lambda item: item[0],
            )[1]

            outcome_emoji = "🏆" if team1_win >= team2_win and team1_win > draw else ("🤝" if draw > team1_win and draw > team2_win else "🏆")
            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,rgba(251,191,36,0.08) 0%,rgba(59,130,246,0.08) 100%);
                    border:1px solid rgba(255,215,0,0.2);
                    border-radius:16px;padding:16px 20px;margin:16px 0 20px 0;
                    text-align:center;
                ">
                    <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#94a3b8;font-family:Inter,sans-serif;margin-bottom:4px;">Predicted Outcome</div>
                    <div style="font-size:28px;font-weight:900;color:#fbbf24;font-family:Inter,sans-serif;">{outcome_emoji} {predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            result_cols = st.columns(3)
            with result_cols[0]:
                render_probability_card(f"{team1} Win", team1_win)
            with result_cols[1]:
                render_probability_card("Draw", draw)
            with result_cols[2]:
                render_probability_card(f"{team2} Win", team2_win)

            t1_elo = float(prediction_frame.iloc[0]["team1_elo_rating"])
            t2_elo = float(prediction_frame.iloc[0]["team2_elo_rating"])
            neutral_flag = bool(prediction_frame.iloc[0]["is_neutral"])
            render_score_probability_matrix(
                team1,
                team2,
                t1_elo,
                t2_elo,
                neutral_flag,
            )

            st.caption(
                "Classes are modeled from Team 1's perspective: "
                "0 = loss, 1 = draw, 2 = win."
            )

    if app_view == "Tournament Simulation Engine":
        st.markdown(
            "<h2 style='color:#f1f5f9;font-family:Inter,sans-serif;font-weight:800;"
            "font-size:20px;margin-bottom:4px;'>🎲 Tournament Simulation Engine</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#64748b;font-size:13px;margin-top:0;font-family:Inter,sans-serif;'>"
            "Runs the 72-match group phase, advances 24 automatic qualifiers plus the eight best "
            "third-place teams, then resolves a fixed 32-team knockout bracket via Monte Carlo.</p>",
            unsafe_allow_html=True,
        )

        control_col, summary_col = st.columns([1, 2])
        with control_col:
            iterations = st.slider(
                "Monte Carlo iterations",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
            )
            random_seed = st.number_input(
                "Random seed",
                min_value=1,
                max_value=999_999,
                value=2026,
                step=1,
            )
            run_simulation = st.button(
                "Run Monte Carlo Simulation",
                type="primary",
                use_container_width=True,
            )

        with summary_col:
            st.metric("Group-stage fixtures", f"{len(group_stage_matches):,}")
            st.metric("Tournament teams", f"{len(WORLD_CUP_2026_TEAMS):,}")
            if len(group_stage_matches) != 72:
                st.warning(
                    "Expected 72 group-stage fixtures from the schedule source. "
                    f"Loaded {len(group_stage_matches):,}."
                )

        if run_simulation:
            team_state_records = serialize_tournament_team_states(
                team_states,
                team_lookup,
            )
            simulation_results = run_monte_carlo_simulation(
                iterations=iterations,
                group_matches=group_stage_matches,
                team_state_records=team_state_records,
                _model=model,
                random_seed=int(random_seed),
            )
            single_bracket_data = simulate_single_tournament(
                group_matches=group_stage_matches,
                team_state_records=team_state_records,
                _model=model,
                random_seed=int(random_seed),
            )
            most_likely_winner = simulation_results.iloc[0]
            winner_col, final_col, group_col = st.columns(3)
            with winner_col:
                st.metric(
                    "Most likely winner",
                    str(most_likely_winner["team"]),
                    f"{most_likely_winner['won_tournament']:.1f}%",
                )
            with final_col:
                st.metric(
                    "Final chance",
                    f"{most_likely_winner['reached_final']:.1f}%",
                )
            with group_col:
                st.metric(
                    "Group survival",
                    f"{most_likely_winner['reached_r32']:.1f}%",
                )

            render_simulation_leaderboard(simulation_results)
            render_plotly_bracket(single_bracket_data)
        else:
            st.info("Choose the number of iterations, then run the simulation.")


if __name__ == "__main__":
    main()
