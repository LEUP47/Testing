"""Streamlit app for FIFA World Cup 2026 match outcome prediction.

The app downloads public international football results, engineers ELO and
recent-form features, trains a scikit-learn classifier, and predicts match
outcomes from the perspective of the first selected team.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
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

FEATURE_COLUMNS = [
    "team1_elo_rating",
    "team2_elo_rating",
    "elo_difference",
    "team1_form",
    "team2_form",
    "is_neutral",
]

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
    """Read a CSV directly from the first reachable public URL."""

    errors: List[str] = []
    for url in urls:
        try:
            return pd.read_csv(url, parse_dates=parse_dates), url
        except Exception as exc:
            errors.append(f"{url} ({exc})")

    joined_errors = "; ".join(errors)
    raise ValueError(f"Unable to fetch CSV data from public URLs: {joined_errors}")


@st.cache_data(show_spinner="Downloading 2026 World Cup fixture data...")
def load_world_cup_2026_schedule() -> pd.DataFrame:
    """Fetch and deeply unpack the 2026 World Cup fixture list."""

    import requests

    response = requests.get(WORLD_CUP_2026_URL, timeout=30)
    response.raise_for_status()
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


@st.cache_data(show_spinner="Downloading international match data...")
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
    """Resolve a World Cup team to its latest dataset state or fallback rating."""

    normalized_display = normalize_team_name(display_name)
    dataset_name = team_lookup.get(normalized_display)

    if dataset_name and dataset_name in team_states:
        return team_states[dataset_name]

    for alias in TEAM_ALIASES.get(display_name, ()):
        normalized_alias = normalize_team_name(alias)
        dataset_name = team_lookup.get(normalized_alias)
        if dataset_name and dataset_name in team_states:
            return team_states[dataset_name]

    fallback_elo = BASELINE_STRENGTHS.get(display_name, DEFAULT_ELO)
    return TeamState(elo=fallback_elo, form=7.0)


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


def render_probability_card(label: str, probability: float) -> None:
    """Render a compact probability metric."""

    st.metric(label=label, value=f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title="FIFA World Cup 2026 Match Predictor",
        layout="wide",
    )

    st.title("FIFA World Cup 2026 Match Predictor (Live Data Engine)")
    st.caption(
        "Historical international results are fetched from public URLs and "
        "converted into ELO, recent-form, and venue-context features."
    )

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

    st.sidebar.header("Model Snapshot")
    st.sidebar.metric("Historical matches", f"{int(metrics['matches']):,}")
    st.sidebar.metric("Teams tracked", f"{int(metrics['teams']):,}")
    st.sidebar.metric("Validation accuracy", f"{metrics['accuracy']:.1%}")
    st.sidebar.metric("Validation log-loss", f"{metrics['log_loss']:.3f}")
    st.sidebar.caption(f"Results source: {results_source}")
    st.sidebar.caption(f"Fixture source: {WORLD_CUP_2026_URL}")

    display_lookup = build_display_lookup()
    scheduled_matches = world_cup_schedule.copy()
    scheduled_matches["team1_display"] = scheduled_matches["team1"].apply(
        lambda name: resolve_display_name(str(name), display_lookup)
    )
    scheduled_matches["team2_display"] = scheduled_matches["team2"].apply(
        lambda name: resolve_display_name(str(name), display_lookup)
    )
    scheduled_matches = scheduled_matches.dropna(
        subset=["team1_display", "team2_display"]
    ).reset_index(drop=True)

    prediction_mode = st.radio(
        "Prediction mode",
        ("World Cup fixture", "Custom matchup"),
        horizontal=True,
    )

    selected_ground: Optional[str] = None
    if prediction_mode == "World Cup fixture" and not scheduled_matches.empty:
        fixture_labels = [
            (
                f"{row.date:%b %d} | {row.group} | {row.team1_display} vs "
                f"{row.team2_display} | {row.ground}"
            )
            for row in scheduled_matches.itertuples(index=False)
        ]
        fixture_index = st.selectbox(
            "2026 fixture",
            range(len(fixture_labels)),
            format_func=lambda index: fixture_labels[index],
        )
        fixture = scheduled_matches.iloc[int(fixture_index)]
        team1 = str(fixture["team1_display"])
        team2 = str(fixture["team2_display"])
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

    if st.button("Predict Match Outcome", type="primary", use_container_width=True):
        probabilities = probability_by_class(model, prediction_frame)
        team1_win = probabilities.get(2, 0.0)
        draw = probabilities.get(1, 0.0)
        team2_win = probabilities.get(0, 0.0)

        predicted_label = max(
            ((team1_win, f"{team1} Win"), (draw, "Draw"), (team2_win, f"{team2} Win")),
            key=lambda item: item[0],
        )[1]

        st.subheader(f"Prediction: {predicted_label}")
        result_cols = st.columns(3)
        with result_cols[0]:
            render_probability_card(f"{team1} Win", team1_win)
        with result_cols[1]:
            render_probability_card("Draw", draw)
        with result_cols[2]:
            render_probability_card(f"{team2} Win", team2_win)

        st.caption(
            "Classes are modeled from Team 1's perspective: "
            "0 = loss, 1 = draw, 2 = win."
        )


if __name__ == "__main__":
    main()
