from __future__ import annotations

import base64
import warnings
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Tanmay's IPL Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent

ARTIFACT_FILES = {
    "model": "ipl_model.pkl",
    "team_encoder": "team_encoder.pkl",
    "venue_encoder": "venue_encoder.pkl",
    "winner_encoder": "winner_encoder.pkl",
    "venue_win_rates": "venue_win_rates.pkl",
    "head_to_head": "head_to_head.pkl",
}

TEAM_COLORS = {
    "CSK": "#f5c542",
    "DC": "#3b82f6",
    "GT": "#38bdf8",
    "KKR": "#a855f7",
    "LSG": "#22d3ee",
    "MI": "#2563eb",
    "PBKS": "#ef4444",
    "RCB": "#fb7185",
    "RR": "#f472b6",
    "SRH": "#fb923c",
}

TEAM_LABELS = {
    "CSK": "Chennai Super Kings",
    "DC": "Delhi Capitals",
    "GT": "Gujarat Titans",
    "KKR": "Kolkata Knight Riders",
    "LSG": "Lucknow Super Giants",
    "MI": "Mumbai Indians",
    "PBKS": "Punjab Kings",
    "RCB": "Royal Challengers Bengaluru",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
}

AI_QUOTE = "Where cricket intelligence meets machine learning."
DISCLAIMER = "Predictions are data-driven, but cricket always has surprises."


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for key, filename in ARTIFACT_FILES.items():
        path = APP_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {filename}")
        artifacts[key] = joblib.load(path)
    return artifacts


@st.cache_data(show_spinner=False)
def load_image_base64(filename: str) -> str:
    image_path = APP_DIR / filename
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #07111f;
            --bg-card: rgba(10, 18, 33, 0.70);
            --bg-card-strong: rgba(9, 16, 28, 0.90);
            --line-soft: rgba(255, 255, 255, 0.12);
            --text-main: #f8fbff;
            --text-soft: #a9b7cc;
            --cyan: #22d3ee;
            --blue: #60a5fa;
            --rose: #fb7185;
            --amber: #f59e0b;
            --shadow-main: 0 24px 60px rgba(0, 0, 0, 0.38);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 211, 238, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(251, 113, 133, 0.18), transparent 24%),
                linear-gradient(180deg, #08111f 0%, #06101d 42%, #02050d 100%);
            color: var(--text-main);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(9, 18, 33, 0.96) 0%, rgba(6, 12, 23, 0.98) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }

        .sidebar-card,
        .hero-card,
        .section-card,
        .result-card,
        .info-card,
        .about-card {
            background: var(--bg-card);
            border: 1px solid var(--line-soft);
            box-shadow: var(--shadow-main);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-radius: 26px;
        }

        .hero-card {
            padding: 1.3rem;
            overflow: hidden;
            position: relative;
        }

        .hero-card::before,
        .result-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(135deg, rgba(34, 211, 238, 0.12), transparent 38%),
                linear-gradient(315deg, rgba(251, 113, 133, 0.14), transparent 35%);
            pointer-events: none;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.9fr);
            gap: 1.25rem;
            align-items: center;
        }

        .hero-copy {
            position: relative;
            z-index: 1;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.9rem;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.10);
            color: #d8f6fb;
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .hero-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .hero-logo {
            width: 92px;
            height: 92px;
            object-fit: contain;
            padding: 0.7rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.2rem, 4vw, 4rem);
            line-height: 1.02;
            letter-spacing: -0.03em;
            color: var(--text-main);
        }

        .hero-quote {
            margin: 0.2rem 0 0.9rem;
            font-size: 1.18rem;
            color: #dff8ff;
            font-weight: 600;
        }

        .hero-support {
            margin: 0;
            max-width: 54ch;
            color: var(--text-soft);
            line-height: 1.7;
            font-size: 1rem;
        }

        .hero-disclaimer {
            margin-top: 1rem;
            color: #ffe3b8;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .hero-media {
            position: relative;
            z-index: 1;
        }

        .hero-gif {
            width: 100%;
            max-height: 340px;
            object-fit: cover;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.36);
        }

        .section-heading {
            margin: 0.4rem 0 1rem;
        }

        .section-heading h2 {
            margin: 0;
            font-size: 1.55rem;
            color: var(--text-main);
        }

        .section-heading p {
            margin: 0.3rem 0 0;
            color: var(--text-soft);
            line-height: 1.6;
        }

        .mini-kicker {
            display: inline-block;
            margin-bottom: 0.45rem;
            color: #7dd3fc;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .result-card {
            padding: 1.4rem;
            position: relative;
            overflow: hidden;
        }

        .result-label {
            color: #dff8ff;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .result-winner {
            margin: 0.5rem 0 0.2rem;
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.02;
            letter-spacing: -0.03em;
            color: var(--text-main);
        }

        .result-subcopy {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.6;
        }

        .progress-track {
            margin-top: 1rem;
            height: 14px;
            width: 100%;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
            box-shadow: 0 0 24px rgba(34, 211, 238, 0.35);
        }

        .result-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .result-stat {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .result-stat span {
            display: block;
            color: var(--text-soft);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-stat strong {
            display: block;
            margin-top: 0.3rem;
            font-size: 1.05rem;
            color: var(--text-main);
        }

        .probability-row {
            display: flex;
            gap: 0.9rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .probability-chip {
            flex: 1 1 220px;
            min-width: 0;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .chip-label {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--text-soft);
            font-size: 0.9rem;
        }

        .chip-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .chip-value {
            margin-top: 0.35rem;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .info-card,
        .about-card,
        .sidebar-card {
            padding: 1.15rem;
        }

        .info-card h3,
        .about-card h3,
        .sidebar-card h3 {
            margin: 0;
            color: var(--text-main);
            font-size: 1.05rem;
        }

        .info-card p,
        .about-card p,
        .sidebar-card p,
        .sidebar-card li {
            color: var(--text-soft);
            line-height: 1.65;
        }

        .sidebar-card ul {
            padding-left: 1rem;
            margin: 0.55rem 0 0;
        }

        .creator-pill {
            margin-top: 0.9rem;
            display: inline-flex;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.18), rgba(251, 113, 133, 0.18));
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: var(--text-main);
            font-weight: 700;
        }

        [data-baseweb="tab-list"] {
            gap: 0.65rem;
            margin-top: 0.7rem;
            margin-bottom: 1.2rem;
            background: transparent;
        }

        button[data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.7rem 1rem;
            color: var(--text-soft);
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.22), rgba(96, 165, 250, 0.18));
            color: var(--text-main);
            border-color: rgba(34, 211, 238, 0.35);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(255, 255, 255, 0.06) !important;
            border-radius: 18px !important;
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            min-height: 3.2rem;
            box-shadow: none !important;
        }

        div[data-baseweb="select"] * {
            color: var(--text-main) !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #dff8ff !important;
            font-weight: 600;
        }

        [data-testid="stFormSubmitButton"] button,
        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 18px;
            padding: 0.9rem 1.2rem;
            background: linear-gradient(90deg, #22d3ee 0%, #3b82f6 55%, #fb7185 100%);
            color: #04101d;
            font-weight: 800;
            letter-spacing: 0.02em;
            box-shadow: 0 18px 42px rgba(34, 211, 238, 0.22);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        [data-testid="stFormSubmitButton"] button:hover,
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 46px rgba(59, 130, 246, 0.26);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: var(--shadow-main);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-soft);
        }

        [data-testid="stMetricValue"] {
            color: var(--text-main);
        }

        [data-testid="stProgressBar"] > div > div {
            background: rgba(255, 255, 255, 0.08);
        }

        [data-testid="stProgressBar"] div[role="progressbar"] {
            background: linear-gradient(90deg, #22d3ee 0%, #3b82f6 55%, #fb7185 100%);
        }

        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            box-shadow: var(--shadow-main);
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            padding: 0.1rem 0.35rem;
        }

        [data-testid="stExpander"] summary p {
            color: var(--text-main) !important;
            font-weight: 700;
        }

        .analytics-banner {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-bottom: 1rem;
        }

        .analytics-pill {
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-main);
            font-weight: 600;
        }

        .about-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }

        .signal-stack {
            display: grid;
            gap: 0.75rem;
        }

        .signal-item {
            padding: 0.85rem 0.95rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .signal-item span {
            display: block;
            color: var(--text-soft);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .signal-item strong {
            display: block;
            margin-top: 0.3rem;
            color: var(--text-main);
            font-size: 0.98rem;
            line-height: 1.45;
        }

        @media (max-width: 980px) {
            .hero-grid,
            .about-grid,
            .result-stats {
                grid-template-columns: 1fr;
            }

            .hero-brand {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_team_name(team_code: str) -> str:
    return TEAM_LABELS.get(team_code, team_code)


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def get_team_color(team_code: str) -> str:
    return TEAM_COLORS.get(team_code, "#22d3ee")


def get_team_venue_rate(
    venue_win_rates: dict[str, dict[str, float]],
    team: str,
    venue: str,
) -> float | None:
    rate = venue_win_rates.get(team, {}).get(venue)
    if rate is None:
        return None
    return float(rate)


def get_head_to_head_record(
    head_to_head: dict[str, dict[str, dict[str, float]]],
    team_one: str,
    team_two: str,
) -> dict[str, float | int] | None:
    matchup = head_to_head.get(team_one, {}).get(team_two)
    if matchup:
        matches = int(matchup.get("matches", 0))
        team_one_wins = int(matchup.get("wins", 0))
        team_two_wins = max(matches - team_one_wins, 0)
        team_one_win_rate = float(matchup.get("win_rate", 0.0))
        team_two_win_rate = 1 - team_one_win_rate if matches else 0.0

        return {
            "matches": matches,
            "team_one_wins": team_one_wins,
            "team_two_wins": team_two_wins,
            "team_one_win_rate": team_one_win_rate,
            "team_two_win_rate": team_two_win_rate,
        }

    reverse_matchup = head_to_head.get(team_two, {}).get(team_one)
    if not reverse_matchup:
        return None

    matches = int(reverse_matchup.get("matches", 0))
    team_two_wins = int(reverse_matchup.get("wins", 0))
    team_one_wins = max(matches - team_two_wins, 0)
    team_two_win_rate = float(reverse_matchup.get("win_rate", 0.0))
    team_one_win_rate = 1 - team_two_win_rate if matches else 0.0

    return {
        "matches": matches,
        "team_one_wins": team_one_wins,
        "team_two_wins": team_two_wins,
        "team_one_win_rate": team_one_win_rate,
        "team_two_win_rate": team_two_win_rate,
    }


def get_venue_edge_text(team_one_rate: float | None, team_two_rate: float | None) -> str:
    if team_one_rate is None and team_two_rate is None:
        return "N/A"
    if team_one_rate is None or team_two_rate is None:
        return "Limited data"
    return f"{abs(team_one_rate - team_two_rate) * 100:.1f} pts"


def describe_confidence_band(confidence: float) -> str:
    if confidence >= 70:
        return "Strong lean"
    if confidence >= 58:
        return "Moderate lean"
    return "Close contest"


def build_venue_reasoning(
    team_one: str,
    team_two: str,
    venue: str,
    team_one_rate: float | None,
    team_two_rate: float | None,
) -> str:
    if team_one_rate is None and team_two_rate is None:
        return (
            f"Venue history is limited for both teams at {venue}, so the model leans "
            "more on learned matchup patterns from the training data."
        )

    if team_one_rate is None:
        return (
            f"{team_two} carries the clearer venue signal at {venue} with a tracked win "
            f"rate of {format_pct(team_two_rate)}, while {team_one} has limited venue data here."
        )

    if team_two_rate is None:
        return (
            f"{team_one} carries the clearer venue signal at {venue} with a tracked win "
            f"rate of {format_pct(team_one_rate)}, while {team_two} has limited venue data here."
        )

    if abs(team_one_rate - team_two_rate) < 0.001:
        return (
            f"Both teams bring an almost identical venue profile at {venue}, each sitting near "
            f"{format_pct(team_one_rate)} in the tracked dataset."
        )

    better_team = team_one if team_one_rate > team_two_rate else team_two
    better_rate = team_one_rate if better_team == team_one else team_two_rate
    other_team = team_two if better_team == team_one else team_one
    other_rate = team_two_rate if better_team == team_one else team_one_rate

    return (
        f"{better_team} owns the stronger venue record at {venue}, winning {format_pct(better_rate)} "
        f"of tracked matches there compared with {format_pct(other_rate)} for {other_team}."
    )


def build_head_to_head_reasoning(
    team_one: str,
    team_two: str,
    matchup: dict[str, float | int] | None,
) -> str:
    if not matchup or not matchup["matches"]:
        return (
            f"No strong rivalry sample is available for {team_one} vs {team_two} in the current "
            "analytics layer, so head-to-head history is not a major separator."
        )

    if matchup["team_one_wins"] == matchup["team_two_wins"]:
        return (
            f"This rivalry is perfectly balanced in the tracked sample, with {team_one} and {team_two} "
            f"sharing {matchup['team_one_wins']} wins each across {matchup['matches']} matches."
        )

    leader = team_one if matchup["team_one_wins"] > matchup["team_two_wins"] else team_two
    leader_wins = matchup["team_one_wins"] if leader == team_one else matchup["team_two_wins"]
    win_rate = matchup["team_one_win_rate"] if leader == team_one else matchup["team_two_win_rate"]
    trailer = team_two if leader == team_one else team_one

    return (
        f"The rivalry edge favors {leader}, who has won {leader_wins} of "
        f"{matchup['matches']} tracked meetings against {trailer} ({format_pct(win_rate)})."
    )


def build_ai_explanation(
    predicted_winner: str,
    team_one: str,
    team_two: str,
    venue: str,
    confidence: float,
    team_one_rate: float | None,
    team_two_rate: float | None,
    matchup: dict[str, float | int] | None,
) -> str:
    opponent = team_two if predicted_winner == team_one else team_one
    reasons: list[str] = []

    predicted_rate = team_one_rate if predicted_winner == team_one else team_two_rate
    opponent_rate = team_two_rate if predicted_winner == team_one else team_one_rate

    if predicted_rate is not None and opponent_rate is not None:
        if predicted_rate > opponent_rate:
            reasons.append(
                f"{predicted_winner} has the stronger venue footprint at {venue}, which adds environmental support to the prediction."
            )
        elif predicted_rate < opponent_rate:
            reasons.append(
                f"{opponent} owns the better venue record at {venue}, so the matchup remains more competitive than the headline pick suggests."
            )

    if matchup and matchup["matches"]:
        predicted_win_rate = (
            matchup["team_one_win_rate"] if predicted_winner == team_one else matchup["team_two_win_rate"]
        )
        predicted_wins = matchup["team_one_wins"] if predicted_winner == team_one else matchup["team_two_wins"]

        if predicted_win_rate > 0.5:
            reasons.append(
                f"The rivalry sample also leans toward {predicted_winner}, who has taken {predicted_wins} of {matchup['matches']} tracked meetings."
            )
        elif predicted_win_rate < 0.5:
            reasons.append(
                f"The head-to-head record actually favors {opponent}, which is the main caution flag inside this prediction."
            )
        else:
            reasons.append("Head-to-head history is dead even, so the final edge comes from the model's learned matchup patterns.")

    if confidence >= 70:
        confidence_line = f"The model sees this as a strong edge for {predicted_winner}."
    elif confidence >= 58:
        confidence_line = f"The model gives {predicted_winner} a healthy but not runaway advantage."
    else:
        confidence_line = "The margin is relatively tight, so a few key moments could flip the result."

    reasons.append(confidence_line)
    return " ".join(reasons)


def build_quick_take(result: dict[str, object]) -> str:
    winner = result["predicted_winner"]
    team_one = result["team_one"]
    team_two = result["team_two"]
    confidence = float(result["confidence"])
    opponent = team_two if winner == team_one else team_one

    venue_support = None
    if result["team_one_rate"] is not None and result["team_two_rate"] is not None:
        if result["team_one_rate"] > result["team_two_rate"]:
            venue_support = team_one
        elif result["team_two_rate"] > result["team_one_rate"]:
            venue_support = team_two

    matchup = result["matchup"]
    h2h_support = None
    if matchup and matchup["matches"]:
        if matchup["team_one_wins"] > matchup["team_two_wins"]:
            h2h_support = team_one
        elif matchup["team_two_wins"] > matchup["team_one_wins"]:
            h2h_support = team_two

    if venue_support == winner and h2h_support == winner:
        base = f"{winner} gets the edge with support from both venue form and head-to-head history."
    elif venue_support == winner:
        base = f"{winner} looks slightly better mainly because of the venue record."
    elif h2h_support == winner:
        base = f"{winner} looks slightly better mainly because of the rivalry history."
    elif venue_support == opponent or h2h_support == opponent:
        base = f"The historical signals are mixed, but the model still gives {winner} a narrow edge."
    else:
        base = f"This looks like a balanced matchup, with {winner} holding a small model advantage."

    return f"{base} {describe_confidence_band(confidence)} at {confidence:.1f}% confidence."


def build_signal_snapshot(result: dict[str, object]) -> dict[str, str]:
    matchup = result["matchup"]

    if result["team_one_rate"] is None and result["team_two_rate"] is None:
        venue_text = "Limited venue history for both teams"
    elif result["team_one_rate"] is None:
        venue_text = f"{result['team_two']} has the clearer venue record"
    elif result["team_two_rate"] is None:
        venue_text = f"{result['team_one']} has the clearer venue record"
    else:
        venue_text = (
            f"{result['team_one']}: {format_pct(result['team_one_rate'])} | "
            f"{result['team_two']}: {format_pct(result['team_two_rate'])}"
        )

    if matchup and matchup["matches"]:
        rivalry_text = (
            f"{result['team_one']} {matchup['team_one_wins']} - "
            f"{matchup['team_two_wins']} {result['team_two']} in {matchup['matches']} matches"
        )
    else:
        rivalry_text = "No rivalry sample available"

    confidence_text = f"{describe_confidence_band(float(result['confidence']))} for {result['predicted_winner']}"

    return {
        "venue": venue_text,
        "rivalry": rivalry_text,
        "confidence": confidence_text,
    }


def predict_match_winner(
    team_one: str,
    team_two: str,
    venue: str,
    artifacts: dict[str, object],
) -> dict[str, object]:
    team_encoder = artifacts["team_encoder"]
    venue_encoder = artifacts["venue_encoder"]
    winner_encoder = artifacts["winner_encoder"]
    model = artifacts["model"]
    venue_win_rates = artifacts["venue_win_rates"]
    head_to_head = artifacts["head_to_head"]

    feature_frame = pd.DataFrame(
        [
            {
                "team1": int(team_encoder.transform([team_one])[0]),
                "team2": int(team_encoder.transform([team_two])[0]),
                "venue": int(venue_encoder.transform([venue])[0]),
            }
        ]
    )

    probability_vector = model.predict_proba(feature_frame)[0]

    team_one_index = int(winner_encoder.transform([team_one])[0])
    team_two_index = int(winner_encoder.transform([team_two])[0])

    team_one_raw = float(probability_vector[team_one_index])
    team_two_raw = float(probability_vector[team_two_index])
    pairwise_total = team_one_raw + team_two_raw

    if pairwise_total <= 0:
        team_one_probability = 0.5
        team_two_probability = 0.5
    else:
        team_one_probability = team_one_raw / pairwise_total
        team_two_probability = team_two_raw / pairwise_total

    predicted_winner = team_one if team_one_probability >= team_two_probability else team_two
    confidence = max(team_one_probability, team_two_probability) * 100

    team_one_rate = get_team_venue_rate(venue_win_rates, team_one, venue)
    team_two_rate = get_team_venue_rate(venue_win_rates, team_two, venue)
    matchup = get_head_to_head_record(head_to_head, team_one, team_two)

    return {
        "team_one": team_one,
        "team_two": team_two,
        "venue": venue,
        "predicted_winner": predicted_winner,
        "confidence": confidence,
        "team_one_probability": team_one_probability * 100,
        "team_two_probability": team_two_probability * 100,
        "team_one_rate": team_one_rate,
        "team_two_rate": team_two_rate,
        "matchup": matchup,
        "venue_reasoning": build_venue_reasoning(team_one, team_two, venue, team_one_rate, team_two_rate),
        "head_to_head_reasoning": build_head_to_head_reasoning(team_one, team_two, matchup),
        "ai_explanation": build_ai_explanation(
            predicted_winner,
            team_one,
            team_two,
            venue,
            confidence,
            team_one_rate,
            team_two_rate,
            matchup,
        ),
    }


def build_probability_chart(result: dict[str, object]) -> alt.Chart:
    team_one = result["team_one"]
    team_two = result["team_two"]

    data = pd.DataFrame(
        {
            "Team": [team_one, team_two],
            "Probability": [result["team_one_probability"], result["team_two_probability"]],
            "Color": [get_team_color(team_one), get_team_color(team_two)],
        }
    )

    return (
        alt.Chart(data)
        .mark_bar(size=60, cornerRadiusTopLeft=16, cornerRadiusTopRight=16)
        .encode(
            x=alt.X("Team:N", sort=[team_one, team_two], axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="Win Probability (%)", gridColor="rgba(255,255,255,0.12)"),
            ),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Team:N"),
                alt.Tooltip("Probability:Q", format=".1f", title="Win Probability"),
            ],
        )
        .properties(height=280)
        .configure(background="transparent")
        .configure_axis(
            labelColor="#dce7f7",
            titleColor="#dce7f7",
            domainColor="rgba(255,255,255,0.10)",
            tickColor="rgba(255,255,255,0.10)",
        )
        .configure_view(strokeOpacity=0)
    )


def build_venue_chart(result: dict[str, object]) -> alt.Chart:
    team_one = result["team_one"]
    team_two = result["team_two"]

    team_one_rate = 0 if result["team_one_rate"] is None else result["team_one_rate"] * 100
    team_two_rate = 0 if result["team_two_rate"] is None else result["team_two_rate"] * 100

    data = pd.DataFrame(
        {
            "Team": [team_one, team_two],
            "Venue Win Rate": [team_one_rate, team_two_rate],
            "Color": [get_team_color(team_one), get_team_color(team_two)],
        }
    )

    return (
        alt.Chart(data)
        .mark_bar(size=60, cornerRadiusTopLeft=16, cornerRadiusTopRight=16)
        .encode(
            x=alt.X("Team:N", sort=[team_one, team_two], axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y(
                "Venue Win Rate:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="Venue Win Rate (%)", gridColor="rgba(255,255,255,0.12)"),
            ),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Team:N"),
                alt.Tooltip("Venue Win Rate:Q", format=".1f"),
            ],
        )
        .properties(height=280)
        .configure(background="transparent")
        .configure_axis(
            labelColor="#dce7f7",
            titleColor="#dce7f7",
            domainColor="rgba(255,255,255,0.10)",
            tickColor="rgba(255,255,255,0.10)",
        )
        .configure_view(strokeOpacity=0)
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.image(str(APP_DIR / "ipl-all-team-logo.jpg"), use_container_width=True)

        st.markdown(
            """
            <div class="sidebar-card">
                <h3>Project Snapshot</h3>
                <p>
                    A premium IPL analytics dashboard that blends machine learning predictions
                    with venue intelligence and rivalry context for quick, explainable match forecasts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <h3>Technologies Used</h3>
                <ul>
                    <li>Streamlit for the interactive web experience</li>
                    <li>XGBoost for match outcome prediction</li>
                    <li>Scikit-learn encoders and preprocessing</li>
                    <li>Pandas for feature shaping and analytics</li>
                    <li>Joblib for local model artifact loading</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <h3>Model Details</h3>
                <ul>
                    <li>Model: XGBoost multiclass classifier</li>
                    <li>Primary inputs: Team 1, Team 2, Venue</li>
                    <li>Context layer: venue win rates and head-to-head analytics</li>
                    <li>Dataset window: 2022-2026</li>
                </ul>
                <div class="creator-pill">Built by Tanmay</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    logo_base64 = load_image_base64("ipl-logo.jpg")
    gif_base64 = load_image_base64("Virat-kohli-signing.gif")

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="hero-badge">AI-Powered IPL Intelligence</div>
                    <div class="hero-brand">
                        <img class="hero-logo" src="data:image/jpeg;base64,{logo_base64}" alt="IPL Logo" />
                        <div>
                            <h1 class="hero-title">Tanmay's IPL Predictor</h1>
                            <p class="hero-quote">{AI_QUOTE}</p>
                        </div>
                    </div>
                    <p class="hero-disclaimer">{DISCLAIMER}</p>
                </div>
                <div class="hero-media">
                    <img class="hero-gif" src="data:image/gif;base64,{gif_base64}" alt="Virat Kohli GIF" />
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_tab(
    teams: list[str],
    venues: list[str],
    artifacts: dict[str, object],
) -> None:
    if "team_one" not in st.session_state:
        st.session_state.team_one = teams[0]
    if "team_two" not in st.session_state:
        st.session_state.team_two = teams[1]
    if "venue_choice" not in st.session_state:
        st.session_state.venue_choice = venues[0]
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    st.markdown(
        """
        <div class="section-heading">
            <div class="mini-kicker">Prediction</div>
            <h2>Matchday Control Room</h2>
            <p>Choose the teams, lock in the venue, and generate an AI-backed winner prediction with explainable analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.selectbox("Team 1", teams, key="team_one")
        with col2:
            st.selectbox("Team 2", teams, key="team_two")
        with col3:
            st.selectbox("Venue", venues, key="venue_choice")

        submitted = st.form_submit_button("Run AI Prediction")

    if submitted:
        if st.session_state.team_one == st.session_state.team_two:
            st.session_state.prediction_result = None
            st.warning("Please choose two different teams for a valid IPL matchup.")
        else:
            st.session_state.prediction_result = predict_match_winner(
                st.session_state.team_one,
                st.session_state.team_two,
                st.session_state.venue_choice,
                artifacts,
            )

    result = st.session_state.prediction_result

    if not result:
        st.markdown(
            """
            <div class="info-card">
                <h3>Ready for kickoff</h3>
                <p>
                    Set the matchup above to unlock the winner card, confidence score,
                    venue context, and the AI explanation layer.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    winner = result["predicted_winner"]
    winner_label = format_team_name(winner)
    confidence = result["confidence"]
    winner_color = get_team_color(winner)
    signals = build_signal_snapshot(result)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">🏆 AI Match Forecast</div>
            <h2 class="result-winner">{winner_label} ({winner})</h2>
            <p class="result-subcopy">
                The model currently backs <strong>{winner}</strong> to win this matchup at
                <strong>{result['venue']}</strong> with a confidence score of <strong>{confidence:.1f}%</strong>.
            </p>
            <div class="progress-track">
                <div class="progress-fill" style="width: {confidence:.1f}%; background: linear-gradient(90deg, {winner_color}, #22d3ee);"></div>
            </div>
            <div class="result-stats">
                <div class="result-stat">
                    <span>Venue</span>
                    <strong>{result['venue']}</strong>
                </div>
                <div class="result-stat">
                    <span>Confidence</span>
                    <strong>{confidence:.1f}%</strong>
                </div>
            </div>
            <div class="probability-row">
                <div class="probability-chip">
                    <div class="chip-label">
                        <span class="chip-dot" style="background:{get_team_color(result['team_one'])};"></span>
                        {result['team_one']}
                    </div>
                    <div class="chip-value">{result['team_one_probability']:.1f}%</div>
                </div>
                <div class="probability-chip">
                    <div class="chip-label">
                        <span class="chip-dot" style="background:{get_team_color(result['team_two'])};"></span>
                        {result['team_two']}
                    </div>
                    <div class="chip-value">{result['team_two_probability']:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-heading">
            <div class="mini-kicker">Match Summary</div>
            <h2>Why this pick</h2>
            <p>A quick read first, with full reasoning tucked away below if you want the details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2 = st.columns([1.15, 0.95], gap="large")
    with summary_col1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Quick Take</h3>
                <p>{build_quick_take(result)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_col2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Key Signals</h3>
                <div class="signal-stack">
                    <div class="signal-item">
                        <span>Confidence</span>
                        <strong>{signals['confidence']}</strong>
                    </div>
                    <div class="signal-item">
                        <span>Venue</span>
                        <strong>{signals['venue']}</strong>
                    </div>
                    <div class="signal-item">
                        <span>Rivalry</span>
                        <strong>{signals['rivalry']}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("See detailed reasoning"):
        detail_col1, detail_col2 = st.columns(2, gap="large")
        with detail_col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>Venue Reasoning</h3>
                    <p>{result['venue_reasoning']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with detail_col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>Head-to-Head Reasoning</h3>
                    <p>{result['head_to_head_reasoning']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="info-card">
                <h3>Full AI Explanation</h3>
                <p>{result['ai_explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analytics_tab(
    teams: list[str],
    venues: list[str],
    artifacts: dict[str, object],
) -> None:
    team_one = st.session_state.get("team_one", teams[0])
    team_two = st.session_state.get("team_two", teams[1])
    venue = st.session_state.get("venue_choice", venues[0])

    st.markdown(
        """
        <div class="section-heading">
            <div class="mini-kicker">Analytics</div>
            <h2>Team Comparison Dashboard</h2>
            <p>Use the current selection to compare win probabilities, venue comfort, and rivalry strength in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if team_one == team_two:
        st.markdown(
            """
            <div class="info-card">
                <h3>Matchup needed</h3>
                <p>Select two different teams in the Prediction tab to unlock the analytics comparison view.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = predict_match_winner(team_one, team_two, venue, artifacts)

    st.markdown(
        f"""
        <div class="analytics-banner">
            <div class="analytics-pill">Matchup: {team_one} vs {team_two}</div>
            <div class="analytics-pill">Venue: {venue}</div>
            <div class="analytics-pill">Model Favorite: {result['predicted_winner']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.markdown(
            """
            <div class="section-heading">
                <h2 style="font-size:1.15rem;">Match Win Probability</h2>
                <p>Pairwise probabilities are normalized between the two selected teams for a clean head-to-head forecast.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(build_probability_chart(result), use_container_width=True)

    with chart_col2:
        st.markdown(
            """
            <div class="section-heading">
                <h2 style="font-size:1.15rem;">Venue Win Rates</h2>
                <p>This compares the historical venue success of both teams at the currently selected ground.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(build_venue_chart(result), use_container_width=True)

    matchup = result["matchup"]

    metric_col1, metric_col2, metric_col3 = st.columns(3, gap="large")
    with metric_col1:
        st.metric(
            f"{team_one} venue win rate",
            format_pct(result["team_one_rate"]),
        )
    with metric_col2:
        st.metric(
            f"{team_two} venue win rate",
            format_pct(result["team_two_rate"]),
        )
    with metric_col3:
        st.metric(
            "Head-to-head sample",
            f"{matchup['matches']} matches" if matchup else "N/A",
        )

    insight_col1, insight_col2 = st.columns(2, gap="large")
    with insight_col1:
        rivalry_text = (
            f"{team_one} has {matchup['team_one_wins']} wins and {team_two} has {matchup['team_two_wins']} wins in {matchup['matches']} tracked meetings."
            if matchup
            else "No rivalry sample is currently available for this matchup."
        )
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Rivalry Snapshot</h3>
                <p>{rivalry_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with insight_col2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Analyst Note</h3>
                <p>{result['ai_explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_tab() -> None:
    st.markdown(
        """
        <div class="section-heading">
            <div class="mini-kicker">About Model</div>
            <h2>How this IPL intelligence stack works</h2>
            <p>The app keeps the existing local ML pipeline intact while upgrading the presentation into a cleaner, showcase-ready analytics experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-grid">
            <div class="about-card">
                <h3>Prediction Engine</h3>
                <p>
                    An XGBoost classifier predicts the likely winner using encoded team and venue inputs.
                    The interface then converts the multiclass output into a clear two-team matchup forecast.
                </p>
            </div>
            <div class="about-card">
                <h3>Explainability Layer</h3>
                <p>
                    Historical venue win rates and head-to-head rivalry records are surfaced alongside the prediction
                    so users can understand why the model leans in a specific direction.
                </p>
            </div>
            <div class="about-card">
                <h3>Project Inputs</h3>
                <p>
                    Team 1, Team 2, and Venue drive the ML prediction, while the venue and rivalry datasets power the narrative analytics experience.
                </p>
            </div>
            <div class="about-card">
                <h3>Dataset Window</h3>
                <p>
                    The current project framing uses IPL data from 2022-2026 and loads every trained artifact directly from the local project folder with Joblib.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_col1, model_col2 = st.columns([1.4, 1], gap="large")
    with model_col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Production-style structure</h3>
                <p>
                    This app is organized around cached artifact loading, modular helper functions,
                    UI rendering sections, and prediction utilities so it stays readable and easy to extend.
                </p>
                <p>
                    That means you can keep iterating on features such as richer charts, team assets,
                    or additional match context without touching the core artifact loading flow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with model_col2:
        st.image(str(APP_DIR / "ipl-all-team-logo.jpg"), use_container_width=True)


def main() -> None:
    inject_custom_css()
    render_sidebar()

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:  # pragma: no cover - UI fallback
        st.error(f"Unable to load model artifacts: {error}")
        st.stop()

    teams = list(artifacts["team_encoder"].classes_)
    venues = list(artifacts["venue_encoder"].classes_)

    render_hero()

    prediction_tab, analytics_tab, about_tab = st.tabs(
        ["Prediction", "Team Analytics", "About Model"]
    )

    with prediction_tab:
        render_prediction_tab(teams, venues, artifacts)

    with analytics_tab:
        render_analytics_tab(teams, venues, artifacts)

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
