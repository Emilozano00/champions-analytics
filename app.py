"""
European Football Predictor
Streamlit app for Champions League, Europa League & Conference League prediction.
Three independent models: CL 57.2%, EL 59.4%, ECL 52.7%.
"""

import json
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="European Football Predictor",
    page_icon="⚽",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Competition registry
COMPETITIONS = {
    "Champions League": {
        "icon": "🏆",
        "model_pkl": "champion_model_final.pkl",
        "feature_json": "feature_list_final.json",
        "metrics_json": "metrics_final.json",
        "comp_filter": "is_champions",
        "league_id": 2,
        "color": "#d4af37",
        "algorithm": "LogReg calibrada",
        "matches": 709,
    },
    "Europa League": {
        "icon": "🥈",
        "model_pkl": "europa_model.pkl",
        "feature_json": "europa_feature_list.json",
        "metrics_json": "europa_metrics.json",
        "comp_filter": "is_europa",
        "league_id": 3,
        "color": "#F57C00",
        "algorithm": "Random Forest",
        "matches": 650,
    },
    "Conference League": {
        "icon": "🥉",
        "model_pkl": "conference_model.pkl",
        "feature_json": "conference_feature_list.json",
        "metrics_json": "conference_metrics.json",
        "comp_filter": "is_conference",
        "league_id": 848,
        "color": "#66BB6A",
        "algorithm": "Random Forest",
        "matches": 1112,
    },
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_competition_model(comp_name):
    cfg = COMPETITIONS[comp_name]
    model = joblib.load(MODELS_DIR / cfg["model_pkl"])
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / cfg["feature_json"]) as f:
        feature_list = json.load(f)
    return model, le, feature_list


@st.cache_data
def load_competition_metrics(comp_name):
    cfg = COMPETITIONS[comp_name]
    with open(MODELS_DIR / cfg["metrics_json"]) as f:
        return json.load(f)


@st.cache_data
def load_all_features():
    return pd.read_csv(PROCESSED_DIR / "features_all.csv", low_memory=False)


@st.cache_data
def load_cl_features():
    """Load CL-only features for backward compat (calibration chart uses metrics.json)."""
    return pd.read_csv(PROCESSED_DIR / "features_v3.csv")


@st.cache_data
def load_matches():
    return pd.read_csv(PROCESSED_DIR / "matches.csv")


@st.cache_data
def load_original_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_upcoming_matches():
    """Load upcoming fixtures from 11_update_fixtures.py output (all competitions)."""
    path = PROCESSED_DIR / "upcoming_matches.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
        return df
    # Fallback to old format
    path_old = PROCESSED_DIR / "upcoming.csv"
    if path_old.exists():
        df = pd.read_csv(path_old)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df["league_name"] = "Champions League"
            df["status"] = "NS"
        return df
    return pd.DataFrame()


@st.cache_data
def load_recent_results():
    """Load recent results from 11_update_fixtures.py output (all competitions)."""
    path = PROCESSED_DIR / "recent_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_odds():
    path = PROCESSED_DIR / "odds.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    odds = {}
    for _, row in df.iterrows():
        odds[int(row["fixture_id"])] = {
            "Home": row["odd_home"],
            "Draw": row["odd_draw"],
            "Away": row["odd_away"],
        }
    return odds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_team_form(features_df, team_name, n=5):
    home_matches = features_df[features_df["home_team_name"] == team_name].copy()
    home_matches["team_result"] = home_matches["result"].map({"H": "W", "D": "D", "A": "L"})

    away_matches = features_df[features_df["away_team_name"] == team_name].copy()
    away_matches["team_result"] = away_matches["result"].map({"H": "L", "D": "D", "A": "W"})

    all_matches = pd.concat([
        home_matches[["date", "team_result", "home_team_name", "away_team_name", "home_goals", "away_goals", "result"]],
        away_matches[["date", "team_result", "home_team_name", "away_team_name", "home_goals", "away_goals", "result"]],
    ]).sort_values("date", ascending=False)

    return all_matches.head(n)


def get_latest_rolling_features(features_df, team_name, feature_list):
    home = features_df[features_df["home_team_name"] == team_name].sort_values("date")
    away = features_df[features_df["away_team_name"] == team_name].sort_values("date")

    home_latest = home.iloc[-1] if len(home) > 0 else None
    away_latest = away.iloc[-1] if len(away) > 0 else None

    if home_latest is not None and away_latest is not None:
        if home_latest["date"] >= away_latest["date"]:
            latest = home_latest
            prefix = "home_"
        else:
            latest = away_latest
            prefix = "away_"
    elif home_latest is not None:
        latest = home_latest
        prefix = "home_"
    else:
        latest = away_latest
        prefix = "away_"

    rolling_keys = [
        "rolling_xg_for", "rolling_xg_against", "rolling_xg_diff",
        "rolling_goals_for", "rolling_goals_against",
        "rolling_xg_overperformance",
        "rolling_shots_on_goal", "rolling_shots_accuracy",
        "rolling_possession", "rolling_pass_accuracy",
        "rolling_corners", "rolling_avg_rating",
        "rolling_key_passes", "rolling_duels_won_pct",
        "rolling_dribbles_success", "rolling_points",
    ]
    rolling_data = {}
    for key in rolling_keys:
        col = f"{prefix}{key}"
        rolling_data[key] = latest.get(col, np.nan) if latest is not None else np.nan

    days_col = f"{prefix}days_since_last"
    rolling_data["days_since_last"] = latest.get(days_col, np.nan) if latest is not None else np.nan

    return rolling_data


def get_team_elo(features_df, team_name):
    home = features_df[features_df["home_team_name"] == team_name].sort_values("date")
    away = features_df[features_df["away_team_name"] == team_name].sort_values("date")

    home_latest = home.iloc[-1] if len(home) > 0 else None
    away_latest = away.iloc[-1] if len(away) > 0 else None

    if home_latest is not None and away_latest is not None:
        if home_latest["date"] >= away_latest["date"]:
            return home_latest.get("elo_home", np.nan)
        else:
            return away_latest.get("elo_away", np.nan)
    elif home_latest is not None:
        return home_latest.get("elo_home", np.nan)
    elif away_latest is not None:
        return away_latest.get("elo_away", np.nan)
    return np.nan


def get_domestic_features(features_df, team_name, role):
    if role == "home":
        matches = features_df[features_df["home_team_name"] == team_name].sort_values("date")
        cols = [
            "home_domestic_home_goals_avg",
            "home_domestic_goals_against_last5",
            "home_domestic_league_position",
        ]
    else:
        matches = features_df[features_df["away_team_name"] == team_name].sort_values("date")
        cols = [
            "away_domestic_away_goals_avg",
            "away_domestic_xg_against_last5",
        ]

    if len(matches) == 0:
        return {c: np.nan for c in cols}

    latest = matches.iloc[-1]
    return {c: latest.get(c, np.nan) for c in cols}


def build_prediction_row(features_df, home_team, away_team, feature_list, is_knockout):
    home_rolling = get_latest_rolling_features(features_df, home_team, feature_list)
    away_rolling = get_latest_rolling_features(features_df, away_team, feature_list)

    row = {}
    rolling_keys = [
        "rolling_xg_for", "rolling_xg_against", "rolling_xg_diff",
        "rolling_goals_for", "rolling_goals_against",
        "rolling_xg_overperformance",
        "rolling_shots_on_goal", "rolling_shots_accuracy",
        "rolling_possession", "rolling_pass_accuracy",
        "rolling_corners", "rolling_avg_rating",
        "rolling_key_passes", "rolling_duels_won_pct",
        "rolling_dribbles_success", "rolling_points",
    ]
    for key in rolling_keys:
        row[f"home_{key}"] = home_rolling.get(key, np.nan)
        row[f"away_{key}"] = away_rolling.get(key, np.nan)

    row["home_days_since_last"] = home_rolling.get("days_since_last", np.nan)
    row["away_days_since_last"] = away_rolling.get("days_since_last", np.nan)

    row["diff_rolling_xg"] = (
        row.get("home_rolling_xg_for", np.nan) - row.get("away_rolling_xg_for", np.nan)
        if not (pd.isna(row.get("home_rolling_xg_for")) or pd.isna(row.get("away_rolling_xg_for")))
        else np.nan
    )
    row["diff_rolling_goals"] = (
        row.get("home_rolling_goals_for", np.nan) - row.get("away_rolling_goals_for", np.nan)
        if not (pd.isna(row.get("home_rolling_goals_for")) or pd.isna(row.get("away_rolling_goals_for")))
        else np.nan
    )
    row["diff_rolling_form"] = (
        row.get("home_rolling_points", np.nan) - row.get("away_rolling_points", np.nan)
        if not (pd.isna(row.get("home_rolling_points")) or pd.isna(row.get("away_rolling_points")))
        else np.nan
    )
    row["is_knockout"] = is_knockout

    # Competition flags
    row["is_champions"] = 0
    row["is_europa"] = 0
    row["is_conference"] = 0

    # ELO
    elo_home = get_team_elo(features_df, home_team)
    elo_away = get_team_elo(features_df, away_team)
    row["elo_home"] = elo_home
    row["elo_away"] = elo_away
    if not (pd.isna(elo_home) or pd.isna(elo_away)):
        row["elo_diff"] = elo_home - elo_away
        row["elo_expected_home"] = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    else:
        row["elo_diff"] = np.nan
        row["elo_expected_home"] = np.nan

    # Domestic
    home_dom = get_domestic_features(features_df, home_team, "home")
    away_dom = get_domestic_features(features_df, away_team, "away")
    row.update(home_dom)
    row.update(away_dom)

    elo_info = {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": row.get("elo_diff", np.nan),
    }

    X = np.array([[row.get(f, np.nan) for f in feature_list]])
    return X, elo_info


def odds_to_probs(odds_dict):
    raw = {k: 1.0 / v for k, v in odds_dict.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def form_badge(result):
    if result == "W":
        return "🟢 W"
    elif result == "D":
        return "🟡 D"
    else:
        return "🔴 L"


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #d4af37;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #8888aa;
        margin-top: 0;
    }
    .prob-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333355;
    }
    .prob-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #d4af37;
    }
    .prob-label {
        font-size: 0.9rem;
        color: #8888aa;
        margin-top: 4px;
    }
    .confidence-high { color: #4CAF50; font-weight: 600; }
    .confidence-med { color: #FFC107; font-weight: 600; }
    .confidence-low { color: #F44336; font-weight: 600; }
    .stat-compare {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #333355;
        margin: 4px 0;
    }
    .match-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #333355;
        margin: 6px 0;
    }
    .footer-text {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px 0;
        border-top: 1px solid #333355;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App header + competition selector
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">⚽ European Football Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Champions League &bull; Europa League &bull; Conference League</p>', unsafe_allow_html=True)

# Competition selector
comp_options = [f"{cfg['icon']} {name}" for name, cfg in COMPETITIONS.items()]
selected_label = st.selectbox("Selecciona competicion", comp_options, index=0, label_visibility="collapsed")
# Extract name without icon
selected_comp = selected_label.split(" ", 1)[1]
comp_cfg = COMPETITIONS[selected_comp]

# Load data for selected competition
model, le, feature_list = load_competition_model(selected_comp)
all_features_df = load_all_features()
comp_features_df = all_features_df[all_features_df[comp_cfg["comp_filter"]] == 1].copy()

# Also load shared data
original_metrics = load_original_metrics()
all_upcoming_df = load_upcoming_matches()
all_recent_df = load_recent_results()
odds_data = load_odds()

# Filter upcoming & recent for selected competition
comp_upcoming_df = (
    all_upcoming_df[all_upcoming_df["league_name"] == selected_comp]
    if len(all_upcoming_df) > 0 else pd.DataFrame()
)
comp_recent_df = (
    all_recent_df[all_recent_df["league_name"] == selected_comp]
    if len(all_recent_df) > 0 else pd.DataFrame()
)

# Season 2025 for selected competition
s25 = comp_features_df[comp_features_df["season"] == 2025].sort_values("date", ascending=False)
if len(s25) > 0:
    last_date = pd.to_datetime(s25.iloc[0]["date"]).strftime("%d %b %Y")
else:
    last_date = "N/A"


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    f"{comp_cfg['icon']} {selected_comp} Hoy",
    "🔮 Prediccion de Partido",
    "📈 Rendimiento del Modelo",
    "ℹ️ Metodologia",
])

# ===== TAB 1: Competition Today =====
with tab1:
    st.header(f"{comp_cfg['icon']} {selected_comp} 2025/26")

    # Current phase info
    if len(comp_upcoming_df) > 0:
        current_round = comp_upcoming_df.iloc[0]["round"]
        first_match_date = comp_upcoming_df["date"].min()
        st.markdown(f"**Fase actual:** {current_round}")
        st.markdown(f"**Proximo partido:** {first_match_date.strftime('%d %b %Y')}")
    elif len(comp_recent_df) > 0:
        last_round = comp_recent_df.iloc[0]["round"]
        st.markdown(f"**Ultima fase disputada:** {last_round}")

    # Calendar (CL specific)
    if selected_comp == "Champions League":
        with st.expander("Calendario Champions League 2025/26"):
            st.markdown("""
            | Fase | Ida | Vuelta |
            |---|---|---|
            | Knockout Play-offs | 18-19 feb | 25-26 feb |
            | **Octavos de Final** | **10-11 mar** | **17-18 mar** |
            | Cuartos de Final | 7-8 abr | 14-15 abr |
            | Semifinales | 28-29 abr | 5-6 may |
            | **Final (Budapest)** | **30 mayo** | |
            """)

    # Upcoming matches
    st.subheader("Proximos Partidos")
    if len(comp_upcoming_df) > 0:
        for _, match in comp_upcoming_df.iterrows():
            fid = match["fixture_id"]
            date_str = match["date"].strftime("%d %b %H:%M")
            odds_str = ""
            if fid in odds_data:
                o = odds_data[fid]
                odds_str = f"  |  Odds: {o.get('Home', '-')} / {o.get('Draw', '-')} / {o.get('Away', '-')}"

            st.markdown(
                f'<div class="match-card">'
                f'<strong>{date_str}</strong> — {match["home_team"]} vs {match["away_team"]}'
                f'<br><small>{match["round"]}{odds_str}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        if selected_comp == "Champions League":
            st.info("No hay partidos programados. Ejecuta `python3 11_update_fixtures.py` para actualizar.")
        else:
            st.info(f"No hay partidos programados para {selected_comp}. Ejecuta `python3 11_update_fixtures.py` para actualizar.")

    # Recent results (from 11_update_fixtures.py or fallback to features_all)
    st.subheader("Resultados Recientes")
    if len(comp_recent_df) > 0:
        for _, r in comp_recent_df.head(10).iterrows():
            date_str = r["date"].strftime("%d %b")
            hg = int(r["home_goals"]) if pd.notna(r["home_goals"]) else 0
            ag = int(r["away_goals"]) if pd.notna(r["away_goals"]) else 0
            if hg > ag:
                score_display = f"**{r['home_team']}** {hg}-{ag} {r['away_team']}"
            elif ag > hg:
                score_display = f"{r['home_team']} {hg}-{ag} **{r['away_team']}**"
            else:
                score_display = f"{r['home_team']} {hg}-{ag} {r['away_team']}"

            st.markdown(
                f'<div class="match-card">'
                f'<strong>{date_str}</strong> — {score_display}'
                f'<br><small>{r["round"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif len(s25) > 0:
        # Fallback to features_all data
        recent = s25.head(10)
        for _, r in recent.iterrows():
            date_str = pd.to_datetime(r["date"]).strftime("%d %b")
            hg = int(r["home_goals"])
            ag = int(r["away_goals"])
            if hg > ag:
                score_display = f"**{r['home_team_name']}** {hg}-{ag} {r['away_team_name']}"
            elif ag > hg:
                score_display = f"{r['home_team_name']} {hg}-{ag} **{r['away_team_name']}**"
            else:
                score_display = f"{r['home_team_name']} {hg}-{ag} {r['away_team_name']}"

            st.markdown(
                f'<div class="match-card">'
                f'<strong>{date_str}</strong> — {score_display}'
                f'<br><small>{r["round"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(f"No hay resultados recientes para {selected_comp}.")


# ===== TAB 2: Prediction =====
with tab2:
    st.header("Prediccion de Partido")
    st.caption(f"Modelo: {selected_comp} ({comp_cfg['algorithm']})")

    # Quick-pick from upcoming matches
    home_default = None
    away_default = None
    if len(comp_upcoming_df) > 0:
        st.subheader("Partidos programados")
        quick_options = ["Seleccionar manualmente..."]
        match_lookup = {}
        for _, m in comp_upcoming_df.iterrows():
            date_str = m["date"].strftime("%d %b")
            label = f"{date_str}: {m['home_team']} vs {m['away_team']}"
            quick_options.append(label)
            match_lookup[label] = (m["home_team"], m["away_team"])

        quick_pick = st.selectbox("Partido rapido", quick_options, index=0, label_visibility="collapsed")
        if quick_pick != "Seleccionar manualmente..." and quick_pick in match_lookup:
            home_default, away_default = match_lookup[quick_pick]

        st.markdown("---")

    # Teams from selected competition (2025 season preferred, fallback to all)
    if len(s25) > 0:
        teams = sorted(
            set(s25["home_team_name"].unique()) | set(s25["away_team_name"].unique())
        )
    else:
        teams = sorted(
            set(comp_features_df["home_team_name"].unique()) | set(comp_features_df["away_team_name"].unique())
        )

    # Set default indices if quick-pick was used
    home_idx = None
    away_idx = None
    if home_default and home_default in teams:
        home_idx = teams.index(home_default)
    if away_default and away_default in teams:
        away_idx = teams.index(away_default)

    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox(
            "Equipo Local", teams,
            index=home_idx, placeholder="Selecciona equipo local",
        )
    with col_away:
        away_team = st.selectbox(
            "Equipo Visitante", teams,
            index=away_idx, placeholder="Selecciona equipo visitante",
        )

    # Detect knockout phase from upcoming data
    knockout_pattern = r"(Round of|Quarter|Semi|Final|Play-offs|Knockout)"
    is_knockout = 0
    if len(comp_upcoming_df) > 0:
        current_round_str = comp_upcoming_df.iloc[0]["round"]
        if re.search(knockout_pattern, current_round_str, re.IGNORECASE):
            is_knockout = 1
    elif len(comp_recent_df) > 0:
        last_round_str = comp_recent_df.iloc[0]["round"]
        if re.search(knockout_pattern, last_round_str, re.IGNORECASE):
            is_knockout = 1

    predict_btn = st.button("Predecir", type="primary")

    if predict_btn and home_team and away_team:
        if home_team == away_team:
            st.error("Selecciona dos equipos diferentes.")
        else:
            # Use competition-filtered data for rolling features
            X, elo_info = build_prediction_row(comp_features_df, home_team, away_team, feature_list, is_knockout)
            proba = model.predict_proba(X)[0]
            classes = list(le.classes_)

            prob_dict = {c: p for c, p in zip(classes, proba)}
            p_home = prob_dict.get("H", 0)
            p_draw = prob_dict.get("D", 0)
            p_away = prob_dict.get("A", 0)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    label=f"Victoria Local — {home_team}",
                    value=f"{p_home:.0%}",
                )
            with c2:
                st.metric(
                    label="Empate",
                    value=f"{p_draw:.0%}",
                )
            with c3:
                st.metric(
                    label=f"Victoria Visitante — {away_team}",
                    value=f"{p_away:.0%}",
                )

            # Interpretation
            st.markdown("")
            max_prob = max(p_home, p_draw, p_away)
            if p_home == max_prob:
                favored = home_team
                outcome = "una victoria local"
            elif p_away == max_prob:
                favored = away_team
                outcome = "una victoria visitante"
            else:
                favored = None
                outcome = "un empate"

            if favored:
                st.markdown(f"El modelo favorece **{outcome} de {favored}** con **{max_prob:.0%}** de probabilidad.")
            else:
                st.markdown(f"El modelo favorece **{outcome}** con **{max_prob:.0%}** de probabilidad.")

            if max_prob > 0.50:
                st.success("🟢 Confianza alta — El modelo tiene una opinion clara")
            elif max_prob >= 0.40:
                st.warning("🟡 Confianza media — Partido parejo, podria ir para cualquier lado")
            else:
                st.error("🔴 Confianza baja — Mucha incertidumbre, no confies ciegamente")

            # ELO strength
            if not pd.isna(elo_info.get("elo_diff", np.nan)):
                elo_diff = elo_info["elo_diff"]
                if abs(elo_diff) > 10:
                    stronger = home_team if elo_diff > 0 else away_team
                    st.markdown(f"**Diferencia de fuerza (ELO):** {stronger} es **{abs(elo_diff):.0f} puntos** mas fuerte historicamente")
                else:
                    st.markdown(f"**Diferencia de fuerza (ELO):** Equipos de nivel similar ({abs(elo_diff):.0f} puntos de diferencia)")

            # Form
            st.markdown("---")
            st.subheader("Forma Reciente")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown(f"**{home_team}**")
                form_home = get_team_form(comp_features_df, home_team)
                form_str = " ".join(form_badge(r["team_result"]) for _, r in form_home.iterrows())
                st.markdown(form_str)
            with col_f2:
                st.markdown(f"**{away_team}**")
                form_away = get_team_form(comp_features_df, away_team)
                form_str = " ".join(form_badge(r["team_result"]) for _, r in form_away.iterrows())
                st.markdown(form_str)

            # Rolling stats comparison
            st.subheader("Comparacion de Stats (promedio ultimos 5 partidos)")
            home_rolling = get_latest_rolling_features(comp_features_df, home_team, feature_list)
            away_rolling = get_latest_rolling_features(comp_features_df, away_team, feature_list)

            stats_display = [
                ("Goles Esperados (xG)", "rolling_xg_for", ".2f"),
                ("Goles Anotados", "rolling_goals_for", ".1f"),
                ("Goles Recibidos", "rolling_goals_against", ".1f"),
                ("Tiros a Puerta", "rolling_shots_on_goal", ".1f"),
                ("Posesion (%)", "rolling_possession", ".1f"),
                ("Precision Pases (%)", "rolling_pass_accuracy", ".1f"),
                ("Puntos (forma)", "rolling_points", ".1f"),
            ]

            compare_data = []
            for label, key, fmt in stats_display:
                h_val = home_rolling.get(key, np.nan)
                a_val = away_rolling.get(key, np.nan)
                compare_data.append({
                    "Stat": label,
                    home_team: f"{h_val:{fmt}}" if not pd.isna(h_val) else "—",
                    away_team: f"{a_val:{fmt}}" if not pd.isna(a_val) else "—",
                })

            if not pd.isna(elo_info.get("elo_home", np.nan)):
                compare_data.append({
                    "Stat": "ELO Rating",
                    home_team: f"{elo_info['elo_home']:.0f}",
                    away_team: f"{elo_info['elo_away']:.0f}" if not pd.isna(elo_info.get("elo_away", np.nan)) else "—",
                })

            st.dataframe(pd.DataFrame(compare_data), height=330)

            # Odds comparison (any competition with odds data)
            matching_odds = None
            if len(comp_upcoming_df) > 0:
                for _, um in comp_upcoming_df.iterrows():
                    if um["home_team"] == home_team and um["away_team"] == away_team:
                        fid = um["fixture_id"]
                        if fid in odds_data:
                            matching_odds = odds_data[fid]
                        break

                if matching_odds:
                    st.markdown("---")
                    st.subheader("Que dice el mercado?")

                    market_probs = odds_to_probs(matching_odds)
                    m_home = market_probs.get("Home", 0)
                    m_draw = market_probs.get("Draw", 0)
                    m_away = market_probs.get("Away", 0)

                    compare_odds = pd.DataFrame({
                        "": ["Victoria Local", "Empate", "Victoria Visitante"],
                        "Modelo": [f"{p_home:.0%}", f"{p_draw:.0%}", f"{p_away:.0%}"],
                        "Mercado (Pinnacle)": [f"{m_home:.0%}", f"{m_draw:.0%}", f"{m_away:.0%}"],
                        "Diferencia": [
                            f"{(p_home - m_home):+.0%}",
                            f"{(p_draw - m_draw):+.0%}",
                            f"{(p_away - m_away):+.0%}",
                        ],
                    })
                    st.dataframe(compare_odds, height=145)

                    diffs = {
                        f"la victoria de {home_team}": p_home - m_home,
                        "el empate": p_draw - m_draw,
                        f"la victoria de {away_team}": p_away - m_away,
                    }
                    for desc, diff in diffs.items():
                        if abs(diff) > 0.05:
                            direction = "mas probable" if diff > 0 else "menos probable"
                            st.markdown(f"El modelo ve **{desc}** como **{direction}** que el mercado ({abs(diff):.0%} de diferencia)")

                    st.caption(
                        "Las casas de apuestas tienen mas informacion que nuestro modelo. "
                        "Diferencias grandes pueden indicar factores que no capturamos "
                        "(lesiones, motivacion, cambios tacticos, etc.)"
                    )


# ===== TAB 3: Model Performance =====
with tab3:
    st.header("Rendimiento del Modelo")
    st.markdown(
        "Transparencia total: asi es como rinde cada modelo. "
        "Predecir futbol es dificil. Nuestros modelos son mejor que tirar una moneda "
        "pero no son infalibles. Usalos como **una herramienta mas**, no como verdad absoluta."
    )

    # Comparative table — always visible
    st.subheader("Comparativa de los 3 Modelos")
    comp_table = []
    for name, cfg in COMPETITIONS.items():
        m = load_competition_metrics(name)
        acc = m.get("accuracy", 0)
        ll = m.get("log_loss", 0)
        f1 = m.get("f1_macro", 0)
        comp_table.append({
            "Competicion": f"{cfg['icon']} {name}",
            "Partidos": cfg["matches"],
            "Accuracy": f"{acc:.1%}",
            "Log Loss": f"{ll:.3f}",
            "F1 Macro": f"{f1:.3f}",
            "Algoritmo": cfg["algorithm"],
        })
    st.dataframe(pd.DataFrame(comp_table), height=145)

    # Why differences
    st.markdown("""
    **Por que las diferencias?**
    - **Europa League (59.4%)** tiene la mayor accuracy porque entrena con datos de EL + Conference League (1,762 partidos),
      y los equipos de EL tienen mejor cobertura de ELO y datos domesticos que Conference.
    - **Champions League (57.2%)** tiene menos partidos (709) pero equipos de elite con excelente cobertura de datos.
    - **Conference League (52.7%)** es la mas dificil: equipos pequenos con poca cobertura de ELO (47%) y
      datos domesticos (17%), lo que limita las features mas predictivas del modelo.
    """)

    st.markdown("---")

    # Selected competition details
    st.subheader(f"Detalle: {comp_cfg['icon']} {selected_comp}")

    comp_metrics = load_competition_metrics(selected_comp)
    acc = comp_metrics.get("accuracy", 0)
    ll = comp_metrics.get("log_loss", 0)
    f1 = comp_metrics.get("f1_macro", 0)
    n_feat = comp_metrics.get("n_features", 20)
    n_matches = comp_cfg["matches"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision del Modelo", f"{acc:.1%}")
    with col2:
        st.metric("Partidos Analizados", f"{n_matches:,}")
    with col3:
        st.metric("Variables del Modelo", f"{n_feat}")

    st.markdown(f"**Algoritmo:** {comp_cfg['algorithm']} | **Log Loss:** {ll:.3f} | **F1 Macro:** {f1:.3f}")

    # Training variants for this competition
    if "all_results" in comp_metrics:
        st.subheader("Variantes de Entrenamiento")
        var_data = []
        for var_name, models in comp_metrics["all_results"].items():
            for model_name, m in models.items():
                var_data.append({
                    "Variante": var_name,
                    "Modelo": model_name,
                    "Accuracy": f"{m['accuracy']:.1%}",
                    "Log Loss": f"{m['log_loss']:.3f}",
                    "F1 Macro": f"{m['f1_macro']:.3f}",
                })
        st.dataframe(pd.DataFrame(var_data), height=350)

    # CL-specific: walk-forward details and calibration chart
    if selected_comp == "Champions League":
        cl_metrics_final = comp_metrics

        baseline_acc = original_metrics["averages"]["Baseline (always H)"]["avg_accuracy"]
        improvement = acc - baseline_acc

        st.markdown(f"Acierta el resultado en **casi 6 de cada 10 partidos**.")
        st.markdown(f"Si siempre apostaramos al equipo local, acertariamos un {baseline_acc:.0%}. Nuestro modelo mejora esto en **{improvement:.0%}**.")

        # Walk-forward results
        st.subheader("Resultados por Periodo (Walk-Forward)")
        st.markdown("Evaluamos el modelo de forma temporal: entrenamos con datos pasados y probamos con datos futuros, simulando uso real.")

        wf_data = []
        split_names = ["Split 1", "Split 2", "Split 3"]
        baselines = []
        for split_name, split_results in original_metrics["walk_forward_results"].items():
            bl = split_results.get("Baseline (always H)", {})
            baselines.append(bl.get("accuracy", 0))
        for i, ps in enumerate(cl_metrics_final.get("per_split", [])):
            bl_acc = baselines[i] if i < len(baselines) else 0
            wf_data.append({
                "Periodo": split_names[i],
                "Modelo": f"{ps['accuracy']:.0%}",
                "Baseline": f"{bl_acc:.0%}",
                "Log Loss": f"{ps['log_loss']:.3f}",
                "F1 Macro": f"{ps['f1_macro']:.3f}",
            })
        st.dataframe(pd.DataFrame(wf_data), height=145)

        # Calibration chart
        st.subheader("Calibracion del Modelo")
        st.markdown("Cuando el modelo dice '50% de probabilidad', realmente pasa el 50% de las veces?")

        split3_key = [k for k in original_metrics["walk_forward_results"] if "Split 3" in k][0]
        rf_split3 = original_metrics["walk_forward_results"][split3_key]["Random Forest"]
        cal = rf_split3.get("calibration", {})

        cal_data = pd.DataFrame([
            {"Resultado": "Victoria Local (H)", "Predicho": cal["H"]["predicted_prob"], "Real": cal["H"]["actual_freq"]},
            {"Resultado": "Empate (D)", "Predicho": cal["D"]["predicted_prob"], "Real": cal["D"]["actual_freq"]},
            {"Resultado": "Victoria Visitante (A)", "Predicho": cal["A"]["predicted_prob"], "Real": cal["A"]["actual_freq"]},
        ])

        cal_melted = cal_data.melt(id_vars="Resultado", var_name="Tipo", value_name="Probabilidad")

        chart = alt.Chart(cal_melted).mark_bar().encode(
            x=alt.X("Tipo:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Probabilidad:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 0.6])),
            color=alt.Color("Tipo:N", scale=alt.Scale(
                domain=["Predicho", "Real"],
                range=["#d4af37", "#4CAF50"],
            )),
            column=alt.Column("Resultado:N", title=None, header=alt.Header(labelAngle=0)),
        ).properties(height=300, width=150)

        st.altair_chart(chart)

    # Feature importance (same for all — same 20 features)
    st.subheader("Que factores importan mas?")
    st.markdown("Las 10 variables mas importantes del modelo (ordenadas por impacto en la prediccion):")

    feat_explanations = {
        "home_rolling_points": ("Puntos recientes (local)", "La forma reciente del equipo local — la variable #1"),
        "elo_diff": ("Diferencia de ELO", "ELO Rating: Un numero que resume la fuerza historica del equipo"),
        "home_domestic_home_goals_avg": ("Goles en casa (liga domestica)", "Promedio de goles como local en su liga domestica"),
        "elo_home": ("ELO del equipo local", "La fuerza historica acumulada del equipo local"),
        "elo_away": ("ELO del equipo visitante", "La fuerza historica acumulada del equipo visitante"),
        "away_domestic_xg_against_last5": ("xG en contra reciente (liga visitante)", "Goles esperados en contra en los ultimos 5 partidos de liga del visitante"),
        "home_rolling_shots_accuracy": ("Precision de tiros (local)", "Porcentaje de tiros que van a puerta"),
        "home_rolling_avg_rating": ("Rating promedio (local)", "Calificacion promedio de jugadores locales en partidos recientes"),
        "home_rolling_corners": ("Corners recientes (local)", "Presion ofensiva del equipo local medida en corners"),
        "home_rolling_duels_won_pct": ("Duelos ganados (local)", "Porcentaje de duelos individuales ganados"),
        "away_rolling_duels_won_pct": ("Duelos ganados (visitante)", "Porcentaje de duelos individuales ganados por el visitante"),
        "away_rolling_xg_overperformance": ("Sobrerendimiento xG (visitante)", "Diferencia entre goles reales y esperados del visitante"),
        "home_rolling_key_passes": ("Pases clave (local)", "Pases que crean oportunidades de gol"),
        "home_rolling_pass_accuracy": ("Precision de pases (local)", "Equipos que conectan pases tienden a controlar partidos"),
        "away_rolling_goals_for": ("Goles recientes (visitante)", "El poder ofensivo reciente del equipo visitante"),
        "home_rolling_xg_diff": ("Diferencia xG (local)", "Diferencia entre goles esperados a favor y en contra"),
        "away_domestic_away_goals_avg": ("Goles de visita (liga domestica)", "Promedio de goles como visitante en su liga domestica"),
        "away_rolling_corners": ("Corners recientes (visitante)", "Presion ofensiva del visitante"),
        "home_days_since_last": ("Dias de descanso (local)", "El descanso entre partidos afecta el rendimiento"),
        "away_rolling_points": ("Puntos recientes (visitante)", "La forma reciente del visitante"),
    }

    for i, fname in enumerate(feature_list[:10], 1):
        explanation = feat_explanations.get(fname, (fname, ""))
        tag = ""
        if "elo" in fname:
            tag = " `ELO`"
        elif "domestic" in fname:
            tag = " `DOMESTIC`"
        st.markdown(f"**{i}. {explanation[0]}**{tag}")
        st.caption(explanation[1])


# ===== TAB 4: Methodology =====
with tab4:
    st.header("Metodologia")

    st.subheader("Datos")
    st.markdown("""
    - **2,471 partidos** de tres competiciones UEFA
    - **Champions League:** 709 partidos (3 temporadas: 2023, 2024, 2025)
    - **Europa League:** 650 partidos (3 temporadas: 2023, 2024, 2025)
    - **Conference League:** 1,112 partidos (3 temporadas: 2023, 2024, 2025)
    - Datos de [API-Football](https://www.api-football.com/) incluyendo estadisticas de partido y rendimiento de jugadores
    - **ELO ratings** de [clubelo.com](http://clubelo.com/) — ratings historicos de fuerza de cada equipo
    - **Datos de ligas domesticas** — rendimiento de cada equipo en su liga local
    """)

    st.subheader("Por que 3 modelos independientes?")
    st.markdown("""
    Experimentamos con un modelo unificado (entrenar con las 3 competiciones), pero **no mejoro** la
    precision para Champions League. Cada competicion tiene caracteristicas distintas:

    - **Champions League** — equipos de elite, partidos cerrados, baja ventaja de local (+15%)
    - **Europa League** — mayor ventaja de local (+20%), equipos de nivel intermedio
    - **Conference League** — equipos mas dispares, menos cobertura de datos (ELO: 47%, domestico: 17%)

    Los modelos independientes permiten capturar estas diferencias. Europa League se beneficia
    de entrenar con datos de Conference League (59.4% accuracy), mientras Conference League
    necesita datos de todas las competiciones para rendir mejor (transfer learning).
    """)

    st.subheader("Variables del modelo (20 por competicion)")
    st.markdown("""
    Cada modelo usa las mismas **20 variables** seleccionadas por importancia:

    **Competicion europea (13 variables)** — promedio de los ultimos 5 partidos:
    Puntos (forma), precision de tiros, rating de jugadores, corners, duelos ganados,
    sobrerendimiento xG, pases clave, precision de pases, goles del visitante, diferencia xG,
    corners del visitante, dias de descanso, puntos del visitante

    **ELO Ratings (3 variables):**
    Diferencia de ELO (la variable #2 del modelo), ELO del local, ELO del visitante

    **Liga domestica (4 variables):**
    Goles en casa, goles de visita, xG en contra, todas de la liga local de cada equipo
    """)

    st.subheader("Algoritmos")
    st.markdown("""
    | Competicion | Algoritmo | Por que? |
    |---|---|---|
    | Champions League | LogReg calibrada (C=0.01, L2) | Mejor precision con pocos datos (709 partidos) |
    | Europa League | Random Forest (500 arboles) | Se beneficia de mas datos (EL + ECL = 1,762) |
    | Conference League | Random Forest (500 arboles) | Transfer learning con todas las competiciones |
    """)

    st.subheader("Evaluacion: Walk-Forward Validation")
    st.markdown("""
    Simulamos uso real: entrenamos con datos pasados y probamos con datos futuros:

    1. Entrenamos con datos de 2023, predecimos inicio de 2024
    2. Entrenamos con datos de 2023 + inicio de 2024, predecimos final de 2024
    3. Entrenamos con datos de 2023 + 2024, predecimos 2025

    Esto es mas honesto que mezclar datos aleatoriamente.
    """)

    st.subheader("Evolucion del proyecto")
    st.markdown("""
    | Version | Descripcion | Accuracy CL |
    |---------|-------------|-------------|
    | v1 (base) | Random Forest, 38 features rolling CL | 51.5% |
    | v2 (+ domestic) | +21 features domesticas | 49.9% (descartado) |
    | v3 (+ ELO) | +4 ELO + 5 domestic features | 53.7% |
    | **v4 (final CL)** | **LogReg calibrada, 20 features** | **57.2%** |
    | **v5 (multi-comp)** | **+ Europa League + Conference League** | **EL: 59.4%, ECL: 52.7%** |
    """)

    st.subheader("Que NO pueden hacer los modelos?")
    st.markdown("""
    - **No consideran lesiones** ni suspensiones de jugadores
    - **No saben de motivacion** ni presion de la aficion
    - **No detectan cambios tacticos** ni fichajes recientes
    - **No tienen informacion en tiempo real** — se basan en datos historicos
    - **Predecir empates es muy dificil** — es el resultado mas impredecible en futbol
    """)


# Footer
st.markdown(
    f'<div class="footer-text">'
    f'Hecho con datos de API-Football y clubelo.com · '
    f'3 modelos: CL 57.2% · EL 59.4% · ECL 52.7% · '
    f'Actualizado al {last_date}'
    f'</div>',
    unsafe_allow_html=True,
)
