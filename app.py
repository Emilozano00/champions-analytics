"""
⚽ Champions League Predictor
Streamlit app for Champions League match outcome prediction.
Uses model v3: ELO ratings + top domestic features (47 variables).
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
    page_title="Champions League Predictor",
    page_icon="⚽",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "champion_model_v3.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_list_v3.json") as f:
        feature_list = json.load(f)
    return model, le, feature_list


@st.cache_data
def load_features():
    return pd.read_csv(PROCESSED_DIR / "features_v3.csv")


@st.cache_data
def load_matches():
    return pd.read_csv(PROCESSED_DIR / "matches.csv")


@st.cache_data
def load_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics_v3():
    with open(MODELS_DIR / "metrics_v3.json") as f:
        return json.load(f)


@st.cache_data
def load_upcoming_fixtures():
    """Load upcoming fixtures from processed CSV."""
    path = PROCESSED_DIR / "upcoming.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_odds():
    """Load Pinnacle odds from processed CSV. Returns dict {fixture_id: {Home, Draw, Away}}."""
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
    """Get last N results for a team as list of W/D/L."""
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
    """Get the most recent rolling features for a team."""
    home = features_df[features_df["home_team_name"] == team_name].sort_values("date")
    away = features_df[features_df["away_team_name"] == team_name].sort_values("date")

    home_latest = home.iloc[-1] if len(home) > 0 else None
    away_latest = away.iloc[-1] if len(away) > 0 else None

    # Pick whichever is more recent
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

    # Extract rolling features for this team
    rolling_data = {}
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
        col = f"{prefix}{key}"
        rolling_data[key] = latest.get(col, np.nan) if latest is not None else np.nan

    days_col = f"{prefix}days_since_last"
    rolling_data["days_since_last"] = latest.get(days_col, np.nan) if latest is not None else np.nan

    return rolling_data


def get_team_elo(features_df, team_name):
    """Get the latest ELO rating for a team from historical match data."""
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
    """Get latest domestic features for a team.
    role: 'home' or 'away' — determines which domestic columns to extract.
    """
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
    """Build a feature row for prediction from the latest available rolling stats.
    Returns (X array for model, elo_info dict for display).
    """
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

    # Derived features
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

    # ELO features
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

    # Domestic features
    home_dom = get_domestic_features(features_df, home_team, "home")
    away_dom = get_domestic_features(features_df, away_team, "away")
    row.update(home_dom)
    row.update(away_dom)

    # ELO info for display in prediction tab
    elo_info = {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": row.get("elo_diff", np.nan),
    }

    # Build array in feature_list order
    X = np.array([[row.get(f, np.nan) for f in feature_list]])
    return X, elo_info


def odds_to_probs(odds_dict):
    """Convert decimal odds to normalized implied probabilities."""
    raw = {k: 1.0 / v for k, v in odds_dict.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def form_badge(result):
    """Return a colored badge for W/D/L."""
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
# App header
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">⚽ Champions League Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicciones basadas en datos para la UEFA Champions League</p>', unsafe_allow_html=True)

# Load everything
model, le, feature_list = load_model()
features_df = load_features()
matches_df = load_matches()
metrics = load_metrics()
metrics_v3 = load_metrics_v3()
upcoming_df = load_upcoming_fixtures()
odds_data = load_odds()

# Determine current phase
s25 = features_df[features_df["season"] == 2025].sort_values("date", ascending=False)
last_date = pd.to_datetime(s25.iloc[0]["date"]).strftime("%d %b %Y")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Champions League Hoy",
    "🔮 Predicción de Partido",
    "📈 Rendimiento del Modelo",
    "ℹ️ Metodología",
])

# ===== TAB 1: Champions League Hoy =====
with tab1:
    st.header("Champions League 2025")

    # Current phase
    if len(upcoming_df) > 0:
        current_round = upcoming_df.iloc[0]["round"]
        st.markdown(f"**Fase actual:** {current_round}")
    else:
        current_round = s25.iloc[0]["round"]
        st.markdown(f"**Última fase disputada:** {current_round}")

    # Upcoming matches
    st.subheader("Próximos Partidos")
    if len(upcoming_df) > 0:
        for _, match in upcoming_df.iterrows():
            fid = match["fixture_id"]
            date_str = pd.to_datetime(match["date"]).strftime("%d %b %H:%M")
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
        st.info("No hay partidos pendientes en este momento.")

    # Recent results
    st.subheader("Resultados Recientes")
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


# ===== TAB 2: Predicción de Partido =====
with tab2:
    st.header("Predicción de Partido")

    # Teams from 2025 season
    teams_2025 = sorted(
        set(s25["home_team_name"].unique()) | set(s25["away_team_name"].unique())
    )

    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("Equipo Local", teams_2025, index=None, placeholder="Selecciona equipo local")
    with col_away:
        away_team = st.selectbox("Equipo Visitante", teams_2025, index=None, placeholder="Selecciona equipo visitante")

    # Determine if this is a knockout matchup
    knockout_pattern = r"(Round of|Quarter|Semi|Final|Play-offs|Knockout)"
    is_knockout = 0
    if len(upcoming_df) > 0:
        current_round_str = upcoming_df.iloc[0]["round"]
        if re.search(knockout_pattern, current_round_str, re.IGNORECASE):
            is_knockout = 1

    predict_btn = st.button("Predecir", type="primary")

    if predict_btn and home_team and away_team:
        if home_team == away_team:
            st.error("Selecciona dos equipos diferentes.")
        else:
            # Build prediction
            X, elo_info = build_prediction_row(features_df, home_team, away_team, feature_list, is_knockout)
            proba = model.predict_proba(X)[0]
            classes = list(le.classes_)

            # Map probabilities
            prob_dict = {c: p for c, p in zip(classes, proba)}
            p_home = prob_dict.get("H", 0)
            p_draw = prob_dict.get("D", 0)
            p_away = prob_dict.get("A", 0)

            # Display probabilities
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

            # Confidence level
            if max_prob > 0.50:
                st.success("🟢 Confianza alta — El modelo tiene una opinión clara")
            elif max_prob >= 0.40:
                st.warning("🟡 Confianza media — Partido parejo, podría ir para cualquier lado")
            else:
                st.error("🔴 Confianza baja — Mucha incertidumbre, no confíes ciegamente")

            # ELO strength comparison
            if not pd.isna(elo_info.get("elo_diff", np.nan)):
                elo_diff = elo_info["elo_diff"]
                if abs(elo_diff) > 10:
                    stronger = home_team if elo_diff > 0 else away_team
                    st.markdown(f"**Diferencia de fuerza (ELO):** {stronger} es **{abs(elo_diff):.0f} puntos** más fuerte históricamente")
                else:
                    st.markdown(f"**Diferencia de fuerza (ELO):** Equipos de nivel similar ({abs(elo_diff):.0f} puntos de diferencia)")

            # Form section
            st.markdown("---")
            st.subheader("Forma Reciente")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown(f"**{home_team}**")
                form_home = get_team_form(features_df, home_team)
                form_str = " ".join(form_badge(r["team_result"]) for _, r in form_home.iterrows())
                st.markdown(form_str)
            with col_f2:
                st.markdown(f"**{away_team}**")
                form_away = get_team_form(features_df, away_team)
                form_str = " ".join(form_badge(r["team_result"]) for _, r in form_away.iterrows())
                st.markdown(form_str)

            # Rolling stats comparison
            st.subheader("Comparación de Stats (promedio últimos 5 partidos)")
            home_rolling = get_latest_rolling_features(features_df, home_team, feature_list)
            away_rolling = get_latest_rolling_features(features_df, away_team, feature_list)

            stats_display = [
                ("Goles Esperados (xG)", "rolling_xg_for", ".2f"),
                ("Goles Anotados", "rolling_goals_for", ".1f"),
                ("Goles Recibidos", "rolling_goals_against", ".1f"),
                ("Tiros a Puerta", "rolling_shots_on_goal", ".1f"),
                ("Posesión (%)", "rolling_possession", ".1f"),
                ("Precisión Pases (%)", "rolling_pass_accuracy", ".1f"),
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

            # Add ELO to comparison if available
            if not pd.isna(elo_info.get("elo_home", np.nan)):
                compare_data.append({
                    "Stat": "ELO Rating",
                    home_team: f"{elo_info['elo_home']:.0f}",
                    away_team: f"{elo_info['elo_away']:.0f}" if not pd.isna(elo_info.get("elo_away", np.nan)) else "—",
                })

            st.dataframe(pd.DataFrame(compare_data), height=330)

            # Odds comparison
            # Find if there's a matching fixture in upcoming
            matching_odds = None
            for _, um in upcoming_df.iterrows():
                if um["home_team"] == home_team and um["away_team"] == away_team:
                    fid = um["fixture_id"]
                    if fid in odds_data:
                        matching_odds = odds_data[fid]
                    break

            if matching_odds:
                st.markdown("---")
                st.subheader("📊 ¿Qué dice el mercado?")

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

                # Insights
                diffs = {
                    f"la victoria de {home_team}": p_home - m_home,
                    "el empate": p_draw - m_draw,
                    f"la victoria de {away_team}": p_away - m_away,
                }
                for desc, diff in diffs.items():
                    if abs(diff) > 0.05:
                        direction = "más probable" if diff > 0 else "menos probable"
                        st.markdown(f"💡 El modelo ve **{desc}** como **{direction}** que el mercado ({abs(diff):.0%} de diferencia)")

                st.caption(
                    "Las casas de apuestas tienen más información que nuestro modelo. "
                    "Diferencias grandes pueden indicar factores que no capturamos "
                    "(lesiones, motivación, cambios tácticos, etc.)"
                )


# ===== TAB 3: Rendimiento del Modelo =====
with tab3:
    st.header("Rendimiento del Modelo")
    st.markdown(
        "Transparencia total: así es cómo rinde nuestro modelo. "
        "Predecir fútbol es difícil. Nuestro modelo es mejor que tirar una moneda "
        "pero no es infalible. Úsalo como **una herramienta más**, no como verdad absoluta."
    )

    # v3 headline metrics
    best_acc = metrics_v3["accuracy"]
    best_ll = metrics_v3["log_loss"]
    baseline_acc = metrics["averages"]["Baseline (always H)"]["avg_accuracy"]
    improvement = best_acc - baseline_acc
    v1_acc = metrics_v3["per_variant"]["A"]["accuracy"]
    elo_improvement = best_acc - v1_acc

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precisión del Modelo", f"{best_acc:.0%}", f"+{improvement:.0%} vs baseline")
    with col2:
        st.metric("Partidos Analizados", "709")
    with col3:
        st.metric("Variables del Modelo", "47", f"+9 vs v1 (ELO + domestic)")

    st.markdown(f"Acierta el resultado en **{best_acc*10:.1f} de cada 10 partidos**.")
    st.markdown(f"Si siempre apostáramos al equipo local, acertaríamos un {baseline_acc:.0%}. Nuestro modelo mejora esto en **{improvement:.0%}**.")
    st.markdown(f"La incorporación de ELO ratings y datos domésticos mejoró la precisión en **+{elo_improvement:.1%}** y redujo el log loss a **{best_ll:.3f}**.")

    # Walk-forward results (from original metrics.json for split details)
    st.subheader("Resultados por Periodo (Walk-Forward)")
    st.markdown("Evaluamos el modelo de forma temporal: entrenamos con datos pasados y probamos con datos futuros, simulando uso real.")

    wf_data = []
    for split_name, split_results in metrics["walk_forward_results"].items():
        short_name = split_name.split(":")[0]
        rf = split_results.get("Random Forest", {})
        bl = split_results.get("Baseline (always H)", {})
        wf_data.append({
            "Periodo": short_name,
            "Modelo": f"{rf.get('accuracy', 0):.0%}",
            "Baseline": f"{bl.get('accuracy', 0):.0%}",
            "Log Loss": f"{rf.get('log_loss', 0):.3f}",
            "F1 Macro": f"{rf.get('f1_macro', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(wf_data), height=145)

    # Calibration chart
    st.subheader("Calibración del Modelo")
    st.markdown("¿Cuando el modelo dice '50% de probabilidad', realmente pasa el 50% de las veces?")

    # Build calibration from split 3 (most representative)
    split3_key = [k for k in metrics["walk_forward_results"] if "Split 3" in k][0]
    rf_split3 = metrics["walk_forward_results"][split3_key]["Random Forest"]
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

    # Performance by phase
    st.subheader("Rendimiento por Fase")

    phase_data = pd.DataFrame([
        {"Fase": "Clasificación", "Precisión": "~53%", "Nota": "Más predecible — equipos con diferencias claras de nivel"},
        {"Fase": "Fase de Grupos / Liga", "Precisión": "~50%", "Nota": "Rendimiento sólido en la fase principal"},
        {"Fase": "Eliminatorias", "Precisión": "~43%", "Nota": "Más difícil — partidos cerrados, factor emocional alto"},
    ])
    st.dataframe(phase_data, height=145)

    # Feature importance (from v3)
    st.subheader("¿Qué factores importan más?")
    st.markdown("Las variables que más influyen en las predicciones del modelo v3:")

    feat_explanations = {
        "elo_diff": ("Diferencia de ELO", "ELO Rating: Un número que resume la fuerza histórica del equipo. Equipos con más victorias y contra mejores rivales tienen ELO más alto."),
        "elo_expected_home": ("Probabilidad esperada por ELO", "Probabilidad teórica de victoria local basada en la diferencia de fuerza entre ambos equipos"),
        "elo_home": ("ELO del equipo local", "La fuerza histórica acumulada del equipo local, basada en todos sus resultados pasados"),
        "elo_away": ("ELO del equipo visitante", "La fuerza histórica acumulada del equipo visitante"),
        "home_domestic_home_goals_avg": ("Goles en casa (liga doméstica)", "Promedio de goles como local en su liga doméstica — indica poder ofensivo en casa"),
        "away_domestic_xg_against_last5": ("xG en contra reciente (liga visitante)", "Goles esperados en contra en los últimos 5 partidos de liga del visitante"),
        "away_domestic_away_goals_avg": ("Goles de visita (liga doméstica)", "Promedio de goles como visitante en su liga doméstica"),
        "home_domestic_goals_against_last5": ("Goles recibidos (liga local)", "Goles recibidos en los últimos 5 partidos de liga del equipo local"),
        "home_domestic_league_position": ("Posición en liga doméstica", "Posición actual en la tabla de su liga doméstica"),
        "home_rolling_points": ("Puntos recientes (local)", "La forma reciente del equipo local en Champions League"),
        "away_rolling_goals_for": ("Goles recientes del visitante", "El poder ofensivo reciente del equipo visitante en Champions"),
        "home_rolling_pass_accuracy": ("Precisión de pases reciente (local)", "Equipos que conectan pases tienden a controlar partidos"),
        "away_rolling_possession": ("Posesión reciente (visitante)", "Control del balón del equipo visitante en sus últimos 5 partidos"),
        "away_rolling_corners": ("Córners recientes (visitante)", "Presión ofensiva del visitante medida en córners"),
        "home_rolling_possession": ("Posesión reciente (local)", "Control del balón del equipo local"),
        "away_rolling_avg_rating": ("Rating promedio (visitante)", "Calificación promedio de jugadores visitantes"),
        "home_rolling_avg_rating": ("Rating promedio (local)", "Calificación promedio de jugadores locales"),
        "home_days_since_last": ("Días desde el último partido (local)", "El descanso entre partidos afecta el rendimiento"),
    }

    top_features = metrics_v3.get("feature_importance_top15", [])[:10]
    for i, feat in enumerate(top_features, 1):
        fname = feat["feature"]
        explanation = feat_explanations.get(fname, (fname, ""))
        tag = ""
        if "elo" in fname:
            tag = " `ELO`"
        elif "domestic" in fname:
            tag = " `DOMESTIC`"
        st.markdown(f"**{i}. {explanation[0]}**{tag}")
        st.caption(explanation[1])


# ===== TAB 4: Metodología =====
with tab4:
    st.header("Metodología")

    st.subheader("Datos")
    st.markdown("""
    - **709 partidos** de la UEFA Champions League
    - **3 temporadas**: 2023, 2024 y 2025
    - Datos de [API-Football](https://www.api-football.com/) incluyendo estadísticas de partido y rendimiento de jugadores
    - **ELO ratings** de [clubelo.com](http://clubelo.com/) — ratings históricos de fuerza de cada equipo
    - **Datos de ligas domésticas** — rendimiento de cada equipo en su liga local
    """)

    st.subheader("Variables del modelo (47 total)")
    st.markdown("""
    **Rolling Champions League (32 variables)** — promedio de los últimos 5 partidos de Champions:

    *Ofensivas:* Goles esperados (xG), goles anotados, tiros a puerta, precisión de tiros, regates exitosos, pases clave

    *Defensivas:* Goles recibidos, tackles, intercepciones, duelos ganados

    *Generales:* Posesión, precisión de pases, córners, puntos recientes (forma), calificación promedio de jugadores

    **Contextuales (6 variables):**
    Días de descanso (local/visitante), si es partido de eliminatoria, diferencias rolling (xG, goles, forma)

    **ELO Ratings (4 variables):**
    ELO del local, ELO del visitante, diferencia de ELO, probabilidad esperada por ELO.
    El ELO es un sistema de puntuación que resume la fuerza histórica de un equipo.
    Equipos con más victorias y contra mejores rivales acumulan un ELO más alto.
    Es la variable más importante del modelo.

    **Liga doméstica (5 variables):**
    Goles en casa/fuera, goles recibidos, xG en contra, y posición en la tabla de la liga local.
    Capturan el rendimiento del equipo fuera de Champions League.
    """)

    st.subheader("ELO Ratings")
    st.markdown("""
    El sistema ELO fue inventado para el ajedrez y adaptado al fútbol por [clubelo.com](http://clubelo.com/).
    Funciona así:

    - Cada equipo tiene un número (generalmente entre 1200 y 2000)
    - Ganar contra un equipo fuerte sube mucho tu ELO; ganar contra uno débil sube poco
    - Perder contra un equipo débil baja mucho tu ELO; perder contra uno fuerte baja poco
    - El ELO se actualiza después de cada partido

    En nuestro modelo, la **diferencia de ELO** entre los dos equipos es la variable #1 en importancia.
    Tiene una correlación de +0.49 con el resultado, más alta que cualquier otra variable.
    """)

    st.subheader("¿Cómo evaluamos el modelo?")
    st.markdown("""
    Usamos **walk-forward validation**, que simula cómo funcionaría el modelo en la vida real:

    1. Entrenamos con datos de 2023, predecimos inicio de 2024
    2. Entrenamos con datos de 2023 + inicio de 2024, predecimos final de 2024
    3. Entrenamos con datos de 2023 + 2024, predecimos 2025

    Esto es más honesto que mezclar todos los datos aleatoriamente, porque en la realidad no puedes usar datos del futuro para predecir.
    """)

    st.subheader("Evolución del modelo")
    st.markdown("""
    | Versión | Variables | Precisión | Cambio |
    |---------|-----------|-----------|--------|
    | v1 (base) | 38 (rolling CL) | 51.5% | — |
    | v2 (+ domestic) | 59 (+ 21 domésticas) | 49.9% | -1.6% (descartado) |
    | **v3 (+ ELO + top domestic)** | **47 (+ 4 ELO + 5 domestic)** | **53.7%** | **+2.2%** |

    La v2 demostró que agregar muchas features domésticas genera ruido con solo 709 muestras.
    La v3 usa solo las 5 mejores features domésticas combinadas con ELO, logrando la mejor precisión.
    """)

    st.subheader("¿Qué NO puede hacer el modelo?")
    st.markdown("""
    - **No considera lesiones** ni suspensiones de jugadores
    - **No sabe de motivación** ni presión de la afición
    - **No detecta cambios tácticos** ni fichajes recientes
    - **No tiene información en tiempo real** — se basa en datos históricos
    - **Predecir empates es muy difícil** — es el resultado más impredecible en fútbol
    """)

    st.subheader("Modelo técnico")
    st.markdown(f"""
    - **Algoritmo:** Random Forest (scikit-learn, 300 árboles)
    - **Features:** 47 variables (32 rolling CL + 6 contextuales + 4 ELO + 5 domésticas)
    - **Precisión promedio:** {best_acc:.1%} en walk-forward validation
    - **Log loss:** {best_ll:.3f}
    - **Mejor que el baseline** (siempre apostar al local: ~50%)
    - **ELO como variable #1** — la diferencia de fuerza histórica es el mejor predictor
    """)


# Footer
st.markdown(
    f'<div class="footer-text">'
    f'Hecho con datos de API-Football y clubelo.com · Modelo v3 actualizado al {last_date}'
    f'</div>',
    unsafe_allow_html=True,
)
