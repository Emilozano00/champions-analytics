"""
03_feature_engineering.py

Transforms raw Champions League JSON data (fixtures, statistics, players)
into structured CSVs with rolling features ready for predictive modeling.

Outputs:
  - data/processed/matches.csv  — one row per match, all raw stats
  - data/processed/features.csv — rolling features + derived features
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
EXPLORATION_DIR = RAW_DIR / "exploration"
STATISTICS_DIR = RAW_DIR / "statistics"
PLAYERS_DIR = RAW_DIR / "players"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from API stat type names to snake_case column names
STAT_NAME_MAP = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "total_shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_insidebox",
    "Shots outsidebox": "shots_outsidebox",
    "Fouls": "fouls",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes",
    "Passes accurate": "passes_accurate",
    "Passes %": "pass_pct",
    "expected_goals": "xg",
    "goals_prevented": "goals_prevented",
}

ROLLING_WINDOW = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_stat_value(stat_type: str, value):
    """Parse a single statistic value based on its type."""
    if value is None:
        return 0.0 if stat_type not in ("Ball Possession", "Passes %", "expected_goals") else None
    if stat_type in ("Ball Possession", "Passes %"):
        return float(str(value).replace("%", ""))
    if stat_type == "expected_goals":
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def parse_team_stats(statistics_list: list) -> dict:
    """Convert a list of {type, value} dicts into a flat dict with snake_case keys."""
    result = {}
    for stat in statistics_list:
        col_name = STAT_NAME_MAP.get(stat["type"])
        if col_name:
            result[col_name] = parse_stat_value(stat["type"], stat["value"])
    return result


def compute_player_aggregates(players_list: list) -> dict:
    """Aggregate player-level stats for a single team in a single fixture."""
    ratings = []
    key_passes = 0
    duels_total = 0
    duels_won = 0
    dribbles_success = 0
    tackles_total = 0
    interceptions = 0

    for p in players_list:
        s = p["statistics"][0]
        minutes = s["games"]["minutes"]
        if minutes is None or minutes <= 0:
            continue

        rating = s["games"]["rating"]
        if rating is not None:
            try:
                ratings.append(float(rating))
            except (ValueError, TypeError):
                pass

        key_passes += (s["passes"]["key"] or 0)
        duels_total += (s["duels"]["total"] or 0)
        duels_won += (s["duels"]["won"] or 0)
        dribbles_success += (s["dribbles"]["success"] or 0)
        tackles_total += (s["tackles"]["total"] or 0)
        interceptions += (s["tackles"]["interceptions"] or 0)

    return {
        "avg_rating": sum(ratings) / len(ratings) if ratings else None,
        "key_passes": key_passes,
        "duels_total": duels_total,
        "duels_won": duels_won,
        "dribbles_success": dribbles_success,
        "tackles_total": tackles_total,
        "interceptions": interceptions,
    }


def determine_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    elif home_goals < away_goals:
        return "A"
    return "D"


def is_knockout_round(round_name: str) -> bool:
    patterns = r"(Round of|Quarter|Semi|Final|Play-offs|Preliminary|Qualifying)"
    return bool(re.search(patterns, round_name, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Phase 1 & 2: Build matches base table with stats + player aggregates
# ---------------------------------------------------------------------------

def build_matches_df() -> pd.DataFrame:
    """Load all fixtures, merge statistics and player aggregates."""
    rows = []
    seasons = sorted(EXPLORATION_DIR.iterdir())

    for season_dir in seasons:
        fixtures_file = season_dir / "fixtures.json"
        if not fixtures_file.exists():
            continue

        with open(fixtures_file) as f:
            fixtures_data = json.load(f)

        season_label = season_dir.name

        for match in fixtures_data["response"]:
            status = match["fixture"]["status"]["short"]
            if status != "FT":
                continue

            fixture_id = match["fixture"]["id"]
            home_team_id = match["teams"]["home"]["id"]
            away_team_id = match["teams"]["away"]["id"]

            row = {
                "fixture_id": fixture_id,
                "date": pd.to_datetime(match["fixture"]["date"]),
                "season": season_label,
                "round": match["league"]["round"],
                "home_team_id": home_team_id,
                "home_team_name": match["teams"]["home"]["name"],
                "away_team_id": away_team_id,
                "away_team_name": match["teams"]["away"]["name"],
                "home_goals": match["goals"]["home"],
                "away_goals": match["goals"]["away"],
                "result": determine_result(match["goals"]["home"], match["goals"]["away"]),
            }

            # --- Statistics ---
            stats_file = STATISTICS_DIR / f"{fixture_id}.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats_data = json.load(f)

                for team_entry in stats_data.get("response", []):
                    tid = team_entry["team"]["id"]
                    parsed = parse_team_stats(team_entry["statistics"])
                    if tid == home_team_id:
                        prefix = "home_"
                    elif tid == away_team_id:
                        prefix = "away_"
                    else:
                        continue
                    for k, v in parsed.items():
                        row[prefix + k] = v

            # --- Player aggregates ---
            players_file = PLAYERS_DIR / f"{fixture_id}.json"
            if players_file.exists():
                with open(players_file) as f:
                    players_data = json.load(f)

                for team_entry in players_data.get("response", []):
                    tid = team_entry["team"]["id"]
                    agg = compute_player_aggregates(team_entry["players"])
                    if tid == home_team_id:
                        prefix = "home_"
                    elif tid == away_team_id:
                        prefix = "away_"
                    else:
                        continue
                    for k, v in agg.items():
                        row[prefix + k] = v

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Phase 3: Rolling features
# ---------------------------------------------------------------------------

def compute_rolling_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build team-centric view, compute rolling stats, map back to matches."""

    team_rows = []
    for _, m in matches_df.iterrows():
        fid = m["fixture_id"]
        date = m["date"]

        # Points
        if m["result"] == "H":
            home_pts, away_pts = 3, 0
        elif m["result"] == "A":
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

        # Shots accuracy: shots_on_goal / total_shots
        home_shots_acc = (
            m.get("home_shots_on_goal", 0) / m.get("home_total_shots", 1)
            if m.get("home_total_shots") and m.get("home_total_shots") > 0
            else None
        )
        away_shots_acc = (
            m.get("away_shots_on_goal", 0) / m.get("away_total_shots", 1)
            if m.get("away_total_shots") and m.get("away_total_shots") > 0
            else None
        )

        # Duels won pct
        home_duels_pct = (
            m.get("home_duels_won", 0) / m.get("home_duels_total", 1)
            if m.get("home_duels_total") and m.get("home_duels_total") > 0
            else None
        )
        away_duels_pct = (
            m.get("away_duels_won", 0) / m.get("away_duels_total", 1)
            if m.get("away_duels_total") and m.get("away_duels_total") > 0
            else None
        )

        # Home team row
        team_rows.append({
            "fixture_id": fid,
            "date": date,
            "team_id": m["home_team_id"],
            "is_home": True,
            "goals_for": m["home_goals"],
            "goals_against": m["away_goals"],
            "xg_for": m.get("home_xg"),
            "xg_against": m.get("away_xg"),
            "shots_on_goal": m.get("home_shots_on_goal"),
            "shots_accuracy": home_shots_acc,
            "possession": m.get("home_possession"),
            "pass_accuracy": m.get("home_pass_pct"),
            "corner_kicks": m.get("home_corner_kicks"),
            "avg_rating": m.get("home_avg_rating"),
            "key_passes": m.get("home_key_passes"),
            "duels_won_pct": home_duels_pct,
            "dribbles_success": m.get("home_dribbles_success"),
            "points": home_pts,
        })

        # Away team row
        team_rows.append({
            "fixture_id": fid,
            "date": date,
            "team_id": m["away_team_id"],
            "is_home": False,
            "goals_for": m["away_goals"],
            "goals_against": m["home_goals"],
            "xg_for": m.get("away_xg"),
            "xg_against": m.get("home_xg"),
            "shots_on_goal": m.get("away_shots_on_goal"),
            "shots_accuracy": away_shots_acc,
            "possession": m.get("away_possession"),
            "pass_accuracy": m.get("away_pass_pct"),
            "corner_kicks": m.get("away_corner_kicks"),
            "avg_rating": m.get("away_avg_rating"),
            "key_passes": m.get("away_key_passes"),
            "duels_won_pct": away_duels_pct,
            "dribbles_success": m.get("away_dribbles_success"),
            "points": away_pts,
        })

    team_df = pd.DataFrame(team_rows)
    team_df = team_df.sort_values(["team_id", "date"]).reset_index(drop=True)

    # Derived columns needed for rolling
    team_df["xg_diff"] = team_df["xg_for"] - team_df["xg_against"]
    team_df["xg_overperformance"] = team_df["goals_for"] - team_df["xg_for"]

    # Define which columns to roll and their output names
    roll_cols = {
        "xg_for": "rolling_xg_for",
        "xg_against": "rolling_xg_against",
        "xg_diff": "rolling_xg_diff",
        "goals_for": "rolling_goals_for",
        "goals_against": "rolling_goals_against",
        "xg_overperformance": "rolling_xg_overperformance",
        "shots_on_goal": "rolling_shots_on_goal",
        "shots_accuracy": "rolling_shots_accuracy",
        "possession": "rolling_possession",
        "pass_accuracy": "rolling_pass_accuracy",
        "corner_kicks": "rolling_corners",
        "avg_rating": "rolling_avg_rating",
        "key_passes": "rolling_key_passes",
        "duels_won_pct": "rolling_duels_won_pct",
        "dribbles_success": "rolling_dribbles_success",
        "points": "rolling_points",
    }

    # Compute rolling means (shifted so current match isn't included)
    for src_col, dst_col in roll_cols.items():
        team_df[dst_col] = (
            team_df.groupby("team_id")[src_col]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    # Compute days since last match per team
    team_df["days_since_last"] = (
        team_df.groupby("team_id")["date"]
        .transform(lambda s: s.diff().dt.days)
    )

    # Split back into home and away rolling features
    rolling_feature_cols = list(roll_cols.values()) + ["days_since_last"]

    home_rolling = (
        team_df[team_df["is_home"]][["fixture_id"] + rolling_feature_cols]
        .rename(columns={c: f"home_{c}" for c in rolling_feature_cols})
    )
    away_rolling = (
        team_df[~team_df["is_home"]][["fixture_id"] + rolling_feature_cols]
        .rename(columns={c: f"away_{c}" for c in rolling_feature_cols})
    )

    return home_rolling, away_rolling


# ---------------------------------------------------------------------------
# Phase 4: Build features.csv
# ---------------------------------------------------------------------------

def build_features_df(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Merge rolling features, add derived columns, produce final features CSV."""

    home_rolling, away_rolling = compute_rolling_features(matches_df)

    features = matches_df.merge(home_rolling, on="fixture_id", how="left")
    features = features.merge(away_rolling, on="fixture_id", how="left")

    # Derived diff features (home - away)
    features["diff_rolling_xg"] = features["home_rolling_xg_for"] - features["away_rolling_xg_for"]
    features["diff_rolling_goals"] = features["home_rolling_goals_for"] - features["away_rolling_goals_for"]
    features["diff_rolling_form"] = features["home_rolling_points"] - features["away_rolling_points"]

    # Knockout flag
    features["is_knockout"] = features["round"].apply(is_knockout_round).astype(int)

    return features


# ---------------------------------------------------------------------------
# Phase 5: Summary
# ---------------------------------------------------------------------------

def print_summary(matches_df: pd.DataFrame, features_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    print(f"\nmatches.csv shape: {matches_df.shape}")
    print(f"features.csv shape: {features_df.shape}")

    print(f"\nTarget distribution (result):")
    dist = matches_df["result"].value_counts()
    for label in ["H", "D", "A"]:
        count = dist.get(label, 0)
        pct = count / len(matches_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    # NaN counts in rolling features
    rolling_cols = [c for c in features_df.columns if "rolling_" in c]
    nan_counts = features_df[rolling_cols].isna().sum()
    print(f"\nNaN counts in rolling features:")
    for col, cnt in nan_counts.items():
        if cnt > 0:
            print(f"  {col}: {cnt}")
    if nan_counts.sum() == 0:
        print("  No NaNs in rolling features")

    # Top correlations with numeric result
    result_map = {"H": 1, "D": 0, "A": -1}
    features_df["result_numeric"] = features_df["result"].map(result_map)
    numeric_cols = features_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "result_numeric"]
    correlations = features_df[numeric_cols].corrwith(features_df["result_numeric"]).abs().sort_values(ascending=False)
    print(f"\nTop 10 correlations with result:")
    for col, corr in correlations.head(10).items():
        print(f"  {col}: {corr:.4f}")
    features_df.drop(columns=["result_numeric"], inplace=True)

    # xG descriptive stats
    print(f"\nxG descriptive stats (home):")
    print(features_df["home_xg"].describe().to_string())
    print(f"\nxG descriptive stats (away):")
    print(features_df["away_xg"].describe().to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Phase 1 & 2: Building matches table with stats and player aggregates...")
    matches_df = build_matches_df()
    matches_path = PROCESSED_DIR / "matches.csv"
    matches_df.to_csv(matches_path, index=False)
    print(f"  Saved {matches_path} — {matches_df.shape[0]} rows, {matches_df.shape[1]} columns")

    print("\nPhase 3 & 4: Computing rolling features and building features table...")
    features_df = build_features_df(matches_df)
    features_path = PROCESSED_DIR / "features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"  Saved {features_path} — {features_df.shape[0]} rows, {features_df.shape[1]} columns")

    print_summary(matches_df, features_df)


if __name__ == "__main__":
    main()
