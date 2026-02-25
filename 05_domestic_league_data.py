#!/usr/bin/env python3
"""
05_domestic_league_data.py — Descarga y procesa datos de ligas domésticas
de los equipos de Champions League para agregar features de forma doméstica.

Checkpoint/resume: todos los archivos se verifican antes de descargar.
Rate limit: 6.5s entre llamadas (plan Pro = 10 req/min).

Fases:
  1. Mapear equipos → liga doméstica principal
  2. Descargar fixtures de ligas domésticas
  3. Descargar statistics de fixtures donde juegan equipos CL
  4. Construir features domésticas por equipo/fecha
  5. Merge con features.csv → features_v2.csv
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Configuración ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_FOOTBALL_KEY")
if not API_KEY:
    sys.exit("ERROR: La variable de entorno API_FOOTBALL_KEY no está definida.")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASONS = [2023, 2024, 2025]
SLEEP = 6.5

BASE_DIR = Path(__file__).resolve().parent
EXPLORATION_DIR = BASE_DIR / "data" / "raw" / "exploration"
DOMESTIC_DIR = BASE_DIR / "data" / "raw" / "domestic"
DOMESTIC_STATS_DIR = DOMESTIC_DIR / "statistics"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Known top-tier league IDs (for prioritization when team plays in multiple)
TOP_LEAGUES = {
    39,   # Premier League
    140,  # La Liga
    78,   # Bundesliga
    135,  # Serie A
    61,   # Ligue 1
    88,   # Eredivisie
    94,   # Liga Portugal
    144,  # Belgian Pro League
    203,  # Super Lig (Turkey)
    218,  # Tipp3 Bundesliga (Austria)
    235,  # Premiership (Scotland)
    179,  # Scottish Premiership
    119,  # Superliga (Denmark)
    113,  # Allsvenskan (Sweden)
    106,  # Ekstraklasa (Poland)
    197,  # Super League (Greece)
    283,  # 1. HNL (Croatia)
    210,  # Super Liga (Serbia)
    333,  # Premier Division (Ireland)
    239,  # Tipico Bundesliga (Austria)
    103,  # Eliteserien (Norway)
    271,  # Meistriliiga (Estonia)
    286,  # Superliga (Albania)
    373,  # Premyer Liqa (Azerbaijan)
    332,  # Premiership (N. Ireland)
    207,  # Swiss Super League
    318,  # Cymru Premier (Wales)
    188,  # Virsliga (Latvia)
    371,  # Erovnuli Liga (Georgia)
    262,  # Liga 1 (Romania)
    345,  # Czech First League
    172,  # Fortuna Liga (Slovakia)
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def sep(title: str, char: str = "═", width: int = 90) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def api_get(endpoint: str, params: dict) -> dict | None:
    """Llama a la API con rate limit. Devuelve JSON o None si falla."""
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        errors = data.get("errors")
        if errors and (isinstance(errors, list) and errors or isinstance(errors, dict) and errors):
            print(f"    API error: {errors}")
            return None
        return data
    except requests.RequestException as e:
        print(f"    Request error: {e}")
        return None


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_exists_and_valid(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < 20:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        resp = data.get("response")
        return isinstance(resp, list) and len(resp) > 0
    except (json.JSONDecodeError, OSError):
        return False


def load_teams_by_season() -> dict[int, list[dict]]:
    """Load CL teams from exploration JSONs. Returns {season: [{id, name, country}, ...]}."""
    teams_by_season = {}
    for season in SEASONS:
        path = EXPLORATION_DIR / str(season) / "teams.json"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = []
        for entry in data.get("response", []):
            t = entry["team"]
            teams.append({"id": t["id"], "name": t["name"], "country": t.get("country", "")})
        teams_by_season[season] = teams
    return teams_by_season


def load_cl_fixtures() -> pd.DataFrame:
    """Load all CL fixtures from features.csv for date reference."""
    return pd.read_csv(PROCESSED_DIR / "features.csv")


# ── FASE 1: Mapear equipos a ligas domésticas ────────────────────────────────

def phase1_team_leagues(teams_by_season: dict) -> dict:
    """
    For each (team, season), find their primary domestic league.
    Returns {team_id: {season: {league_id, league_name, country}}}.
    Saves/loads checkpoint from data/processed/team_leagues.json.
    """
    sep("FASE 1: Mapear equipos a ligas domésticas")

    checkpoint_path = PROCESSED_DIR / "team_leagues.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        print(f"  Checkpoint cargado: {len(mapping)} equipos mapeados")
    else:
        mapping = {}

    # Build list of (team_id, season) combos still needed
    needed = []
    for season, teams in teams_by_season.items():
        for team in teams:
            tid = str(team["id"])
            if tid not in mapping:
                mapping[tid] = {"name": team["name"], "country": team["country"], "seasons": {}}
            if str(season) not in mapping[tid].get("seasons", {}):
                needed.append((team["id"], team["name"], season))

    if not needed:
        print("  Todos los equipos ya están mapeados.")
        return mapping

    print(f"  Equipos por mapear: {len(needed)}")
    print(f"  Tiempo estimado: {fmt_time(len(needed) * SLEEP)}")

    api_calls = 0
    for i, (team_id, team_name, season) in enumerate(needed):
        print(f"  [{i+1}/{len(needed)}] {team_name} (id={team_id}, season={season})...", end="", flush=True)

        time.sleep(SLEEP)
        data = api_get("/leagues", {"team": team_id, "season": season, "type": "League"})
        api_calls += 1

        tid = str(team_id)
        if tid not in mapping:
            mapping[tid] = {"name": team_name, "country": "", "seasons": {}}

        if data is None or not data.get("response"):
            print(" no league found")
            mapping[tid]["seasons"][str(season)] = None
        else:
            leagues = data["response"]
            # Pick the best league: prefer known top leagues, then first result
            best = None
            for league_entry in leagues:
                lg = league_entry["league"]
                if lg["id"] in TOP_LEAGUES:
                    best = {"league_id": lg["id"], "league_name": lg["name"], "country": league_entry.get("country", {}).get("name", "")}
                    break
            if best is None and leagues:
                lg = leagues[0]["league"]
                best = {"league_id": lg["id"], "league_name": lg["name"], "country": leagues[0].get("country", {}).get("name", "")}

            mapping[tid]["seasons"][str(season)] = best
            if best:
                print(f" -> {best['league_name']} (id={best['league_id']})")
            else:
                print(" no league found")

        # Save checkpoint every 20 calls
        if api_calls % 20 == 0:
            save_json(mapping, checkpoint_path)

    save_json(mapping, checkpoint_path)
    print(f"\n  Fase 1 completada. {api_calls} llamadas API. Guardado en {checkpoint_path}")
    return mapping


# ── FASE 2: Descargar fixtures de ligas domésticas ───────────────────────────

def phase2_domestic_fixtures(mapping: dict) -> dict[str, Path]:
    """
    Download all fixtures for each domestic league-season combo.
    Returns {league_season_key: filepath}.
    """
    sep("FASE 2: Descargar fixtures de ligas domésticas")
    DOMESTIC_DIR.mkdir(parents=True, exist_ok=True)

    # Collect unique (league_id, season) combos
    league_seasons = set()
    for tid, info in mapping.items():
        for season_str, league_info in info.get("seasons", {}).items():
            if league_info and "league_id" in league_info:
                league_seasons.add((league_info["league_id"], int(season_str)))

    print(f"  Ligas domésticas únicas (liga, temporada): {len(league_seasons)}")

    # Check which already exist
    needed = []
    paths = {}
    for league_id, season in sorted(league_seasons):
        key = f"{league_id}_{season}"
        path = DOMESTIC_DIR / f"{key}_fixtures.json"
        paths[key] = path
        if file_exists_and_valid(path):
            continue
        needed.append((league_id, season, key, path))

    already = len(league_seasons) - len(needed)
    print(f"  Ya descargados: {already}")
    print(f"  Por descargar: {len(needed)}")

    if needed:
        print(f"  Tiempo estimado: {fmt_time(len(needed) * SLEEP)}")

    api_calls = 0
    for i, (league_id, season, key, path) in enumerate(needed):
        print(f"  [{i+1}/{len(needed)}] Liga {league_id}, temporada {season}...", end="", flush=True)

        time.sleep(SLEEP)
        data = api_get("/fixtures", {"league": league_id, "season": season, "status": "FT"})
        api_calls += 1

        if data and data.get("response"):
            save_json(data, path)
            n = len(data["response"])
            print(f" {n} fixtures")
        else:
            # Save empty response so we don't retry
            save_json({"response": []}, path)
            print(" 0 fixtures")

    print(f"\n  Fase 2 completada. {api_calls} llamadas API.")
    return paths


# ── FASE 3: Descargar statistics de fixtures con equipos CL ──────────────────

def phase3_domestic_statistics(mapping: dict) -> int:
    """
    Download statistics for domestic fixtures involving CL teams only.
    """
    sep("FASE 3: Descargar statistics de fixtures domésticos (solo equipos CL)")
    DOMESTIC_STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Build set of CL team IDs per season
    cl_teams_by_season = {}
    for tid, info in mapping.items():
        for season_str, league_info in info.get("seasons", {}).items():
            season = int(season_str)
            if season not in cl_teams_by_season:
                cl_teams_by_season[season] = set()
            cl_teams_by_season[season].add(int(tid))

    # Collect unique (league_id, season) combos
    league_seasons = set()
    for tid, info in mapping.items():
        for season_str, league_info in info.get("seasons", {}).items():
            if league_info and "league_id" in league_info:
                league_seasons.add((league_info["league_id"], int(season_str)))

    # For each league-season, load fixtures, filter for CL teams, download stats
    fixtures_to_download = []
    for league_id, season in sorted(league_seasons):
        key = f"{league_id}_{season}"
        path = DOMESTIC_DIR / f"{key}_fixtures.json"
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cl_teams = cl_teams_by_season.get(season, set())

        for fix in data.get("response", []):
            fid = fix["fixture"]["id"]
            home_id = fix["teams"]["home"]["id"]
            away_id = fix["teams"]["away"]["id"]

            if home_id in cl_teams or away_id in cl_teams:
                stats_path = DOMESTIC_STATS_DIR / f"{fid}.json"
                if not file_exists_and_valid(stats_path):
                    fixtures_to_download.append((fid, stats_path))

    print(f"  Fixtures domésticos de equipos CL que necesitan statistics: {len(fixtures_to_download)}")
    if fixtures_to_download:
        print(f"  Tiempo estimado: {fmt_time(len(fixtures_to_download) * SLEEP)}")

    api_calls = 0
    for i, (fid, stats_path) in enumerate(fixtures_to_download):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(fixtures_to_download)}] descargando...", flush=True)

        time.sleep(SLEEP)
        data = api_get("/fixtures/statistics", {"fixture": fid})
        api_calls += 1

        if data and data.get("response"):
            save_json(data, stats_path)
        else:
            save_json({"response": []}, stats_path)

    print(f"\n  Fase 3 completada. {api_calls} llamadas API.")
    return api_calls


# ── FASE 4: Construir features domésticas ────────────────────────────────────

def parse_stat_value(stat_type: str, value):
    """Parse a stat value from the API format."""
    if value is None:
        return None
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


def build_domestic_matches_df(mapping: dict) -> pd.DataFrame:
    """
    Build a DataFrame of all domestic matches involving CL teams,
    with parsed statistics.
    """
    print("  Construyendo tabla de partidos domésticos...")

    # Collect unique (league_id, season) and CL team sets
    league_seasons = set()
    cl_teams_by_season = {}
    for tid, info in mapping.items():
        for season_str, league_info in info.get("seasons", {}).items():
            season = int(season_str)
            if league_info and "league_id" in league_info:
                league_seasons.add((league_info["league_id"], season))
            if season not in cl_teams_by_season:
                cl_teams_by_season[season] = set()
            cl_teams_by_season[season].add(int(tid))

    rows = []
    for league_id, season in sorted(league_seasons):
        key = f"{league_id}_{season}"
        path = DOMESTIC_DIR / f"{key}_fixtures.json"
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for fix in data.get("response", []):
            fid = fix["fixture"]["id"]
            home_id = fix["teams"]["home"]["id"]
            away_id = fix["teams"]["away"]["id"]
            home_goals = fix["goals"]["home"]
            away_goals = fix["goals"]["away"]

            if home_goals is None or away_goals is None:
                continue

            row = {
                "fixture_id": fid,
                "date": pd.to_datetime(fix["fixture"]["date"]),
                "league_id": league_id,
                "season": season,
                "home_team_id": home_id,
                "home_team_name": fix["teams"]["home"]["name"],
                "away_team_id": away_id,
                "away_team_name": fix["teams"]["away"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
            }

            # Parse statistics if available
            stats_path = DOMESTIC_STATS_DIR / f"{fid}.json"
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as sf:
                    stats_data = json.load(sf)

                for team_entry in stats_data.get("response", []):
                    tid = team_entry["team"]["id"]
                    if tid == home_id:
                        prefix = "home_"
                    elif tid == away_id:
                        prefix = "away_"
                    else:
                        continue
                    for stat in team_entry.get("statistics", []):
                        if stat["type"] == "expected_goals":
                            row[f"{prefix}xg"] = parse_stat_value("expected_goals", stat["value"])

            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("date").reset_index(drop=True)
    print(f"  Total partidos domésticos cargados: {len(df)}")
    return df


def compute_league_table(matches: pd.DataFrame, team_id: int, before_date) -> dict | None:
    """
    Compute league standings for a team using matches before the given date.
    Returns dict with position, ppg, win_rate, etc.
    """
    # All matches in this league-season before the date
    before = matches[matches["date"] < before_date]
    if len(before) == 0:
        return None

    # Build standings for all teams in the league
    standings = {}
    for _, m in before.iterrows():
        hid = m["home_team_id"]
        aid = m["away_team_id"]

        if hid not in standings:
            standings[hid] = {"pts": 0, "played": 0, "gd": 0}
        if aid not in standings:
            standings[aid] = {"pts": 0, "played": 0, "gd": 0}

        hg = m["home_goals"]
        ag = m["away_goals"]

        standings[hid]["played"] += 1
        standings[aid]["played"] += 1
        standings[hid]["gd"] += hg - ag
        standings[aid]["gd"] += ag - hg

        if hg > ag:
            standings[hid]["pts"] += 3
        elif hg < ag:
            standings[aid]["pts"] += 3
        else:
            standings[hid]["pts"] += 1
            standings[aid]["pts"] += 1

    # Sort by points then goal difference
    sorted_teams = sorted(standings.keys(), key=lambda t: (standings[t]["pts"], standings[t]["gd"]), reverse=True)

    if team_id not in standings:
        return None

    position = sorted_teams.index(team_id) + 1
    played = standings[team_id]["played"]
    pts = standings[team_id]["pts"]

    return {
        "position": position,
        "ppg": pts / played if played > 0 else 0,
        "played": played,
    }


def compute_domestic_features_for_team(
    domestic_df: pd.DataFrame,
    team_id: int,
    before_date,
    league_id: int,
    season: int,
    n: int = 5,
) -> dict:
    """
    Compute domestic features for a team using league matches before the given date.
    """
    result = {
        "domestic_points_last5": np.nan,
        "domestic_goals_for_last5": np.nan,
        "domestic_goals_against_last5": np.nan,
        "domestic_xg_for_last5": np.nan,
        "domestic_xg_against_last5": np.nan,
        "domestic_league_position": np.nan,
        "domestic_ppg": np.nan,
        "domestic_win_rate": np.nan,
        "domestic_home_goals_avg": np.nan,
        "domestic_away_goals_avg": np.nan,
    }

    # Filter domestic matches for this league+season
    league_matches = domestic_df[
        (domestic_df["league_id"] == league_id) & (domestic_df["season"] == season)
    ].copy()

    if len(league_matches) == 0:
        return result

    # Team matches before the CL date
    team_home = league_matches[
        (league_matches["home_team_id"] == team_id) & (league_matches["date"] < before_date)
    ]
    team_away = league_matches[
        (league_matches["away_team_id"] == team_id) & (league_matches["date"] < before_date)
    ]

    # Build team-centric rows
    team_rows = []
    for _, m in team_home.iterrows():
        team_rows.append({
            "date": m["date"],
            "goals_for": m["home_goals"],
            "goals_against": m["away_goals"],
            "xg_for": m.get("home_xg"),
            "xg_against": m.get("away_xg"),
            "is_home": True,
            "points": 3 if m["home_goals"] > m["away_goals"] else (1 if m["home_goals"] == m["away_goals"] else 0),
            "win": 1 if m["home_goals"] > m["away_goals"] else 0,
        })
    for _, m in team_away.iterrows():
        team_rows.append({
            "date": m["date"],
            "goals_for": m["away_goals"],
            "goals_against": m["home_goals"],
            "xg_for": m.get("away_xg"),
            "xg_against": m.get("home_xg"),
            "is_home": False,
            "points": 3 if m["away_goals"] > m["home_goals"] else (1 if m["away_goals"] == m["home_goals"] else 0),
            "win": 1 if m["away_goals"] > m["home_goals"] else 0,
        })

    if not team_rows:
        return result

    team_df = pd.DataFrame(team_rows).sort_values("date")

    # Season-to-date stats
    total_played = len(team_df)
    total_pts = team_df["points"].sum()
    total_wins = team_df["win"].sum()

    result["domestic_ppg"] = total_pts / total_played
    result["domestic_win_rate"] = total_wins / total_played

    # Home/away goal averages
    home_matches = team_df[team_df["is_home"]]
    away_matches = team_df[~team_df["is_home"]]
    if len(home_matches) > 0:
        result["domestic_home_goals_avg"] = home_matches["goals_for"].mean()
    if len(away_matches) > 0:
        result["domestic_away_goals_avg"] = away_matches["goals_for"].mean()

    # Last N matches
    last_n = team_df.tail(n)
    result["domestic_points_last5"] = last_n["points"].sum()
    result["domestic_goals_for_last5"] = last_n["goals_for"].mean()
    result["domestic_goals_against_last5"] = last_n["goals_against"].mean()

    # xG last N (may have NaN)
    xg_for = last_n["xg_for"].dropna()
    xg_against = last_n["xg_against"].dropna()
    if len(xg_for) > 0:
        result["domestic_xg_for_last5"] = xg_for.mean()
    if len(xg_against) > 0:
        result["domestic_xg_against_last5"] = xg_against.mean()

    # League position
    table_info = compute_league_table(league_matches, team_id, before_date)
    if table_info:
        result["domestic_league_position"] = table_info["position"]

    return result


def phase4_build_domestic_features(mapping: dict, domestic_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each CL match, compute domestic features for home and away teams.
    Returns a DataFrame with fixture_id + domestic features.
    """
    sep("FASE 4: Construir features domésticas")

    features_df = load_cl_fixtures()
    features_df["date"] = pd.to_datetime(features_df["date"])

    print(f"  Partidos CL: {len(features_df)}")
    print(f"  Partidos domésticos disponibles: {len(domestic_df)}")

    # Build team->league mapping for quick lookup
    team_league = {}  # (team_id, season) -> (league_id)
    for tid, info in mapping.items():
        for season_str, league_info in info.get("seasons", {}).items():
            if league_info and "league_id" in league_info:
                team_league[(int(tid), int(season_str))] = league_info["league_id"]

    domestic_rows = []
    total = len(features_df)
    for i, (_, match) in enumerate(features_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Procesando partido {i+1}/{total}...")

        fid = match["fixture_id"]
        cl_date = match["date"]
        season = match["season"]
        home_tid = match["home_team_id"]
        away_tid = match["away_team_id"]

        row = {"fixture_id": fid}

        # Home team domestic features
        home_league = team_league.get((home_tid, season))
        if home_league:
            home_feats = compute_domestic_features_for_team(
                domestic_df, home_tid, cl_date, home_league, season
            )
            for k, v in home_feats.items():
                row[f"home_{k}"] = v
        else:
            for k in [
                "domestic_points_last5", "domestic_goals_for_last5",
                "domestic_goals_against_last5", "domestic_xg_for_last5",
                "domestic_xg_against_last5", "domestic_league_position",
                "domestic_ppg", "domestic_win_rate",
                "domestic_home_goals_avg", "domestic_away_goals_avg",
            ]:
                row[f"home_{k}"] = np.nan

        # Away team domestic features
        away_league = team_league.get((away_tid, season))
        if away_league:
            away_feats = compute_domestic_features_for_team(
                domestic_df, away_tid, cl_date, away_league, season
            )
            for k, v in away_feats.items():
                row[f"away_{k}"] = v
        else:
            for k in [
                "domestic_points_last5", "domestic_goals_for_last5",
                "domestic_goals_against_last5", "domestic_xg_for_last5",
                "domestic_xg_against_last5", "domestic_league_position",
                "domestic_ppg", "domestic_win_rate",
                "domestic_home_goals_avg", "domestic_away_goals_avg",
            ]:
                row[f"away_{k}"] = np.nan

        domestic_rows.append(row)

    return pd.DataFrame(domestic_rows)


# ── FASE 5: Merge con features existentes ────────────────────────────────────

def phase5_merge_and_save(domestic_features_df: pd.DataFrame):
    """Merge domestic features into features.csv and save as features_v2.csv."""
    sep("FASE 5: Merge con features existentes")

    features_df = load_cl_fixtures()
    print(f"  features.csv: {features_df.shape}")

    merged = features_df.merge(domestic_features_df, on="fixture_id", how="left")

    # Derived feature: position difference (positive = home better positioned, i.e., lower number)
    merged["diff_domestic_position"] = (
        merged["away_domestic_league_position"] - merged["home_domestic_league_position"]
    )

    output_path = PROCESSED_DIR / "features_v2.csv"
    merged.to_csv(output_path, index=False)
    print(f"  features_v2.csv guardado: {merged.shape}")

    # Stats
    domestic_cols = [c for c in merged.columns if "domestic_" in c]
    has_domestic = merged[domestic_cols].notna().any(axis=1).sum()
    missing_domestic = len(merged) - has_domestic
    print(f"\n  Partidos CON features domésticas: {has_domestic}/{len(merged)} ({has_domestic/len(merged)*100:.1f}%)")
    print(f"  Partidos SIN features domésticas: {missing_domestic}/{len(merged)} ({missing_domestic/len(merged)*100:.1f}%)")

    # NaN report
    print(f"\n  NaN por feature doméstica:")
    for col in sorted(domestic_cols):
        nan_count = merged[col].isna().sum()
        pct = nan_count / len(merged) * 100
        print(f"    {col:<50s} {nan_count:>4d} ({pct:.1f}%)")

    # Correlations with result
    result_map = {"H": 1, "D": 0, "A": -1}
    merged["result_numeric"] = merged["result"].map(result_map)
    print(f"\n  Correlaciones de features domésticas con resultado:")
    for col in sorted(domestic_cols):
        if merged[col].notna().sum() > 50:
            corr = merged[col].corr(merged["result_numeric"])
            if not np.isnan(corr):
                print(f"    {col:<50s} {corr:+.4f}")

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sep("DOMESTIC LEAGUE DATA PIPELINE", char="█")

    # Estimate total work
    teams_by_season = load_teams_by_season()
    total_combos = sum(len(teams) for teams in teams_by_season.values())
    print(f"\n  Equipos por temporada: {', '.join(f'{s}: {len(t)}' for s, t in teams_by_season.items())}")
    print(f"  Total (equipo, temporada) combos: {total_combos}")
    print(f"\n  Estimación de llamadas API:")
    print(f"    Fase 1 (mapeo ligas):     hasta {total_combos} llamadas")
    print(f"    Fase 2 (fixtures ligas):  hasta ~50-60 llamadas")
    print(f"    Fase 3 (statistics):      hasta ~2000-3000 llamadas")
    print(f"    Total estimado:           ~{total_combos + 55 + 2500} llamadas")
    print(f"    Tiempo estimado:          ~{fmt_time((total_combos + 55 + 2500) * SLEEP)}")
    print(f"\n  (Los checkpoints reducen esto si se re-ejecuta)")

    # Phase 1
    mapping = phase1_team_leagues(teams_by_season)

    # Phase 2
    phase2_domestic_fixtures(mapping)

    # Phase 3
    phase3_domestic_statistics(mapping)

    # Phase 4
    domestic_df = build_domestic_matches_df(mapping)

    domestic_features_df = phase4_build_domestic_features(mapping, domestic_df)

    # Phase 5
    phase5_merge_and_save(domestic_features_df)

    sep("PIPELINE COMPLETADO", char="█")


if __name__ == "__main__":
    main()
