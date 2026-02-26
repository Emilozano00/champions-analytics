#!/usr/bin/env python3
"""
09_europa_league_data.py — Expand dataset with Europa League + Conference League.

Hypothesis: more European competition data (same teams, similar structure)
will improve Champions League prediction accuracy.

Usage:
  python3 09_europa_league_data.py            # Phase 1 only (exploration, 6 API calls)
  python3 09_europa_league_data.py --download  # All 6 phases

Phases:
  1. Exploration: download & count fixtures for EL + ECL
  2. Download statistics + players for all new fixtures
  3. Unified feature engineering (CL + EL + ECL)
  4. ELO integration
  5. Domestic features (pragmatic: reuse existing, NaN for new teams)
  6. Re-train and compare (3 variants)
"""

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
SLEEP = 6.5
SEASONS = [2023, 2024, 2025]

LEAGUES = {
    2:   {"name": "Champions League",   "dir": "exploration", "level": 3},
    3:   {"name": "Europa League",       "dir": "europa",      "level": 2},
    848: {"name": "Conference League",   "dir": "conference",  "level": 1},
}

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
STATS_DIR = RAW_DIR / "statistics"
PLAYERS_DIR = RAW_DIR / "players"
ELO_DIR = RAW_DIR / "elo"
DOMESTIC_DIR = RAW_DIR / "domestic"
DOMESTIC_STATS_DIR = DOMESTIC_DIR / "statistics"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

CLASSES = ["A", "D", "H"]
FINAL_ACCURACY = 0.5721
FINAL_LOG_LOSS = 0.9331

# Final model's 20 features
FINAL_FEATURES = [
    "home_rolling_points", "elo_diff", "home_domestic_home_goals_avg",
    "elo_home", "elo_away", "away_domestic_xg_against_last5",
    "home_rolling_shots_accuracy", "home_rolling_avg_rating",
    "home_rolling_corners", "home_rolling_duels_won_pct",
    "away_rolling_duels_won_pct", "away_rolling_xg_overperformance",
    "home_rolling_key_passes", "home_rolling_pass_accuracy",
    "away_rolling_goals_for", "home_rolling_xg_diff",
    "away_domestic_away_goals_avg", "away_rolling_corners",
    "home_days_since_last", "away_rolling_points",
]

# Stat name mapping (same as 03_feature_engineering.py)
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

# ELO name mapping (from 07_elo_ratings.py + new teams for EL/ECL)
NAME_MAP = {
    # Premier League
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle": "Newcastle", "Arsenal": "Arsenal", "Liverpool": "Liverpool",
    "Chelsea": "Chelsea", "Tottenham": "Tottenham", "Aston Villa": "Aston Villa",
    "West Ham": "West Ham", "Brighton": "Brighton",
    "Wolverhampton Wanderers": "Wolverhampton", "Everton": "Everton",
    # La Liga
    "Real Madrid": "Real Madrid", "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico", "Real Sociedad": "Sociedad",
    "Sevilla": "Sevilla", "Villarreal": "Villarreal", "Girona": "Girona",
    "Athletic Club": "Bilbao", "Real Betis": "Betis", "Rayo Vallecano": "Vallecano",
    # Bundesliga
    "Bayern München": "Bayern", "Bayern Munich": "Bayern",
    "Borussia Dortmund": "Dortmund", "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Frankfurt", "Bayer Leverkusen": "Leverkusen",
    "Union Berlin": "Union Berlin", "VfB Stuttgart": "Stuttgart",
    "SC Freiburg": "Freiburg", "TSG Hoffenheim": "Hoffenheim",
    "Borussia Monchengladbach": "Gladbach", "FC Augsburg": "Augsburg",
    "1. FC Heidenheim 1846": "Heidenheim",
    # Serie A
    "Inter": "Inter", "AC Milan": "AC Milan", "Napoli": "Napoli",
    "Juventus": "Juventus", "Atalanta": "Atalanta", "Lazio": "Lazio",
    "Bologna": "Bologna", "AS Roma": "AS Roma", "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    # Ligue 1
    "Paris Saint Germain": "Paris SG", "Marseille": "Marseille",
    "Lens": "Lens", "Monaco": "Monaco", "Lille": "Lille", "Nice": "Nice",
    "Stade Brestois 29": "Brest", "Lyon": "Lyon", "Rennes": "Rennes",
    "Toulouse": "Toulouse", "Stade Reims": "Reims", "Nantes": "Nantes",
    "Montpellier": "Montpellier", "Strasbourg": "Strasbourg",
    # Portugal
    "Benfica": "Benfica", "FC Porto": "Porto", "Sporting CP": "Sporting",
    "SC Braga": "Braga", "Vitoria Guimaraes": "Guimaraes",
    # Netherlands
    "PSV Eindhoven": "PSV", "Ajax": "Ajax", "Feyenoord": "Feyenoord",
    "Twente": "Twente", "AZ Alkmaar": "AZ",
    # Belgium
    "Club Brugge KV": "Brugge", "Antwerp": "Antwerp", "Genk": "Genk",
    "Union St. Gilloise": "Union SG", "RSC Anderlecht": "Anderlecht",
    "Gent": "Gent", "Cercle Brugge": "Cercle Brugge",
    # Turkey
    "Galatasaray": "Galatasaray", "Fenerbahçe": "Fenerbahce",
    "Besiktas": "Besiktas", "Istanbul Basaksehir": "Basaksehir",
    "Trabzonspor": "Trabzonspor",
    # Scotland
    "Celtic": "Celtic", "Rangers": "Rangers",
    "Heart Of Midlothian": "Hearts", "Aberdeen": "Aberdeen",
    # Austria
    "Red Bull Salzburg": "Salzburg", "Sturm Graz": "Sturm Graz",
    "SK Rapid Wien": "Rapid Wien", "LASK": "LASK",
    "Wolfsberger AC": "Wolfsberg",
    # Switzerland
    "BSC Young Boys": "Young Boys", "FC Basel 1893": "Basel",
    "FC Lugano": "Lugano", "Servette FC": "Servette",
    "FC Zurich": "Zuerich", "FC St. Gallen": "St. Gallen",
    # Czech Republic
    "Sparta Praha": "Sparta Praha", "Slavia Praha": "Slavia Praha",
    "Plzen": "Viktoria Plzen", "FK Mlada Boleslav": "Mlada Boleslav",
    # Denmark
    "FC Copenhagen": "Koebenhavn", "FC Midtjylland": "Midtjylland",
    "Aarhus": "AGF", "Silkeborg IF": "Silkeborg",
    "FC Nordsjaelland": "Nordsjaelland",
    # Sweden
    "Malmo FF": "Malmoe", "BK Hacken": "Hacken",
    "IF Elfsborg": "Elfsborg", "Djurgarden": "Djurgarden",
    # Norway
    "Molde": "Molde", "Bodo/Glimt": "Bodoe Glimt",
    "Brann": "Brann", "Rosenborg": "Rosenborg",
    # Greece
    "Olympiakos Piraeus": "Olympiakos", "PAOK": "PAOK",
    "Panathinaikos": "Panathinaikos", "AEK Athens FC": "AEK",
    "Aris": "Aris", "Asteras Tripolis": "Asteras Tripolis",
    # Croatia
    "Dinamo Zagreb": "Dinamo Zagreb", "HNK Rijeka": "Rijeka",
    "Hajduk Split": "Hajduk Split", "NK Osijek": "Osijek",
    # Serbia
    "FK Crvena Zvezda": "Crvena Zvezda", "FK Partizan": "Partizan",
    "TSC Backa Topola": "Backa Topola",
    # Ukraine
    "Shakhtar Donetsk": "Shakhtar", "Dynamo Kyiv": "Dynamo Kyiv",
    "Dnipro-1": "Dnipro", "Zorya Luhansk": "Zorya",
    # Romania
    "FCSB": "FCSB", "Farul Constanta": "Farul",
    "CFR 1907 Cluj": "CFR Cluj",
    # Hungary
    "Ferencvarosi TC": "Ferencvaros", "Puskas Akademia FC": "Felcsut",
    # Cyprus
    "Apoel Nicosia": "APOEL", "Pafos": "Paphos",
    "AEK Larnaca": "AEK Larnaca", "Omonia Nicosia": "Omonia",
    # Poland
    "Lech Poznan": "Lech", "Raków Częstochowa": "Rakow",
    "Jagiellonia": "Jagiellonia", "Legia Warszawa": "Legia",
    # Bulgaria
    "Ludogorets": "Razgrad", "CSKA Sofia 1948": "CSKA 1948",
    # Israel
    "Maccabi Haifa": "Maccabi Haifa", "Maccabi Tel Aviv": "Maccabi Tel-Aviv",
    # Slovakia
    "Slovan Bratislava": "Slovan Bratislava",
    # Kazakhstan
    "FC Astana": "FK Astana", "Ordabasy": "Ordabasy",
    "Kairat Almaty": "Kairat",
    # Moldova
    "Sheriff Tiraspol": "Sheriff Tiraspol",
    "Milsami Orhei": "Milsami Orhei", "Petrocub": "Petrocub",
    # Finland
    "HJK helsinki": "HJK Helsinki", "KuPS": "Kuopio",
    # Ireland
    "Shamrock Rovers": "Shamrock Rovers", "Shelbourne": "Shelbourne",
    # Northern Ireland
    "Linfield": "Linfield", "Larne": "Larne",
    # Iceland
    "Breidablik": "Breidablik", "Vikingur Reykjavik": "Vikingur",
    # Faroe Islands
    "KI Klaksvik": "Klaksvik", "Vikingur Gota": "Vikingur Gota",
    # North Macedonia
    "Shkendija": "Shkendija", "Struga": "Struga",
    # Albania
    "Partizani": "Partizani Tirana",
    "Egnatia Rrogozhinë": "Egnatia",
    # Kosovo
    "Drita": "Drita", "Ballkani": "Ballkani",
    # Bosnia
    "Zrinjski": "Zrinjski Mostar",
    "Borac Banja Luka": "Borac Banja Luka",
    # Lithuania
    "FK Zalgiris Vilnius": "Zalgiris", "Panevėžys": "Panevezys",
    # Estonia
    "Flora Tallinn": "Flora Tallinn", "FC Levadia Tallinn": "Levadia",
    # Latvia
    "Rīgas FS": "Rigas", "Valmiera / BSS": "Valmiera",
    # Georgia
    "Dinamo Tbilisi": "Dinamo Tbilisi", "Dinamo Batumi": "Batumi",
    "Saburtalo": "Saburtalo", "FC Noah": "Noah",
    # Armenia
    "Pyunik Yerevan": "Pyunik", "FC Urartu": "Urartu",
    # Montenegro
    "Buducnost Podgorica": "Buducnost", "Dečić": "Decic",
    # Belarus
    "Bate Borisov": "BATE", "Dinamo Minsk": "Dinamo Minsk",
    # Slovenia
    "Olimpija Ljubljana": "Olimpija Ljubljana", "Celje": "Celje",
    # San Marino
    "Tre Penne": "Tre Penne", "Virtus": "SS Virtus",
    # Malta
    "Hamrun Spartans": "Hamrun",
    # Gibraltar
    "Lincoln Red Imps FC": "Lincoln",
    # Andorra
    "UE Santa Coloma": "Santa Coloma",
    "Atlètic Club d'Escaldes": "Atletic Club Escaldes",
    "Inter Club d'Escaldes": "Escaldes",
    # Luxembourg
    "FC Differdange 03": "Differdange",
    "Swift Hesperange": "Swift Hesperange",
    # Wales
    "The New Saints": "The New Saints",
    # Extra EL/ECL teams
    "Stade Rennais": "Rennes",
    "Qarabag": "Qarabag",
    "Slovan Liberec": "Slovan Liberec",
    "FC Sheriff": "Sheriff Tiraspol",
    "Maccabi Netanya": "Maccabi Netanya",
    "Flora": "Flora Tallinn",
    "Legia Warsaw": "Legia",
    "Steaua Bucuresti": "FCSB",
    "FC Twente": "Twente",
    "FC Heidenheim": "Heidenheim",
    "Vitoria SC": "Guimaraes",
    "Anderlecht": "Anderlecht",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(title, char="=", width=70):
    print(f"\n{char * width}\n{title}\n{char * width}")


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def api_get(endpoint, params):
    """Call API-Football with rate limit. Returns JSON or None."""
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        errors = data.get("errors")
        if errors and (isinstance(errors, list) and errors or isinstance(errors, dict) and errors):
            return None
        return data
    except requests.RequestException:
        return None


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_exists_and_valid(path):
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


def parse_stat_value(stat_type, value):
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


def parse_team_stats(statistics_list):
    result = {}
    for stat in statistics_list:
        col_name = STAT_NAME_MAP.get(stat["type"])
        if col_name:
            result[col_name] = parse_stat_value(stat["type"], stat["value"])
    return result


def compute_player_aggregates(players_list):
    ratings, key_passes, duels_total, duels_won = [], 0, 0, 0
    dribbles_success, tackles_total, interceptions = 0, 0, 0
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


def determine_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    elif home_goals < away_goals:
        return "A"
    return "D"


def is_knockout_round(round_name):
    patterns = r"(Round of|Quarter|Semi|Final|Play-offs|Preliminary|Qualifying|Knockout)"
    return bool(re.search(patterns, str(round_name), re.IGNORECASE))


# ===========================================================================
# Phase 1: Exploration
# ===========================================================================

def phase1_exploration():
    """Download fixture lists for EL and ECL. Always runs (6 API calls)."""
    sep("FASE 1: Exploración de Europa League + Conference League")

    if not API_KEY:
        sys.exit("ERROR: API_FOOTBALL_KEY no definida.")

    summary = {}  # {league_id: {season: {total, ft, other}}}

    for league_id in [3, 848]:
        info = LEAGUES[league_id]
        summary[league_id] = {}

        for season in SEASONS:
            out_dir = RAW_DIR / info["dir"] / str(season)
            out_path = out_dir / "fixtures.json"

            # Check if already downloaded
            if file_exists_and_valid(out_path):
                with open(out_path) as f:
                    data = json.load(f)
            else:
                print(f"  Descargando {info['name']} {season}...", end="", flush=True)
                time.sleep(SLEEP)
                data = api_get("/fixtures", {"league": league_id, "season": season})
                if data and data.get("response"):
                    save_json(data, out_path)
                    print(f" {len(data['response'])} fixtures")
                else:
                    print(" ERROR")
                    data = {"response": []}

            fixtures = data.get("response", [])
            ft = sum(1 for f in fixtures if f["fixture"]["status"]["short"] == "FT")
            total = len(fixtures)
            summary[league_id][season] = {"total": total, "ft": ft, "other": total - ft}

    # Also count CL (reference)
    cl_summary = {}
    for season in SEASONS:
        path = RAW_DIR / "exploration" / str(season) / "fixtures.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            fixtures = data.get("response", [])
            ft = sum(1 for f in fixtures if f["fixture"]["status"]["short"] == "FT")
            cl_summary[season] = {"ft": ft}
        else:
            cl_summary[season] = {"ft": 0}

    # Print summary table
    print(f"\n  {'Liga':<25s}", end="")
    for s in SEASONS:
        print(f" {s:>8d}", end="")
    print(f" {'Total FT':>10s}")
    print(f"  {'-'*25}" + f" {'-'*8}" * len(SEASONS) + f" {'-'*10}")

    grand_total = 0
    for league_id in [3, 848]:
        name = LEAGUES[league_id]["name"]
        total_ft = 0
        print(f"  {name:<25s}", end="")
        for s in SEASONS:
            ft = summary[league_id][s]["ft"]
            total_ft += ft
            print(f" {ft:>8d}", end="")
        print(f" {total_ft:>10d}")
        grand_total += total_ft

    cl_total = sum(cl_summary[s]["ft"] for s in SEASONS)
    print(f"  {'Champions (referencia)':<25s}", end="")
    for s in SEASONS:
        print(f" {cl_summary[s]['ft']:>8d}", end="")
    print(f" {cl_total:>10d}")
    grand_total += cl_total

    print(f"  {'TOTAL':<25s}", end="")
    print(f"{'':>{8 * len(SEASONS)}s}", end="")
    print(f" {grand_total:>10d}")

    # Estimate downloads needed
    new_ft = sum(summary[lid][s]["ft"] for lid in [3, 848] for s in SEASONS)
    new_downloads = new_ft * 2  # stats + players
    est_time = new_downloads * SLEEP

    print(f"\n  Fixtures nuevos FT: {new_ft}")
    print(f"  Descargas necesarias: ~{new_downloads} (statistics + players)")
    print(f"  Tiempo estimado: ~{fmt_time(est_time)}")

    return summary


# ===========================================================================
# Phase 2: Download statistics + players
# ===========================================================================

def phase2_download(summary):
    """Download stats and players for all EL/ECL fixtures."""
    sep("FASE 2: Descarga de statistics + players")

    if not API_KEY:
        sys.exit("ERROR: API_FOOTBALL_KEY no definida.")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all FT fixtures from EL and ECL
    fixtures = []
    for league_id in [3, 848]:
        info = LEAGUES[league_id]
        for season in SEASONS:
            path = RAW_DIR / info["dir"] / str(season) / "fixtures.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for fix in data.get("response", []):
                if fix["fixture"]["status"]["short"] != "FT":
                    continue
                fid = fix["fixture"]["id"]
                home = fix["teams"]["home"]["name"]
                away = fix["teams"]["away"]["name"]
                fixtures.append({
                    "id": fid, "label": f"{home} vs {away}",
                    "league_id": league_id, "season": season,
                })

    print(f"  Fixtures FT totales (EL+ECL): {len(fixtures)}")

    # Download statistics
    stats_needed = [f for f in fixtures
                    if not file_exists_and_valid(STATS_DIR / f"{f['id']}.json")]
    print(f"  Statistics pendientes: {len(stats_needed)}")

    if stats_needed:
        print(f"  Tiempo estimado stats: {fmt_time(len(stats_needed) * SLEEP)}")
        for i, fix in enumerate(stats_needed):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    stats [{i+1}/{len(stats_needed)}] {fix['label']}...", flush=True)
            time.sleep(SLEEP)
            data = api_get("/fixtures/statistics", {"fixture": fix["id"]})
            if data is not None:
                save_json(data, STATS_DIR / f"{fix['id']}.json")
            else:
                print(f"    FAIL: {fix['id']}")

    # Download players
    players_needed = [f for f in fixtures
                      if not file_exists_and_valid(PLAYERS_DIR / f"{f['id']}.json")]
    print(f"\n  Players pendientes: {len(players_needed)}")

    if players_needed:
        print(f"  Tiempo estimado players: {fmt_time(len(players_needed) * SLEEP)}")
        for i, fix in enumerate(players_needed):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    players [{i+1}/{len(players_needed)}] {fix['label']}...", flush=True)
            time.sleep(SLEEP)
            data = api_get("/fixtures/players", {"fixture": fix["id"]})
            if data is not None:
                save_json(data, PLAYERS_DIR / f"{fix['id']}.json")
            else:
                print(f"    FAIL: {fix['id']}")

    total_done = len(fixtures) * 2 - len(stats_needed) - len(players_needed)
    total_new = len(stats_needed) + len(players_needed)
    print(f"\n  Descargas completadas: {total_new} nuevas, {total_done} ya existían")


# ===========================================================================
# Phase 3: Unified Feature Engineering
# ===========================================================================

def phase3_feature_engineering():
    """Build matches_all.csv and features_all.csv from CL + EL + ECL."""
    sep("FASE 3: Feature Engineering Unificado (CL + EL + ECL)")

    # 3a. Load all fixtures from all 3 leagues
    rows = []
    for league_id, info in LEAGUES.items():
        league_dir = RAW_DIR / info["dir"]
        for season in SEASONS:
            fixtures_file = league_dir / str(season) / "fixtures.json"
            if not fixtures_file.exists():
                continue
            with open(fixtures_file) as f:
                data = json.load(f)

            for match in data.get("response", []):
                status = match["fixture"]["status"]["short"]
                if status != "FT":
                    continue

                fixture_id = match["fixture"]["id"]
                home_team_id = match["teams"]["home"]["id"]
                away_team_id = match["teams"]["away"]["id"]
                home_goals = match["goals"]["home"]
                away_goals = match["goals"]["away"]

                if home_goals is None or away_goals is None:
                    continue

                row = {
                    "fixture_id": fixture_id,
                    "date": pd.to_datetime(match["fixture"]["date"]),
                    "season": season,
                    "round": match["league"]["round"],
                    "league_id": league_id,
                    "league_name": info["name"],
                    "competition_level": info["level"],
                    "home_team_id": home_team_id,
                    "home_team_name": match["teams"]["home"]["name"],
                    "away_team_id": away_team_id,
                    "away_team_name": match["teams"]["away"]["name"],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": determine_result(home_goals, away_goals),
                }

                # Statistics
                stats_file = STATS_DIR / f"{fixture_id}.json"
                if stats_file.exists():
                    with open(stats_file) as f:
                        stats_data = json.load(f)
                    for team_entry in stats_data.get("response", []):
                        tid = team_entry["team"]["id"]
                        parsed = parse_team_stats(team_entry["statistics"])
                        prefix = "home_" if tid == home_team_id else (
                            "away_" if tid == away_team_id else None)
                        if prefix:
                            for k, v in parsed.items():
                                row[prefix + k] = v

                # Player aggregates
                players_file = PLAYERS_DIR / f"{fixture_id}.json"
                if players_file.exists():
                    with open(players_file) as f:
                        players_data = json.load(f)
                    for team_entry in players_data.get("response", []):
                        tid = team_entry["team"]["id"]
                        agg = compute_player_aggregates(team_entry["players"])
                        prefix = "home_" if tid == home_team_id else (
                            "away_" if tid == away_team_id else None)
                        if prefix:
                            for k, v in agg.items():
                                row[prefix + k] = v

                rows.append(row)

    matches_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Save matches_all.csv
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches_path = PROCESSED_DIR / "matches_all.csv"
    matches_df.to_csv(matches_path, index=False)

    print(f"  Partidos cargados: {len(matches_df)}")
    for lid, info in LEAGUES.items():
        n = len(matches_df[matches_df["league_id"] == lid])
        print(f"    {info['name']}: {n}")

    # 3b. Rolling features (team-centric, across ALL European competitions)
    print("\n  Calculando rolling features...")
    team_rows = []
    for _, m in matches_df.iterrows():
        fid = m["fixture_id"]
        date = m["date"]

        if m["result"] == "H":
            home_pts, away_pts = 3, 0
        elif m["result"] == "A":
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

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

        base = {
            "fixture_id": fid, "date": date,
            "goals_for": None, "goals_against": None,
            "xg_for": None, "xg_against": None,
            "shots_on_goal": None, "shots_accuracy": None,
            "possession": None, "pass_accuracy": None,
            "corner_kicks": None, "avg_rating": None,
            "key_passes": None, "duels_won_pct": None,
            "dribbles_success": None, "points": None,
        }

        # Home team row
        home_row = {**base,
            "team_id": m["home_team_id"], "is_home": True,
            "goals_for": m["home_goals"], "goals_against": m["away_goals"],
            "xg_for": m.get("home_xg"), "xg_against": m.get("away_xg"),
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
        }
        team_rows.append(home_row)

        # Away team row
        away_row = {**base,
            "team_id": m["away_team_id"], "is_home": False,
            "goals_for": m["away_goals"], "goals_against": m["home_goals"],
            "xg_for": m.get("away_xg"), "xg_against": m.get("home_xg"),
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
        }
        team_rows.append(away_row)

    team_df = pd.DataFrame(team_rows).sort_values(["team_id", "date"]).reset_index(drop=True)
    team_df["xg_diff"] = team_df["xg_for"] - team_df["xg_against"]
    team_df["xg_overperformance"] = team_df["goals_for"] - team_df["xg_for"]

    roll_cols = {
        "xg_for": "rolling_xg_for", "xg_against": "rolling_xg_against",
        "xg_diff": "rolling_xg_diff", "goals_for": "rolling_goals_for",
        "goals_against": "rolling_goals_against",
        "xg_overperformance": "rolling_xg_overperformance",
        "shots_on_goal": "rolling_shots_on_goal",
        "shots_accuracy": "rolling_shots_accuracy",
        "possession": "rolling_possession", "pass_accuracy": "rolling_pass_accuracy",
        "corner_kicks": "rolling_corners", "avg_rating": "rolling_avg_rating",
        "key_passes": "rolling_key_passes", "duels_won_pct": "rolling_duels_won_pct",
        "dribbles_success": "rolling_dribbles_success", "points": "rolling_points",
    }

    for src_col, dst_col in roll_cols.items():
        team_df[dst_col] = (
            team_df.groupby("team_id")[src_col]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    team_df["days_since_last"] = (
        team_df.groupby("team_id")["date"]
        .transform(lambda s: s.diff().dt.days)
    )

    rolling_feature_cols = list(roll_cols.values()) + ["days_since_last"]

    home_rolling = (
        team_df[team_df["is_home"]][["fixture_id"] + rolling_feature_cols]
        .rename(columns={c: f"home_{c}" for c in rolling_feature_cols})
    )
    away_rolling = (
        team_df[~team_df["is_home"]][["fixture_id"] + rolling_feature_cols]
        .rename(columns={c: f"away_{c}" for c in rolling_feature_cols})
    )

    # 3c. Merge rolling features
    features_df = matches_df.merge(home_rolling, on="fixture_id", how="left")
    features_df = features_df.merge(away_rolling, on="fixture_id", how="left")

    # 3d. Derived features
    features_df["diff_rolling_xg"] = features_df["home_rolling_xg_for"] - features_df["away_rolling_xg_for"]
    features_df["diff_rolling_goals"] = features_df["home_rolling_goals_for"] - features_df["away_rolling_goals_for"]
    features_df["diff_rolling_form"] = features_df["home_rolling_points"] - features_df["away_rolling_points"]
    features_df["is_knockout"] = features_df["round"].apply(is_knockout_round).astype(int)

    # 3e. Competition features
    features_df["is_champions"] = (features_df["league_id"] == 2).astype(int)
    features_df["is_europa"] = (features_df["league_id"] == 3).astype(int)
    features_df["is_conference"] = (features_df["league_id"] == 848).astype(int)

    print(f"\n  matches_all.csv: {matches_df.shape}")
    print(f"  features (pre-ELO): {features_df.shape}")

    # Distribution
    print(f"\n  Distribución de resultado:")
    dist = features_df["result"].value_counts()
    for label in ["H", "D", "A"]:
        count = dist.get(label, 0)
        pct = count / len(features_df) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    return matches_df, features_df


# ===========================================================================
# Phase 4: ELO Integration
# ===========================================================================

def phase4_elo(features_df):
    """Download new ELO snapshots and assign to all matches."""
    sep("FASE 4: Integración ELO")
    ELO_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing snapshots
    snapshots = {}
    for csv_file in ELO_DIR.glob("*.csv"):
        date_str = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                snapshots[date_str] = df
        except Exception:
            pass

    print(f"  Snapshots ELO existentes: {len(snapshots)}")

    # Find new dates needed
    all_dates = features_df["date"].dt.date.unique()
    needed = []
    for d in sorted(all_dates):
        date_str = d.strftime("%Y-%m-%d")
        if date_str not in snapshots:
            needed.append((d, date_str))

    print(f"  Fechas nuevas a descargar: {len(needed)}")

    if needed:
        print(f"  Tiempo estimado: {fmt_time(len(needed) * 2)}")
        for i, (d, date_str) in enumerate(needed):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1}/{len(needed)}] {date_str}...", flush=True)
            time.sleep(2)
            try:
                resp = requests.get(f"http://api.clubelo.com/{date_str}", timeout=15)
                resp.raise_for_status()
                text = resp.text.strip()
                lines = text.split("\n")
                if len(lines) > 1:
                    path = ELO_DIR / f"{date_str}.csv"
                    with open(path, "w") as f:
                        f.write(text)
                    snapshots[date_str] = pd.read_csv(path)
            except Exception as e:
                print(f"    Error {date_str}: {e}")

        print(f"  Total snapshots: {len(snapshots)}")

    # Assign ELO
    features_df = features_df.copy()
    features_df["elo_home"] = np.nan
    features_df["elo_away"] = np.nan
    matched = 0
    unmatched_teams = set()

    for idx, row in features_df.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        if date_str not in snapshots:
            continue
        snapshot = snapshots[date_str]
        elo_lookup = dict(zip(snapshot["Club"], snapshot["Elo"]))

        for side, team_col in [("home", "home_team_name"), ("away", "away_team_name")]:
            team_name = row[team_col]
            elo_name = NAME_MAP.get(team_name)
            if elo_name and elo_name in elo_lookup:
                features_df.at[idx, f"elo_{side}"] = elo_lookup[elo_name]
            elif elo_name is None:
                # Try exact match
                if team_name in elo_lookup:
                    features_df.at[idx, f"elo_{side}"] = elo_lookup[team_name]
                    NAME_MAP[team_name] = team_name  # Cache for future
                else:
                    unmatched_teams.add(team_name)
            else:
                unmatched_teams.add(team_name)

        if not pd.isna(features_df.at[idx, "elo_home"]) and not pd.isna(features_df.at[idx, "elo_away"]):
            matched += 1

    features_df["elo_diff"] = features_df["elo_home"] - features_df["elo_away"]
    features_df["elo_expected_home"] = 1 / (1 + 10 ** ((features_df["elo_away"] - features_df["elo_home"]) / 400))

    print(f"\n  ELO coverage: {matched}/{len(features_df)} ({matched/len(features_df)*100:.1f}%)")
    if unmatched_teams:
        print(f"  Equipos sin mapeo ELO: {len(unmatched_teams)}")
        for t in sorted(list(unmatched_teams)[:20]):
            print(f"    {t}")
        if len(unmatched_teams) > 20:
            print(f"    ... y {len(unmatched_teams) - 20} más")

    return features_df


# ===========================================================================
# Phase 5: Domestic Features (pragmatic)
# ===========================================================================

def phase5_domestic(features_df):
    """Compute domestic features reusing existing data. NaN for new teams."""
    sep("FASE 5: Features Domésticas (pragmático)")

    team_leagues_path = PROCESSED_DIR / "team_leagues.json"
    if not team_leagues_path.exists():
        print("  team_leagues.json no encontrado. Domestic features = NaN.")
        for prefix in ["home_", "away_"]:
            for feat in ["domestic_points_last5", "domestic_goals_for_last5",
                         "domestic_goals_against_last5", "domestic_xg_for_last5",
                         "domestic_xg_against_last5", "domestic_league_position",
                         "domestic_ppg", "domestic_win_rate",
                         "domestic_home_goals_avg", "domestic_away_goals_avg"]:
                features_df[prefix + feat] = np.nan
        features_df["diff_domestic_position"] = np.nan
        return features_df

    with open(team_leagues_path) as f:
        team_leagues = json.load(f)

    # Build team->league lookup
    team_league_map = {}  # (team_id, season) -> league_id
    for tid, info in team_leagues.items():
        for season_str, league_info in info.get("seasons", {}).items():
            if league_info and "league_id" in league_info:
                team_league_map[(int(tid), int(season_str))] = league_info["league_id"]

    print(f"  Equipos con mapeo doméstico: {len(team_leagues)}")

    # Build domestic matches DataFrame (reuse existing downloaded data)
    print("  Cargando partidos domésticos existentes...")
    domestic_rows = []
    league_seasons = set()
    cl_teams_by_season = {}
    for tid, info in team_leagues.items():
        for season_str, league_info in info.get("seasons", {}).items():
            season = int(season_str)
            if league_info and "league_id" in league_info:
                league_seasons.add((league_info["league_id"], season))
            if season not in cl_teams_by_season:
                cl_teams_by_season[season] = set()
            cl_teams_by_season[season].add(int(tid))

    for league_id, season in sorted(league_seasons):
        key = f"{league_id}_{season}"
        path = DOMESTIC_DIR / f"{key}_fixtures.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for fix in data.get("response", []):
            hg = fix["goals"]["home"]
            ag = fix["goals"]["away"]
            if hg is None or ag is None:
                continue
            fid = fix["fixture"]["id"]
            row = {
                "fixture_id": fid,
                "date": pd.to_datetime(fix["fixture"]["date"]),
                "league_id": league_id,
                "season": season,
                "home_team_id": fix["teams"]["home"]["id"],
                "away_team_id": fix["teams"]["away"]["id"],
                "home_goals": hg,
                "away_goals": ag,
            }
            # xG from statistics if available
            stats_path = DOMESTIC_STATS_DIR / f"{fid}.json"
            if stats_path.exists():
                with open(stats_path) as sf:
                    stats_data = json.load(sf)
                for te in stats_data.get("response", []):
                    tid = te["team"]["id"]
                    prefix = "home_" if tid == row["home_team_id"] else (
                        "away_" if tid == row["away_team_id"] else None)
                    if prefix:
                        for stat in te.get("statistics", []):
                            if stat["type"] == "expected_goals":
                                row[f"{prefix}xg"] = parse_stat_value("expected_goals", stat["value"])
            domestic_rows.append(row)

    domestic_df = pd.DataFrame(domestic_rows)
    if len(domestic_df) > 0:
        domestic_df = domestic_df.sort_values("date").reset_index(drop=True)
    print(f"  Partidos domésticos cargados: {len(domestic_df)}")

    # Compute domestic features for each match
    features_df = features_df.copy()
    domestic_feat_names = [
        "domestic_points_last5", "domestic_goals_for_last5",
        "domestic_goals_against_last5", "domestic_xg_for_last5",
        "domestic_xg_against_last5", "domestic_league_position",
        "domestic_ppg", "domestic_win_rate",
        "domestic_home_goals_avg", "domestic_away_goals_avg",
    ]

    # Initialize columns
    for prefix in ["home_", "away_"]:
        for feat in domestic_feat_names:
            features_df[prefix + feat] = np.nan

    has_domestic = 0
    total = len(features_df)

    for i, (idx, match) in enumerate(features_df.iterrows()):
        if (i + 1) % 500 == 0:
            print(f"    Procesando {i+1}/{total}...")

        cl_date = match["date"]
        season = match["season"]

        for side, tid_col in [("home", "home_team_id"), ("away", "away_team_id")]:
            team_id = match[tid_col]
            league_id = team_league_map.get((team_id, season))
            if league_id is None or len(domestic_df) == 0:
                continue

            feats = _compute_domestic_for_team(domestic_df, team_id, cl_date, league_id, season)
            for k, v in feats.items():
                features_df.at[idx, f"{side}_{k}"] = v

        # Check if at least one domestic feature is non-NaN
        if not pd.isna(features_df.at[idx, "home_domestic_ppg"]) or \
           not pd.isna(features_df.at[idx, "away_domestic_ppg"]):
            has_domestic += 1

    features_df["diff_domestic_position"] = (
        features_df["away_domestic_league_position"] - features_df["home_domestic_league_position"]
    )

    print(f"\n  Cobertura doméstica: {has_domestic}/{total} ({has_domestic/total*100:.1f}%)")

    return features_df


def _compute_domestic_for_team(domestic_df, team_id, before_date, league_id, season, n=5):
    """Compute domestic features for one team before a given date."""
    result = {}
    for k in ["domestic_points_last5", "domestic_goals_for_last5",
              "domestic_goals_against_last5", "domestic_xg_for_last5",
              "domestic_xg_against_last5", "domestic_league_position",
              "domestic_ppg", "domestic_win_rate",
              "domestic_home_goals_avg", "domestic_away_goals_avg"]:
        result[k] = np.nan

    league_matches = domestic_df[
        (domestic_df["league_id"] == league_id) & (domestic_df["season"] == season)
    ]
    if len(league_matches) == 0:
        return result

    team_home = league_matches[
        (league_matches["home_team_id"] == team_id) & (league_matches["date"] < before_date)
    ]
    team_away = league_matches[
        (league_matches["away_team_id"] == team_id) & (league_matches["date"] < before_date)
    ]

    team_rows = []
    for _, m in team_home.iterrows():
        team_rows.append({
            "date": m["date"],
            "goals_for": m["home_goals"], "goals_against": m["away_goals"],
            "xg_for": m.get("home_xg"), "xg_against": m.get("away_xg"),
            "is_home": True,
            "points": 3 if m["home_goals"] > m["away_goals"] else (1 if m["home_goals"] == m["away_goals"] else 0),
            "win": 1 if m["home_goals"] > m["away_goals"] else 0,
        })
    for _, m in team_away.iterrows():
        team_rows.append({
            "date": m["date"],
            "goals_for": m["away_goals"], "goals_against": m["home_goals"],
            "xg_for": m.get("away_xg"), "xg_against": m.get("home_xg"),
            "is_home": False,
            "points": 3 if m["away_goals"] > m["home_goals"] else (1 if m["away_goals"] == m["home_goals"] else 0),
            "win": 1 if m["away_goals"] > m["home_goals"] else 0,
        })

    if not team_rows:
        return result

    team_df = pd.DataFrame(team_rows).sort_values("date")
    total_played = len(team_df)
    result["domestic_ppg"] = team_df["points"].sum() / total_played
    result["domestic_win_rate"] = team_df["win"].sum() / total_played

    home_m = team_df[team_df["is_home"]]
    away_m = team_df[~team_df["is_home"]]
    if len(home_m) > 0:
        result["domestic_home_goals_avg"] = home_m["goals_for"].mean()
    if len(away_m) > 0:
        result["domestic_away_goals_avg"] = away_m["goals_for"].mean()

    last_n = team_df.tail(n)
    result["domestic_points_last5"] = last_n["points"].sum()
    result["domestic_goals_for_last5"] = last_n["goals_for"].mean()
    result["domestic_goals_against_last5"] = last_n["goals_against"].mean()

    xg_for = last_n["xg_for"].dropna()
    xg_against = last_n["xg_against"].dropna()
    if len(xg_for) > 0:
        result["domestic_xg_for_last5"] = xg_for.mean()
    if len(xg_against) > 0:
        result["domestic_xg_against_last5"] = xg_against.mean()

    # League position
    before = league_matches[league_matches["date"] < before_date]
    if len(before) > 0:
        standings = {}
        for _, m in before.iterrows():
            hid, aid = m["home_team_id"], m["away_team_id"]
            for t in [hid, aid]:
                if t not in standings:
                    standings[t] = {"pts": 0, "gd": 0}
            hg, ag = m["home_goals"], m["away_goals"]
            standings[hid]["gd"] += hg - ag
            standings[aid]["gd"] += ag - hg
            if hg > ag:
                standings[hid]["pts"] += 3
            elif hg < ag:
                standings[aid]["pts"] += 3
            else:
                standings[hid]["pts"] += 1
                standings[aid]["pts"] += 1

        if team_id in standings:
            sorted_teams = sorted(standings.keys(),
                                  key=lambda t: (standings[t]["pts"], standings[t]["gd"]),
                                  reverse=True)
            result["domestic_league_position"] = sorted_teams.index(team_id) + 1

    return result


# ===========================================================================
# Phase 6: Re-train and Compare
# ===========================================================================

def phase6_retrain(features_df):
    """Train 3 variants and compare."""
    sep("FASE 6: Re-entrenamiento y Comparación")

    le = LabelEncoder()
    le.fit(CLASSES)

    features_df = features_df.copy()
    features_df["date"] = pd.to_datetime(features_df["date"])
    features_df = features_df.sort_values("date").reset_index(drop=True)

    # Season column should be int for splitting
    features_df["season"] = features_df["season"].astype(int)

    cl_df = features_df[features_df["league_id"] == 2].copy()

    print(f"  Dataset completo: {len(features_df)} partidos")
    print(f"  Solo Champions: {len(cl_df)} partidos")
    print(f"  Features base: {len(FINAL_FEATURES)}")

    # Competition features to add
    comp_features = ["is_champions", "is_europa", "is_conference", "competition_level"]
    extended_features = FINAL_FEATURES + comp_features

    # Check feature availability
    missing = [f for f in extended_features if f not in features_df.columns]
    if missing:
        print(f"  WARNING: Features faltantes: {missing}")
        extended_features = [f for f in extended_features if f in features_df.columns]

    # Build pipeline factory (same as final model)
    def make_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.01, penalty="l2", solver="saga",
                max_iter=2000, random_state=42)),
        ])

    def make_calibrated():
        return CalibratedClassifierCV(make_pipe(), cv=3, method="sigmoid")

    # --- Walk-forward splits ---
    def build_splits_for(df):
        df = df.sort_values("date").reset_index(drop=True)
        s2023 = df[df["season"] == 2023]
        s2024 = df[df["season"] == 2024].sort_values("date")
        s2025 = df[df["season"] == 2025]
        mid = len(s2024) // 2
        return [
            ("Split 1", s2023, s2024.iloc[:mid]),
            ("Split 2", pd.concat([s2023, s2024.iloc[:mid]]), s2024.iloc[mid:]),
            ("Split 3", pd.concat([s2023, s2024]), s2025),
        ]

    def eval_variant(train_df, val_df, features, le):
        """Train on train_df, evaluate on val_df."""
        pipe = make_calibrated()
        X_tr = train_df[features].values
        y_tr = le.transform(train_df["result"])
        X_val = val_df[features].values
        y_val = val_df["result"].values

        pipe.fit(X_tr, y_tr)
        y_pred = le.inverse_transform(pipe.predict(X_val))
        y_proba = pipe.predict_proba(X_val)
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "log_loss": log_loss(y_val, y_proba, labels=le.classes_),
            "f1_macro": f1_score(y_val, y_pred, average="macro", labels=le.classes_),
            "n_train": len(train_df),
            "n_val": len(val_df),
        }

    # === Variant A: CL only, 20 features (reproduce final) ===
    print("\n  Variante A: Solo CL, 20 features (control)...")
    cl_splits = build_splits_for(cl_df)
    a_metrics = []
    for split_name, train, val in cl_splits:
        m = eval_variant(train, val, FINAL_FEATURES, le)
        a_metrics.append(m)
        print(f"    {split_name}: Acc={m['accuracy']:.4f} LL={m['log_loss']:.4f} "
              f"(train={m['n_train']}, val={m['n_val']})")

    a_avg = {
        "accuracy": np.mean([m["accuracy"] for m in a_metrics]),
        "log_loss": np.mean([m["log_loss"] for m in a_metrics]),
        "f1_macro": np.mean([m["f1_macro"] for m in a_metrics]),
        "n_train": a_metrics[-1]["n_train"],
    }
    print(f"    AVG: Acc={a_avg['accuracy']:.4f} LL={a_avg['log_loss']:.4f}")

    # === Variant B: All data train, eval ONLY on CL ===
    print("\n  Variante B: Train ALL, eval solo CL, 20+4 features...")
    all_splits = build_splits_for(features_df)
    b_metrics = []
    for i, ((split_name, train_all, val_all), (_, _, val_cl)) in enumerate(
            zip(all_splits, cl_splits)):
        m = eval_variant(train_all, val_cl, extended_features, le)
        b_metrics.append(m)
        print(f"    {split_name}: Acc={m['accuracy']:.4f} LL={m['log_loss']:.4f} "
              f"(train={m['n_train']}, val={m['n_val']})")

    b_avg = {
        "accuracy": np.mean([m["accuracy"] for m in b_metrics]),
        "log_loss": np.mean([m["log_loss"] for m in b_metrics]),
        "f1_macro": np.mean([m["f1_macro"] for m in b_metrics]),
        "n_train": b_metrics[-1]["n_train"],
    }
    print(f"    AVG: Acc={b_avg['accuracy']:.4f} LL={b_avg['log_loss']:.4f}")

    # === Variant C: All data train, eval on ALL ===
    print("\n  Variante C: Train ALL, eval ALL, 20+4 features...")
    c_metrics = []
    for split_name, train_all, val_all in all_splits:
        m = eval_variant(train_all, val_all, extended_features, le)
        c_metrics.append(m)
        print(f"    {split_name}: Acc={m['accuracy']:.4f} LL={m['log_loss']:.4f} "
              f"(train={m['n_train']}, val={m['n_val']})")

    c_avg = {
        "accuracy": np.mean([m["accuracy"] for m in c_metrics]),
        "log_loss": np.mean([m["log_loss"] for m in c_metrics]),
        "f1_macro": np.mean([m["f1_macro"] for m in c_metrics]),
        "n_train": c_metrics[-1]["n_train"],
    }
    print(f"    AVG: Acc={c_avg['accuracy']:.4f} LL={c_avg['log_loss']:.4f}")

    # === Comparison table ===
    sep("TABLA COMPARATIVA")

    hdr = (f"  {'Variante':<35s} {'N train':>8s} {'Accuracy':>9s} "
           f"{'Log Loss':>9s} {'F1':>7s} {'Δ vs final':>11s}")
    print(hdr)
    print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*9} {'-'*7} {'-'*11}")

    for name, avg in [("A: CL only (control)", a_avg),
                      ("B: All → eval CL", b_avg),
                      ("C: All → eval ALL", c_avg)]:
        delta = avg["accuracy"] - FINAL_ACCURACY
        print(f"  {name:<35s} {avg['n_train']:>8d} {avg['accuracy']:>9.4f} "
              f"{avg['log_loss']:>9.4f} {avg['f1_macro']:>7.4f} {delta:>+11.4f}")

    print(f"\n  Referencia modelo final: Acc={FINAL_ACCURACY} LL={FINAL_LOG_LOSS}")

    # === Save if Variant B improves CL prediction ===
    sep("DECISIÓN")

    b_delta_acc = b_avg["accuracy"] - FINAL_ACCURACY
    b_delta_ll = b_avg["log_loss"] - FINAL_LOG_LOSS
    improved = b_avg["accuracy"] > FINAL_ACCURACY

    print(f"\n  Variante B vs modelo final:")
    print(f"    Accuracy: {b_avg['accuracy']:.4f} vs {FINAL_ACCURACY} (Δ={b_delta_acc:+.4f})")
    print(f"    Log Loss: {b_avg['log_loss']:.4f} vs {FINAL_LOG_LOSS} (Δ={b_delta_ll:+.4f})")

    if improved:
        print(f"\n  MEJORA: +{b_delta_acc:.4f} accuracy en predicción CL")
        print(f"  Guardando modelo v4...")

        # Train on ALL data
        final_pipe = make_calibrated()
        final_pipe.fit(features_df[extended_features].values,
                       le.transform(features_df["result"]))
        train_acc = accuracy_score(
            le.transform(features_df["result"]),
            final_pipe.predict(features_df[extended_features].values))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / "champion_model_v4.pkl"
        joblib.dump(final_pipe, model_path)
        print(f"  Saved: {model_path}")

        feat_path = MODELS_DIR / "feature_list_v4.json"
        with open(feat_path, "w") as f:
            json.dump(extended_features, f, indent=2)
        print(f"  Saved: {feat_path}")

        metrics_v4 = {
            "model": "LogReg tuned + cal(sigmoid) + multi-league",
            "n_features": len(extended_features),
            "features": extended_features,
            "accuracy_cl_eval": round(b_avg["accuracy"], 4),
            "log_loss_cl_eval": round(b_avg["log_loss"], 4),
            "f1_macro_cl_eval": round(b_avg["f1_macro"], 4),
            "accuracy_all_eval": round(c_avg["accuracy"], 4),
            "log_loss_all_eval": round(c_avg["log_loss"], 4),
            "delta_accuracy_vs_final": round(b_delta_acc, 4),
            "delta_log_loss_vs_final": round(b_delta_ll, 4),
            "training_accuracy": round(train_acc, 4),
            "n_matches_total": len(features_df),
            "n_matches_cl": len(cl_df),
            "per_split_b": b_metrics,
            "per_split_c": c_metrics,
        }
        met_path = MODELS_DIR / "metrics_v4.json"
        with open(met_path, "w") as f:
            json.dump(metrics_v4, f, indent=2, default=str)
        print(f"  Saved: {met_path}")
    else:
        print(f"\n  NO MEJORO: más datos no ayudan a predecir CL.")
        print(f"  El modelo final sigue siendo el mejor.")

        metrics_v4 = {
            "improved": False,
            "variant_a": {"accuracy": round(a_avg["accuracy"], 4), "log_loss": round(a_avg["log_loss"], 4)},
            "variant_b": {"accuracy": round(b_avg["accuracy"], 4), "log_loss": round(b_avg["log_loss"], 4)},
            "variant_c": {"accuracy": round(c_avg["accuracy"], 4), "log_loss": round(c_avg["log_loss"], 4)},
            "final_reference": {"accuracy": FINAL_ACCURACY, "log_loss": FINAL_LOG_LOSS},
        }
        met_path = MODELS_DIR / "metrics_v4.json"
        with open(met_path, "w") as f:
            json.dump(metrics_v4, f, indent=2)
        print(f"  Saved for reference: {met_path}")

    # Save features_all.csv regardless
    feat_all_path = PROCESSED_DIR / "features_all.csv"
    features_df.to_csv(feat_all_path, index=False)
    print(f"\n  Saved: {feat_all_path} ({features_df.shape})")


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()
    download_mode = "--download" in sys.argv

    sep("09 — EUROPA LEAGUE + CONFERENCE LEAGUE DATA", "█")
    print(f"  Modo: {'FULL (exploration + download + train)' if download_mode else 'EXPLORATION ONLY'}")

    # Phase 1 always runs
    summary = phase1_exploration()

    if not download_mode:
        print(f"\n  Para ejecutar descarga completa:")
        print(f"  python3 09_europa_league_data.py --download")
        return

    # Phase 2
    phase2_download(summary)

    # Phase 3
    matches_df, features_df = phase3_feature_engineering()

    # Phase 4
    features_df = phase4_elo(features_df)

    # Phase 5
    features_df = phase5_domestic(features_df)

    # Phase 6
    phase6_retrain(features_df)

    elapsed = time.time() - t_start
    sep("PIPELINE COMPLETADO", "█")
    print(f"  Tiempo total: {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
