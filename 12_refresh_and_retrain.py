#!/usr/bin/env python3
"""
12_refresh_and_retrain.py — Refresh data and retrain all 3 models.

Re-executable weekly. Phases:
  1. Download new finished fixtures + statistics + players
  2. Re-run unified feature engineering (CL + EL + ECL)
  3. Retrain all 3 models with updated data
  4. Update upcoming fixtures, recent results & odds
  5. Compare before vs after
  6. Git commit & push
"""

import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
SLEEP = 6.5
SEASONS = [2023, 2024, 2025]
ROLLING_WINDOW = 5

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

STAT_NAME_MAP = {
    "Shots on Goal": "shots_on_goal", "Shots off Goal": "shots_off_goal",
    "Total Shots": "total_shots", "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_insidebox", "Shots outsidebox": "shots_outsidebox",
    "Fouls": "fouls", "Corner Kicks": "corner_kicks", "Offsides": "offsides",
    "Ball Possession": "possession", "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards", "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes", "Passes accurate": "passes_accurate",
    "Passes %": "pass_pct", "expected_goals": "xg",
    "goals_prevented": "goals_prevented",
}

# ELO name mapping (from 09_europa_league_data.py)
NAME_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle": "Newcastle", "Arsenal": "Arsenal", "Liverpool": "Liverpool",
    "Chelsea": "Chelsea", "Tottenham": "Tottenham", "Aston Villa": "Aston Villa",
    "West Ham": "West Ham", "Brighton": "Brighton",
    "Wolverhampton Wanderers": "Wolverhampton", "Everton": "Everton",
    "Nottingham Forest": "Nottingham",
    "Crystal Palace": "Crystal Palace",
    "Real Madrid": "Real Madrid", "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico", "Real Sociedad": "Sociedad",
    "Sevilla": "Sevilla", "Villarreal": "Villarreal", "Girona": "Girona",
    "Athletic Club": "Bilbao", "Real Betis": "Betis", "Rayo Vallecano": "Vallecano",
    "Celta Vigo": "Celta",
    "Bayern München": "Bayern", "Bayern Munich": "Bayern",
    "Borussia Dortmund": "Dortmund", "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Frankfurt", "Bayer Leverkusen": "Leverkusen",
    "Union Berlin": "Union Berlin", "VfB Stuttgart": "Stuttgart",
    "SC Freiburg": "Freiburg", "TSG Hoffenheim": "Hoffenheim",
    "Borussia Monchengladbach": "Gladbach", "FC Augsburg": "Augsburg",
    "1. FC Heidenheim 1846": "Heidenheim",
    "Inter": "Inter", "AC Milan": "AC Milan", "Napoli": "Napoli",
    "Juventus": "Juventus", "Atalanta": "Atalanta", "Lazio": "Lazio",
    "Bologna": "Bologna", "AS Roma": "AS Roma", "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Paris Saint Germain": "Paris SG", "Marseille": "Marseille",
    "Lens": "Lens", "Monaco": "Monaco", "Lille": "Lille", "Nice": "Nice",
    "Stade Brestois 29": "Brest", "Lyon": "Lyon", "Rennes": "Rennes",
    "Toulouse": "Toulouse", "Stade Reims": "Reims", "Nantes": "Nantes",
    "Montpellier": "Montpellier", "Strasbourg": "Strasbourg",
    "Benfica": "Benfica", "FC Porto": "Porto", "Sporting CP": "Sporting",
    "SC Braga": "Braga", "Vitoria Guimaraes": "Guimaraes",
    "PSV Eindhoven": "PSV", "Ajax": "Ajax", "Feyenoord": "Feyenoord",
    "Twente": "Twente", "AZ Alkmaar": "AZ",
    "Club Brugge KV": "Brugge", "Antwerp": "Antwerp", "Genk": "Genk",
    "Union St. Gilloise": "Union SG", "RSC Anderlecht": "Anderlecht",
    "Gent": "Gent", "Cercle Brugge": "Cercle Brugge",
    "Galatasaray": "Galatasaray", "Fenerbahçe": "Fenerbahce",
    "Besiktas": "Besiktas", "Istanbul Basaksehir": "Basaksehir",
    "Trabzonspor": "Trabzonspor",
    "Celtic": "Celtic", "Rangers": "Rangers",
    "Heart Of Midlothian": "Hearts", "Aberdeen": "Aberdeen",
    "Red Bull Salzburg": "Salzburg", "Sturm Graz": "Sturm Graz",
    "SK Rapid Wien": "Rapid Wien", "LASK": "LASK",
    "Wolfsberger AC": "Wolfsberg",
    "BSC Young Boys": "Young Boys", "FC Basel 1893": "Basel",
    "FC Lugano": "Lugano", "Servette FC": "Servette",
    "FC Zurich": "Zuerich", "FC St. Gallen": "St. Gallen",
    "Sparta Praha": "Sparta Praha", "Slavia Praha": "Slavia Praha",
    "Plzen": "Viktoria Plzen", "FK Mlada Boleslav": "Mlada Boleslav",
    "FC Copenhagen": "Koebenhavn", "FC Midtjylland": "Midtjylland",
    "Aarhus": "AGF", "Silkeborg IF": "Silkeborg",
    "FC Nordsjaelland": "Nordsjaelland",
    "Malmo FF": "Malmoe", "BK Hacken": "Hacken",
    "IF Elfsborg": "Elfsborg", "Djurgarden": "Djurgarden",
    "Molde": "Molde", "Bodo/Glimt": "Bodoe Glimt",
    "Brann": "Brann", "Rosenborg": "Rosenborg",
    "Olympiakos Piraeus": "Olympiakos", "PAOK": "PAOK",
    "Panathinaikos": "Panathinaikos", "AEK Athens FC": "AEK",
    "Aris": "Aris", "Asteras Tripolis": "Asteras Tripolis",
    "Dinamo Zagreb": "Dinamo Zagreb", "HNK Rijeka": "Rijeka",
    "Hajduk Split": "Hajduk Split", "NK Osijek": "Osijek",
    "FK Crvena Zvezda": "Crvena Zvezda", "FK Partizan": "Partizan",
    "TSC Backa Topola": "Backa Topola",
    "Shakhtar Donetsk": "Shakhtar", "Dynamo Kyiv": "Dynamo Kyiv",
    "Dnipro-1": "Dnipro", "Zorya Luhansk": "Zorya",
    "FCSB": "FCSB", "Farul Constanta": "Farul", "CFR 1907 Cluj": "CFR Cluj",
    "Ferencvarosi TC": "Ferencvaros", "Puskas Akademia FC": "Felcsut",
    "Apoel Nicosia": "APOEL", "Pafos": "Paphos",
    "AEK Larnaca": "AEK Larnaca", "Omonia Nicosia": "Omonia",
    "Lech Poznan": "Lech", "Raków Częstochowa": "Rakow",
    "Jagiellonia": "Jagiellonia", "Legia Warszawa": "Legia",
    "Ludogorets": "Razgrad", "CSKA Sofia 1948": "CSKA 1948",
    "Maccabi Haifa": "Maccabi Haifa", "Maccabi Tel Aviv": "Maccabi Tel-Aviv",
    "Slovan Bratislava": "Slovan Bratislava",
    "FC Astana": "FK Astana", "Ordabasy": "Ordabasy", "Kairat Almaty": "Kairat",
    "Sheriff Tiraspol": "Sheriff Tiraspol",
    "Milsami Orhei": "Milsami Orhei", "Petrocub": "Petrocub",
    "HJK helsinki": "HJK Helsinki", "KuPS": "Kuopio",
    "Shamrock Rovers": "Shamrock Rovers", "Shelbourne": "Shelbourne",
    "Linfield": "Linfield", "Larne": "Larne",
    "Breidablik": "Breidablik", "Vikingur Reykjavik": "Vikingur",
    "KI Klaksvik": "Klaksvik", "Vikingur Gota": "Vikingur Gota",
    "Shkendija": "Shkendija", "Struga": "Struga",
    "Partizani": "Partizani Tirana", "Egnatia Rrogozhinë": "Egnatia",
    "Drita": "Drita", "Ballkani": "Ballkani",
    "Zrinjski": "Zrinjski Mostar", "Borac Banja Luka": "Borac Banja Luka",
    "FK Zalgiris Vilnius": "Zalgiris", "Panevėžys": "Panevezys",
    "Flora Tallinn": "Flora Tallinn", "FC Levadia Tallinn": "Levadia",
    "Rīgas FS": "Rigas", "Valmiera / BSS": "Valmiera",
    "Dinamo Tbilisi": "Dinamo Tbilisi", "Dinamo Batumi": "Batumi",
    "Saburtalo": "Saburtalo", "FC Noah": "Noah",
    "Pyunik Yerevan": "Pyunik", "FC Urartu": "Urartu",
    "Buducnost Podgorica": "Buducnost", "Dečić": "Decic",
    "Bate Borisov": "BATE", "Dinamo Minsk": "Dinamo Minsk",
    "Olimpija Ljubljana": "Olimpija Ljubljana", "Celje": "Celje",
    "Tre Penne": "Tre Penne", "Virtus": "SS Virtus",
    "Hamrun Spartans": "Hamrun", "Lincoln Red Imps FC": "Lincoln",
    "UE Santa Coloma": "Santa Coloma",
    "Atlètic Club d'Escaldes": "Atletic Club Escaldes",
    "Inter Club d'Escaldes": "Escaldes",
    "FC Differdange 03": "Differdange", "Swift Hesperange": "Swift Hesperange",
    "The New Saints": "The New Saints",
    "Stade Rennais": "Rennes", "Qarabag": "Qarabag",
    "Slovan Liberec": "Slovan Liberec", "FC Sheriff": "Sheriff Tiraspol",
    "Maccabi Netanya": "Maccabi Netanya", "Flora": "Flora Tallinn",
    "Legia Warsaw": "Legia", "Steaua Bucuresti": "FCSB",
    "FC Twente": "Twente", "FC Heidenheim": "Heidenheim",
    "Vitoria SC": "Guimaraes", "Anderlecht": "Anderlecht",
    "Samsunspor": "Samsunspor", "Lausanne": "Lausanne",
    "Sigma Olomouc": "Sigma Olomouc",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def api_get(endpoint, params):
    if not API_KEY:
        sys.exit("ERROR: API_FOOTBALL_KEY not set")
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


def file_ok(path):
    if not path.exists() or path.stat().st_size < 20:
        return False
    try:
        with open(path) as f:
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
        col = STAT_NAME_MAP.get(stat["type"])
        if col:
            result[col] = parse_stat_value(stat["type"], stat["value"])
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
        "key_passes": key_passes, "duels_total": duels_total,
        "duels_won": duels_won, "dribbles_success": dribbles_success,
        "tackles_total": tackles_total, "interceptions": interceptions,
    }


def determine_result(hg, ag):
    return "H" if hg > ag else ("A" if hg < ag else "D")


def is_knockout_round(round_name):
    return bool(re.search(
        r"(Round of|Quarter|Semi|Final|Play-offs|Preliminary|Qualifying|Knockout)",
        str(round_name), re.IGNORECASE))


# ── Phase 1: Download new fixtures ──────────────────────────────────────────

def phase1_download_new():
    banner("FASE 1: Descargar partidos nuevos")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

    total_new_stats = 0
    total_new_players = 0
    new_per_comp = {}

    for league_id, info in LEAGUES.items():
        league_dir = RAW_DIR / info["dir"]
        print(f"\n  {info['name']} (league={league_id})")

        # Re-download fixtures.json for 2025 season to get latest matches
        season = 2025
        out_dir = league_dir / str(season)
        out_path = out_dir / "fixtures.json"

        print(f"    Actualizando fixtures 2025...")
        time.sleep(SLEEP)
        data = api_get("/fixtures", {"league": league_id, "season": season})
        if data and data.get("response"):
            save_json(data, out_path)
            total = len(data["response"])
            ft = sum(1 for f in data["response"] if f["fixture"]["status"]["short"] == "FT")
            print(f"    {ft} FT de {total} totales")
        else:
            print(f"    Error descargando fixtures")

        # Collect all FT fixtures across all seasons
        all_ft_fixtures = []
        for s in SEASONS:
            fx_path = league_dir / str(s) / "fixtures.json"
            if not fx_path.exists():
                continue
            with open(fx_path) as f:
                fdata = json.load(f)
            for fix in fdata.get("response", []):
                if fix["fixture"]["status"]["short"] == "FT":
                    fid = fix["fixture"]["id"]
                    label = f"{fix['teams']['home']['name']} vs {fix['teams']['away']['name']}"
                    all_ft_fixtures.append({"id": fid, "label": label})

        # Find fixtures missing stats or players
        stats_needed = [f for f in all_ft_fixtures if not file_ok(STATS_DIR / f"{f['id']}.json")]
        players_needed = [f for f in all_ft_fixtures if not file_ok(PLAYERS_DIR / f"{f['id']}.json")]

        new_per_comp[info["name"]] = len(stats_needed) + len(players_needed)
        print(f"    Stats pendientes: {len(stats_needed)}, Players pendientes: {len(players_needed)}")

        if stats_needed:
            print(f"    Descargando stats (~{fmt_time(len(stats_needed) * SLEEP)})...")
            for i, fix in enumerate(stats_needed):
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"      [{i+1}/{len(stats_needed)}] {fix['label']}...", flush=True)
                time.sleep(SLEEP)
                d = api_get("/fixtures/statistics", {"fixture": fix["id"]})
                if d:
                    save_json(d, STATS_DIR / f"{fix['id']}.json")
            total_new_stats += len(stats_needed)

        if players_needed:
            print(f"    Descargando players (~{fmt_time(len(players_needed) * SLEEP)})...")
            for i, fix in enumerate(players_needed):
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"      [{i+1}/{len(players_needed)}] {fix['label']}...", flush=True)
                time.sleep(SLEEP)
                d = api_get("/fixtures/players", {"fixture": fix["id"]})
                if d:
                    save_json(d, PLAYERS_DIR / f"{fix['id']}.json")
            total_new_players += len(players_needed)

    total_new = total_new_stats + total_new_players
    print(f"\n  Total descargas nuevas: {total_new} (stats: {total_new_stats}, players: {total_new_players})")
    for comp, count in new_per_comp.items():
        print(f"    {comp}: {count} nuevos")

    return total_new


# ── Phase 2: Feature Engineering ─────────────────────────────────────────────

def phase2_feature_engineering():
    banner("FASE 2: Feature Engineering Unificado")

    # Load all fixtures
    rows = []
    for league_id, info in LEAGUES.items():
        league_dir = RAW_DIR / info["dir"]
        for season in SEASONS:
            fx_path = league_dir / str(season) / "fixtures.json"
            if not fx_path.exists():
                continue
            with open(fx_path) as f:
                data = json.load(f)

            for match in data.get("response", []):
                if match["fixture"]["status"]["short"] != "FT":
                    continue
                fid = match["fixture"]["id"]
                hid = match["teams"]["home"]["id"]
                aid = match["teams"]["away"]["id"]
                hg = match["goals"]["home"]
                ag = match["goals"]["away"]
                if hg is None or ag is None:
                    continue

                row = {
                    "fixture_id": fid,
                    "date": pd.to_datetime(match["fixture"]["date"]),
                    "season": season,
                    "round": match["league"]["round"],
                    "league_id": league_id,
                    "league_name": info["name"],
                    "competition_level": info["level"],
                    "home_team_id": hid,
                    "home_team_name": match["teams"]["home"]["name"],
                    "away_team_id": aid,
                    "away_team_name": match["teams"]["away"]["name"],
                    "home_goals": hg, "away_goals": ag,
                    "result": determine_result(hg, ag),
                }

                # Statistics
                sp = STATS_DIR / f"{fid}.json"
                if sp.exists():
                    with open(sp) as f:
                        sd = json.load(f)
                    for te in sd.get("response", []):
                        tid = te["team"]["id"]
                        parsed = parse_team_stats(te["statistics"])
                        prefix = "home_" if tid == hid else ("away_" if tid == aid else None)
                        if prefix:
                            for k, v in parsed.items():
                                row[prefix + k] = v

                # Players
                pp = PLAYERS_DIR / f"{fid}.json"
                if pp.exists():
                    with open(pp) as f:
                        pd_data = json.load(f)
                    for te in pd_data.get("response", []):
                        tid = te["team"]["id"]
                        agg = compute_player_aggregates(te["players"])
                        prefix = "home_" if tid == hid else ("away_" if tid == aid else None)
                        if prefix:
                            for k, v in agg.items():
                                row[prefix + k] = v

                rows.append(row)

    matches_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(PROCESSED_DIR / "matches_all.csv", index=False)

    print(f"  Partidos totales: {len(matches_df)}")
    for lid, info in LEAGUES.items():
        n = len(matches_df[matches_df["league_id"] == lid])
        print(f"    {info['name']}: {n}")

    # Rolling features
    print("  Calculando rolling features...")
    team_rows = []
    for _, m in matches_df.iterrows():
        fid = m["fixture_id"]
        date = m["date"]
        if m["result"] == "H":
            hp, ap = 3, 0
        elif m["result"] == "A":
            hp, ap = 0, 3
        else:
            hp, ap = 1, 1

        hsa = (m.get("home_shots_on_goal", 0) / m.get("home_total_shots", 1)
               if m.get("home_total_shots") and m.get("home_total_shots") > 0 else None)
        asa = (m.get("away_shots_on_goal", 0) / m.get("away_total_shots", 1)
               if m.get("away_total_shots") and m.get("away_total_shots") > 0 else None)
        hdp = (m.get("home_duels_won", 0) / m.get("home_duels_total", 1)
               if m.get("home_duels_total") and m.get("home_duels_total") > 0 else None)
        adp = (m.get("away_duels_won", 0) / m.get("away_duels_total", 1)
               if m.get("away_duels_total") and m.get("away_duels_total") > 0 else None)

        base = {"fixture_id": fid, "date": date}
        team_rows.append({**base, "team_id": m["home_team_id"], "is_home": True,
            "goals_for": m["home_goals"], "goals_against": m["away_goals"],
            "xg_for": m.get("home_xg"), "xg_against": m.get("away_xg"),
            "shots_on_goal": m.get("home_shots_on_goal"), "shots_accuracy": hsa,
            "possession": m.get("home_possession"), "pass_accuracy": m.get("home_pass_pct"),
            "corner_kicks": m.get("home_corner_kicks"), "avg_rating": m.get("home_avg_rating"),
            "key_passes": m.get("home_key_passes"), "duels_won_pct": hdp,
            "dribbles_success": m.get("home_dribbles_success"), "points": hp})
        team_rows.append({**base, "team_id": m["away_team_id"], "is_home": False,
            "goals_for": m["away_goals"], "goals_against": m["home_goals"],
            "xg_for": m.get("away_xg"), "xg_against": m.get("home_xg"),
            "shots_on_goal": m.get("away_shots_on_goal"), "shots_accuracy": asa,
            "possession": m.get("away_possession"), "pass_accuracy": m.get("away_pass_pct"),
            "corner_kicks": m.get("away_corner_kicks"), "avg_rating": m.get("away_avg_rating"),
            "key_passes": m.get("away_key_passes"), "duels_won_pct": adp,
            "dribbles_success": m.get("away_dribbles_success"), "points": ap})

    team_df = pd.DataFrame(team_rows).sort_values(["team_id", "date"]).reset_index(drop=True)
    team_df["xg_diff"] = team_df["xg_for"] - team_df["xg_against"]
    team_df["xg_overperformance"] = team_df["goals_for"] - team_df["xg_for"]

    roll_cols = {
        "xg_for": "rolling_xg_for", "xg_against": "rolling_xg_against",
        "xg_diff": "rolling_xg_diff", "goals_for": "rolling_goals_for",
        "goals_against": "rolling_goals_against",
        "xg_overperformance": "rolling_xg_overperformance",
        "shots_on_goal": "rolling_shots_on_goal", "shots_accuracy": "rolling_shots_accuracy",
        "possession": "rolling_possession", "pass_accuracy": "rolling_pass_accuracy",
        "corner_kicks": "rolling_corners", "avg_rating": "rolling_avg_rating",
        "key_passes": "rolling_key_passes", "duels_won_pct": "rolling_duels_won_pct",
        "dribbles_success": "rolling_dribbles_success", "points": "rolling_points",
    }
    for src, dst in roll_cols.items():
        team_df[dst] = team_df.groupby("team_id")[src].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    team_df["days_since_last"] = team_df.groupby("team_id")["date"].transform(
        lambda s: s.diff().dt.days)

    rcols = list(roll_cols.values()) + ["days_since_last"]
    home_r = team_df[team_df["is_home"]][["fixture_id"] + rcols].rename(
        columns={c: f"home_{c}" for c in rcols})
    away_r = team_df[~team_df["is_home"]][["fixture_id"] + rcols].rename(
        columns={c: f"away_{c}" for c in rcols})

    features_df = matches_df.merge(home_r, on="fixture_id", how="left")
    features_df = features_df.merge(away_r, on="fixture_id", how="left")

    features_df["diff_rolling_xg"] = features_df["home_rolling_xg_for"] - features_df["away_rolling_xg_for"]
    features_df["diff_rolling_goals"] = features_df["home_rolling_goals_for"] - features_df["away_rolling_goals_for"]
    features_df["diff_rolling_form"] = features_df["home_rolling_points"] - features_df["away_rolling_points"]
    features_df["is_knockout"] = features_df["round"].apply(is_knockout_round).astype(int)
    features_df["is_champions"] = (features_df["league_id"] == 2).astype(int)
    features_df["is_europa"] = (features_df["league_id"] == 3).astype(int)
    features_df["is_conference"] = (features_df["league_id"] == 848).astype(int)

    print(f"  Features shape: {features_df.shape}")

    # ELO
    print("  Integrando ELO...")
    ELO_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for csv_file in ELO_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                snapshots[csv_file.stem] = df
        except Exception:
            pass

    all_dates = features_df["date"].dt.date.unique()
    needed = [(d, d.strftime("%Y-%m-%d")) for d in sorted(all_dates) if d.strftime("%Y-%m-%d") not in snapshots]
    if needed:
        print(f"    Descargando {len(needed)} snapshots ELO nuevos...")
        for i, (d, ds) in enumerate(needed):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"      [{i+1}/{len(needed)}] {ds}...", flush=True)
            time.sleep(2)
            try:
                r = requests.get(f"http://api.clubelo.com/{ds}", timeout=15)
                r.raise_for_status()
                text = r.text.strip()
                if len(text.split("\n")) > 1:
                    p = ELO_DIR / f"{ds}.csv"
                    with open(p, "w") as f:
                        f.write(text)
                    snapshots[ds] = pd.read_csv(p)
            except Exception:
                pass

    features_df["elo_home"] = np.nan
    features_df["elo_away"] = np.nan
    matched = 0
    for idx, row in features_df.iterrows():
        ds = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        if ds not in snapshots:
            continue
        elo_lookup = dict(zip(snapshots[ds]["Club"], snapshots[ds]["Elo"]))
        for side, tcol in [("home", "home_team_name"), ("away", "away_team_name")]:
            tn = row[tcol]
            en = NAME_MAP.get(tn)
            if en and en in elo_lookup:
                features_df.at[idx, f"elo_{side}"] = elo_lookup[en]
            elif en is None and tn in elo_lookup:
                features_df.at[idx, f"elo_{side}"] = elo_lookup[tn]
        if pd.notna(features_df.at[idx, "elo_home"]) and pd.notna(features_df.at[idx, "elo_away"]):
            matched += 1

    features_df["elo_diff"] = features_df["elo_home"] - features_df["elo_away"]
    features_df["elo_expected_home"] = 1 / (1 + 10 ** ((features_df["elo_away"] - features_df["elo_home"]) / 400))
    print(f"    ELO coverage: {matched}/{len(features_df)} ({matched/len(features_df)*100:.1f}%)")

    # Domestic features
    print("  Integrando domestic features...")
    tl_path = PROCESSED_DIR / "team_leagues.json"
    if tl_path.exists():
        with open(tl_path) as f:
            team_leagues = json.load(f)
        team_league_map = {}
        for tid, info in team_leagues.items():
            for ss, li in info.get("seasons", {}).items():
                if li and "league_id" in li:
                    team_league_map[(int(tid), int(ss))] = li["league_id"]

        # Load domestic matches
        domestic_rows = []
        for tid, info in team_leagues.items():
            for ss, li in info.get("seasons", {}).items():
                if li and "league_id" in li:
                    key = f"{li['league_id']}_{ss}"
                    dp = DOMESTIC_DIR / f"{key}_fixtures.json"
                    if not dp.exists():
                        continue
                    with open(dp) as f:
                        dd = json.load(f)
                    for fix in dd.get("response", []):
                        hg = fix["goals"]["home"]
                        ag = fix["goals"]["away"]
                        if hg is None or ag is None:
                            continue
                        dfid = fix["fixture"]["id"]
                        dr = {
                            "fixture_id": dfid,
                            "date": pd.to_datetime(fix["fixture"]["date"]),
                            "league_id": li["league_id"], "season": int(ss),
                            "home_team_id": fix["teams"]["home"]["id"],
                            "away_team_id": fix["teams"]["away"]["id"],
                            "home_goals": hg, "away_goals": ag,
                        }
                        sp = DOMESTIC_STATS_DIR / f"{dfid}.json"
                        if sp.exists():
                            with open(sp) as sf:
                                sdd = json.load(sf)
                            for te in sdd.get("response", []):
                                ttid = te["team"]["id"]
                                px = "home_" if ttid == dr["home_team_id"] else (
                                    "away_" if ttid == dr["away_team_id"] else None)
                                if px:
                                    for stat in te.get("statistics", []):
                                        if stat["type"] == "expected_goals":
                                            dr[f"{px}xg"] = parse_stat_value("expected_goals", stat["value"])
                        domestic_rows.append(dr)

        domestic_df = pd.DataFrame(domestic_rows)
        if len(domestic_df) > 0:
            domestic_df = domestic_df.sort_values("date").drop_duplicates("fixture_id").reset_index(drop=True)
        print(f"    Partidos domesticos: {len(domestic_df)}")

        # Compute domestic features
        dom_feats = [
            "domestic_points_last5", "domestic_goals_for_last5",
            "domestic_goals_against_last5", "domestic_xg_for_last5",
            "domestic_xg_against_last5", "domestic_league_position",
            "domestic_ppg", "domestic_win_rate",
            "domestic_home_goals_avg", "domestic_away_goals_avg",
        ]
        for px in ["home_", "away_"]:
            for feat in dom_feats:
                features_df[px + feat] = np.nan

        has_dom = 0
        for i, (idx, match) in enumerate(features_df.iterrows()):
            if (i + 1) % 500 == 0:
                print(f"      Procesando {i+1}/{len(features_df)}...", flush=True)
            cl_date = match["date"]
            season = match["season"]
            for side, tc in [("home", "home_team_id"), ("away", "away_team_id")]:
                tid = match[tc]
                lid = team_league_map.get((tid, season))
                if lid is None or len(domestic_df) == 0:
                    continue
                feats = _compute_domestic(domestic_df, tid, cl_date, lid, season)
                for k, v in feats.items():
                    features_df.at[idx, f"{side}_{k}"] = v
            if pd.notna(features_df.at[idx, "home_domestic_ppg"]) or pd.notna(features_df.at[idx, "away_domestic_ppg"]):
                has_dom += 1

        features_df["diff_domestic_position"] = (
            features_df["away_domestic_league_position"] - features_df["home_domestic_league_position"])
        print(f"    Cobertura domestica: {has_dom}/{len(features_df)} ({has_dom/len(features_df)*100:.1f}%)")
    else:
        print("    team_leagues.json no encontrado — domestic features = NaN")
        for px in ["home_", "away_"]:
            for feat in ["domestic_points_last5", "domestic_goals_for_last5",
                         "domestic_goals_against_last5", "domestic_xg_for_last5",
                         "domestic_xg_against_last5", "domestic_league_position",
                         "domestic_ppg", "domestic_win_rate",
                         "domestic_home_goals_avg", "domestic_away_goals_avg"]:
                features_df[px + feat] = np.nan
        features_df["diff_domestic_position"] = np.nan

    # Save
    out_path = PROCESSED_DIR / "features_all.csv"
    features_df.to_csv(out_path, index=False)
    print(f"\n  Guardado: {out_path} ({features_df.shape})")

    return features_df


def _compute_domestic(domestic_df, team_id, before_date, league_id, season, n=5):
    result = {k: np.nan for k in [
        "domestic_points_last5", "domestic_goals_for_last5",
        "domestic_goals_against_last5", "domestic_xg_for_last5",
        "domestic_xg_against_last5", "domestic_league_position",
        "domestic_ppg", "domestic_win_rate",
        "domestic_home_goals_avg", "domestic_away_goals_avg"]}

    lm = domestic_df[(domestic_df["league_id"] == league_id) & (domestic_df["season"] == season)]
    if len(lm) == 0:
        return result

    th = lm[(lm["home_team_id"] == team_id) & (lm["date"] < before_date)]
    ta = lm[(lm["away_team_id"] == team_id) & (lm["date"] < before_date)]

    trows = []
    for _, m in th.iterrows():
        trows.append({"date": m["date"], "goals_for": m["home_goals"], "goals_against": m["away_goals"],
            "xg_for": m.get("home_xg"), "xg_against": m.get("away_xg"), "is_home": True,
            "points": 3 if m["home_goals"] > m["away_goals"] else (1 if m["home_goals"] == m["away_goals"] else 0),
            "win": 1 if m["home_goals"] > m["away_goals"] else 0})
    for _, m in ta.iterrows():
        trows.append({"date": m["date"], "goals_for": m["away_goals"], "goals_against": m["home_goals"],
            "xg_for": m.get("away_xg"), "xg_against": m.get("home_xg"), "is_home": False,
            "points": 3 if m["away_goals"] > m["home_goals"] else (1 if m["away_goals"] == m["home_goals"] else 0),
            "win": 1 if m["away_goals"] > m["home_goals"] else 0})

    if not trows:
        return result

    tdf = pd.DataFrame(trows).sort_values("date")
    tp = len(tdf)
    result["domestic_ppg"] = tdf["points"].sum() / tp
    result["domestic_win_rate"] = tdf["win"].sum() / tp

    hm = tdf[tdf["is_home"]]
    am = tdf[~tdf["is_home"]]
    if len(hm) > 0:
        result["domestic_home_goals_avg"] = hm["goals_for"].mean()
    if len(am) > 0:
        result["domestic_away_goals_avg"] = am["goals_for"].mean()

    last = tdf.tail(n)
    result["domestic_points_last5"] = last["points"].sum()
    result["domestic_goals_for_last5"] = last["goals_for"].mean()
    result["domestic_goals_against_last5"] = last["goals_against"].mean()
    xf = last["xg_for"].dropna()
    xa = last["xg_against"].dropna()
    if len(xf) > 0:
        result["domestic_xg_for_last5"] = xf.mean()
    if len(xa) > 0:
        result["domestic_xg_against_last5"] = xa.mean()

    before = lm[lm["date"] < before_date]
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
            st = sorted(standings.keys(), key=lambda t: (standings[t]["pts"], standings[t]["gd"]), reverse=True)
            result["domestic_league_position"] = st.index(team_id) + 1

    return result


# ── Phase 3: Retrain Models ─────────────────────────────────────────────────

def phase3_retrain(features_df):
    banner("FASE 3: Re-entrenar los 3 modelos")

    le = LabelEncoder()
    le.fit(CLASSES)

    features_df = features_df.copy()
    features_df["date"] = pd.to_datetime(features_df["date"])
    features_df["season"] = features_df["season"].astype(int)

    def build_splits(df):
        df = df.sort_values("date").reset_index(drop=True)
        s23 = df[df["season"] == 2023]
        s24 = df[df["season"] == 2024].sort_values("date")
        s25 = df[df["season"] == 2025]
        mid = len(s24) // 2
        return [
            ("Split 1", s23, s24.iloc[:mid]),
            ("Split 2", pd.concat([s23, s24.iloc[:mid]]), s24.iloc[mid:]),
            ("Split 3", pd.concat([s23, s24]), s25),
        ]

    def build_splits_eval_filter(df, eval_filter):
        df = df.sort_values("date").reset_index(drop=True)
        s23 = df[df["season"] == 2023]
        s24 = df[df["season"] == 2024].sort_values("date")
        s25 = df[df["season"] == 2025]
        mid = len(s24) // 2
        splits = [
            ("Split 1", s23, s24.iloc[:mid]),
            ("Split 2", pd.concat([s23, s24.iloc[:mid]]), s24.iloc[mid:]),
            ("Split 3", pd.concat([s23, s24]), s25),
        ]
        return [(n, tr, va[eval_filter(va)]) for n, tr, va in splits if len(va[eval_filter(va)]) > 0]

    feats = [f for f in FINAL_FEATURES if f in features_df.columns]
    results = {}

    # --- Champions League: LogReg calibrada ---
    print("\n  Champions League: LogReg calibrada (C=0.01, L2)")
    cl_df = features_df[features_df["league_id"] == 2].copy()
    cl_splits = build_splits(cl_df)
    accs, lls, f1s = [], [], []
    for sname, tr, va in cl_splits:
        if len(tr) < 10 or len(va) < 10:
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.01, penalty="l2", max_iter=2000, random_state=42)),
        ])
        cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
        cal.fit(tr[feats].values, le.transform(tr["result"]))
        preds = cal.predict(va[feats].values)
        proba = cal.predict_proba(va[feats].values)
        a = accuracy_score(le.transform(va["result"]), preds)
        l = log_loss(le.transform(va["result"]), proba, labels=[0, 1, 2])
        f = f1_score(le.transform(va["result"]), preds, average="macro")
        accs.append(a); lls.append(l); f1s.append(f)
        print(f"    {sname}: Acc={a:.4f} LL={l:.4f} (train={len(tr)}, val={len(va)})")

    cl_acc = np.mean(accs)
    cl_ll = np.mean(lls)
    cl_f1 = np.mean(f1s)
    print(f"    AVG: Acc={cl_acc:.4f} LL={cl_ll:.4f} F1={cl_f1:.4f}")
    results["Champions League"] = {"accuracy": round(cl_acc, 4), "log_loss": round(cl_ll, 4), "f1_macro": round(cl_f1, 4)}

    # Save CL model (train on all CL data)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.01, penalty="l2", max_iter=2000, random_state=42)),
    ])
    cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
    cal.fit(cl_df[feats].values, le.transform(cl_df["result"]))
    joblib.dump(cal, MODELS_DIR / "champion_model_final.pkl")
    with open(MODELS_DIR / "feature_list_final.json", "w") as f:
        json.dump(feats, f, indent=2)

    # Update metrics_final.json
    mf = {"model": "LogReg tuned + cal(sigmoid)", "n_features": len(feats), "features": feats,
           "accuracy": round(cl_acc, 4), "log_loss": round(cl_ll, 4), "f1_macro": round(cl_f1, 4),
           "per_split": [{"accuracy": a, "log_loss": l, "f1_macro": f} for a, l, f in zip(accs, lls, f1s)]}
    with open(MODELS_DIR / "metrics_final.json", "w") as f:
        json.dump(mf, f, indent=2)

    # --- Europa League: RF, train on EL+ECL, eval on EL ---
    print("\n  Europa League: Random Forest (EL+ECL → eval EL)")
    el_ecl_df = features_df[(features_df["league_id"] == 3) | (features_df["league_id"] == 848)].copy()
    el_splits = build_splits_eval_filter(el_ecl_df, lambda v: v["is_europa"] == 1)
    accs, lls, f1s = [], [], []
    for sname, tr, va in el_splits:
        if len(tr) < 10 or len(va) < 10:
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                           class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
        cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
        cal.fit(tr[feats].values, le.transform(tr["result"]))
        preds = cal.predict(va[feats].values)
        proba = cal.predict_proba(va[feats].values)
        a = accuracy_score(le.transform(va["result"]), preds)
        l = log_loss(le.transform(va["result"]), proba, labels=[0, 1, 2])
        f = f1_score(le.transform(va["result"]), preds, average="macro")
        accs.append(a); lls.append(l); f1s.append(f)
        print(f"    {sname}: Acc={a:.4f} LL={l:.4f} (train={len(tr)}, val={len(va)})")

    el_acc = np.mean(accs)
    el_ll = np.mean(lls)
    el_f1 = np.mean(f1s)
    print(f"    AVG: Acc={el_acc:.4f} LL={el_ll:.4f} F1={el_f1:.4f}")
    results["Europa League"] = {"accuracy": round(el_acc, 4), "log_loss": round(el_ll, 4), "f1_macro": round(el_f1, 4)}

    # Save EL model
    el_only = features_df[features_df["league_id"] == 3]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
    cal.fit(el_ecl_df[feats].values, le.transform(el_ecl_df["result"]))
    joblib.dump(cal, MODELS_DIR / "europa_model.pkl")
    with open(MODELS_DIR / "europa_feature_list.json", "w") as f:
        json.dump(feats, f, indent=2)
    with open(MODELS_DIR / "europa_metrics.json", "w") as f:
        json.dump({"competition": "europa", "best_model": "RF", "best_variant": "B: EL+ECL→eval EL",
                    "accuracy": round(el_acc, 4), "log_loss": round(el_ll, 4), "f1_macro": round(el_f1, 4),
                    "n_matches_train": len(el_only), "n_features": len(feats)}, f, indent=2)

    # --- Conference League: RF, train on ALL, eval on ECL ---
    print("\n  Conference League: Random Forest (ALL → eval ECL)")
    ecl_splits = build_splits_eval_filter(features_df, lambda v: v["is_conference"] == 1)
    accs, lls, f1s = [], [], []
    for sname, tr, va in ecl_splits:
        if len(tr) < 10 or len(va) < 10:
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                           class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
        cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
        cal.fit(tr[feats].values, le.transform(tr["result"]))
        preds = cal.predict(va[feats].values)
        proba = cal.predict_proba(va[feats].values)
        a = accuracy_score(le.transform(va["result"]), preds)
        l = log_loss(le.transform(va["result"]), proba, labels=[0, 1, 2])
        f = f1_score(le.transform(va["result"]), preds, average="macro")
        accs.append(a); lls.append(l); f1s.append(f)
        print(f"    {sname}: Acc={a:.4f} LL={l:.4f} (train={len(tr)}, val={len(va)})")

    ecl_acc = np.mean(accs)
    ecl_ll = np.mean(lls)
    ecl_f1 = np.mean(f1s)
    print(f"    AVG: Acc={ecl_acc:.4f} LL={ecl_ll:.4f} F1={ecl_f1:.4f}")
    results["Conference League"] = {"accuracy": round(ecl_acc, 4), "log_loss": round(ecl_ll, 4), "f1_macro": round(ecl_f1, 4)}

    # Save ECL model
    ecl_only = features_df[features_df["league_id"] == 848]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
    cal.fit(features_df[feats].values, le.transform(features_df["result"]))
    joblib.dump(cal, MODELS_DIR / "conference_model.pkl")
    with open(MODELS_DIR / "conference_feature_list.json", "w") as f:
        json.dump(feats, f, indent=2)
    with open(MODELS_DIR / "conference_metrics.json", "w") as f:
        json.dump({"competition": "conference", "best_model": "RF", "best_variant": "C: ALL→eval ECL",
                    "accuracy": round(ecl_acc, 4), "log_loss": round(ecl_ll, 4), "f1_macro": round(ecl_f1, 4),
                    "n_matches_train": len(ecl_only), "n_features": len(feats)}, f, indent=2)

    print(f"\n  Modelos guardados en {MODELS_DIR}/")
    return results


# ── Phase 4: Update Upcoming & Odds ─────────────────────────────────────────

def phase4_update_upcoming():
    banner("FASE 4: Actualizar upcoming matches & odds")

    upcoming_all = []
    recent_all = []

    for league_id, league_name in [(2, "Champions League"), (3, "Europa League"), (848, "Conference League")]:
        print(f"\n  {league_name}...")

        # Upcoming (NS)
        time.sleep(SLEEP)
        data = api_get("/fixtures", {"league": league_id, "season": 2025, "status": "NS"})
        ns = data.get("response", []) if data else []
        print(f"    {len(ns)} pendientes")
        for fx in ns:
            upcoming_all.append({
                "fixture_id": fx["fixture"]["id"], "date": fx["fixture"]["date"],
                "status": "NS", "round": fx["league"]["round"],
                "league_id": league_id, "league_name": league_name,
                "home_team": fx["teams"]["home"]["name"], "home_team_id": fx["teams"]["home"]["id"],
                "away_team": fx["teams"]["away"]["name"], "away_team_id": fx["teams"]["away"]["id"],
                "home_goals": None, "away_goals": None, "ht_home": None, "ht_away": None,
            })

        # Recent (FT)
        time.sleep(SLEEP)
        data = api_get("/fixtures", {"league": league_id, "season": 2025, "status": "FT"})
        ft = data.get("response", []) if data else []
        ft_parsed = []
        for fx in ft:
            ft_parsed.append({
                "fixture_id": fx["fixture"]["id"], "date": fx["fixture"]["date"],
                "status": "FT", "round": fx["league"]["round"],
                "league_id": league_id, "league_name": league_name,
                "home_team": fx["teams"]["home"]["name"], "home_team_id": fx["teams"]["home"]["id"],
                "away_team": fx["teams"]["away"]["name"], "away_team_id": fx["teams"]["away"]["id"],
                "home_goals": fx["goals"]["home"], "away_goals": fx["goals"]["away"],
                "ht_home": fx.get("score", {}).get("halftime", {}).get("home"),
                "ht_away": fx.get("score", {}).get("halftime", {}).get("away"),
            })
        ft_parsed.sort(key=lambda x: x["date"], reverse=True)
        recent_all.extend(ft_parsed[:10])

    # Odds
    odds_rows = []
    if upcoming_all:
        print(f"\n  Descargando odds para {len(upcoming_all)} partidos...")
        for i, m in enumerate(upcoming_all):
            fid = m["fixture_id"]
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{len(upcoming_all)}] {m['home_team']} vs {m['away_team']}...")
            try:
                time.sleep(SLEEP)
                data = api_get("/odds", {"fixture": fid, "bookmaker": 4})
                resp = data.get("response", []) if data else []
                if resp:
                    for bk in resp[0].get("bookmakers", []):
                        if bk["id"] == 4:
                            for bet in bk.get("bets", []):
                                if bet["name"] == "Match Winner":
                                    oh = od = oa = None
                                    for v in bet["values"]:
                                        if v["value"] == "Home": oh = float(v["odd"])
                                        elif v["value"] == "Draw": od = float(v["odd"])
                                        elif v["value"] == "Away": oa = float(v["odd"])
                                    if oh and od and oa:
                                        odds_rows.append({"fixture_id": fid, "odd_home": oh, "odd_draw": od, "odd_away": oa})
                                    break
                            break
            except Exception:
                pass
        print(f"  Odds: {len(odds_rows)}/{len(upcoming_all)}")

    # Save
    pd.DataFrame(upcoming_all).to_csv(PROCESSED_DIR / "upcoming_matches.csv", index=False)
    pd.DataFrame(recent_all).to_csv(PROCESSED_DIR / "recent_results.csv", index=False)
    pd.DataFrame(odds_rows).to_csv(PROCESSED_DIR / "odds.csv", index=False)
    print(f"\n  Guardado: upcoming_matches.csv ({len(upcoming_all)}), recent_results.csv ({len(recent_all)}), odds.csv ({len(odds_rows)})")


# ── Phase 5: Compare ────────────────────────────────────────────────────────

def phase5_compare(new_results, features_df):
    banner("FASE 5: Comparacion antes vs despues")

    old = {
        "Champions League":  {"accuracy": 0.5721, "log_loss": 0.9331},
        "Europa League":     {"accuracy": 0.5937, "log_loss": 0.9477},
        "Conference League": {"accuracy": 0.5272, "log_loss": 1.0016},
    }

    old_matches = {"Champions League": 709, "Europa League": 650, "Conference League": 1112}
    new_matches = {}
    for lid, info in LEAGUES.items():
        new_matches[info["name"]] = len(features_df[features_df["league_id"] == lid])

    print(f"\n  {'Competicion':<20} {'Acc ANTES':>10} {'Acc DESPUES':>12} {'Delta':>8} {'Partidos +':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8} {'-'*12}")

    for comp in ["Champions League", "Europa League", "Conference League"]:
        oa = old[comp]["accuracy"]
        na = new_results[comp]["accuracy"]
        delta = na - oa
        added = new_matches[comp] - old_matches[comp]
        sign = "+" if delta >= 0 else ""
        print(f"  {comp:<20} {oa:>10.4f} {na:>12.4f} {sign}{delta:>7.4f} {'+' + str(added):>12}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("█" * 70)
    print("12 — REFRESH AND RETRAIN")
    print("█" * 70)

    if not API_KEY:
        sys.exit("ERROR: API_FOOTBALL_KEY not set. Export it first.")

    # Phase 1
    total_new = phase1_download_new()

    # Phase 2
    features_df = phase2_feature_engineering()

    # Phase 3
    results = phase3_retrain(features_df)

    # Phase 4
    phase4_update_upcoming()

    # Phase 5
    phase5_compare(results, features_df)

    # Phase 6: Git
    banner("FASE 6: Git commit & push")
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(BASE_DIR), check=True)
        subprocess.run(["git", "commit", "-m",
            "Refresh: retrain with matches through Feb 26, 2026\n\n"
            "Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"],
            cwd=str(BASE_DIR), check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=str(BASE_DIR), check=True)
        print("  Git push exitoso!")
    except subprocess.CalledProcessError as e:
        print(f"  Git error: {e}")
        print("  Ejecuta manualmente: git add -A && git commit && git push")

    print(f"\n{'█' * 70}")
    print("REFRESH COMPLETADO")
    print(f"{'█' * 70}")


if __name__ == "__main__":
    main()
