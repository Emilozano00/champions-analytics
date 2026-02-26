#!/usr/bin/env python3
"""
11_update_fixtures.py — Update upcoming fixtures, recent results & odds.

Re-run weekly to keep the Streamlit app current.

Downloads:
  - Upcoming fixtures (status=NS) for CL, EL, ECL
  - Recent finished fixtures (status=FT, last 10 per competition)
  - Pinnacle odds (bookmaker_id=4) for upcoming fixtures

Saves:
  - data/processed/upcoming_matches.csv
  - data/processed/recent_results.csv

API calls: ~3 (fixtures) + ~3 (FT) + N (odds) ≈ 10-30 calls
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
SLEEP = 6.5
SEASON = 2025

LEAGUES = {
    2:   "Champions League",
    3:   "Europa League",
    848: "Conference League",
}

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(endpoint, params):
    """Make an API call with rate limiting."""
    if not API_KEY:
        print("  ERROR: API_FOOTBALL_KEY not set")
        sys.exit(1)
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    time.sleep(SLEEP)
    return data


def parse_fixture(fx, league_name):
    """Parse a single fixture from API response."""
    f = fx["fixture"]
    t = fx["teams"]
    g = fx["goals"]
    score = fx.get("score", {})
    return {
        "fixture_id": f["id"],
        "date": f["date"],
        "status": f["status"]["short"],
        "round": fx["league"]["round"],
        "league_id": fx["league"]["id"],
        "league_name": league_name,
        "home_team": t["home"]["name"],
        "home_team_id": t["home"]["id"],
        "away_team": t["away"]["name"],
        "away_team_id": t["away"]["id"],
        "home_goals": g.get("home"),
        "away_goals": g.get("away"),
        "ht_home": score.get("halftime", {}).get("home"),
        "ht_away": score.get("halftime", {}).get("away"),
    }


def banner(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("█" * 70)
    print("11 — UPDATE FIXTURES & ODDS")
    print("█" * 70)

    upcoming_all = []
    recent_all = []

    # ── Download upcoming (NS) and recent (FT) for each competition ──────
    for league_id, league_name in LEAGUES.items():
        banner(f"{league_name} (league={league_id})")

        # Upcoming (Not Started)
        print(f"  Descargando partidos pendientes (NS)...")
        data = api_get("fixtures", {
            "league": league_id,
            "season": SEASON,
            "status": "NS",
        })
        ns_fixtures = data.get("response", [])
        print(f"    {len(ns_fixtures)} partidos pendientes")

        for fx in ns_fixtures:
            upcoming_all.append(parse_fixture(fx, league_name))

        # Recent finished (FT) — get last 15 to have margin
        print(f"  Descargando resultados recientes (FT)...")
        data = api_get("fixtures", {
            "league": league_id,
            "season": SEASON,
            "status": "FT",
        })
        ft_fixtures = data.get("response", [])
        print(f"    {len(ft_fixtures)} partidos finalizados en total")

        # Sort by date descending, take last 10
        ft_parsed = [parse_fixture(fx, league_name) for fx in ft_fixtures]
        ft_parsed.sort(key=lambda x: x["date"], reverse=True)
        recent_all.extend(ft_parsed[:10])

    # ── Download odds for upcoming fixtures ──────────────────────────────
    banner("ODDS (Pinnacle, bookmaker_id=4)")

    odds_rows = []
    if upcoming_all:
        print(f"  Descargando odds para {len(upcoming_all)} partidos...")
        for i, match in enumerate(upcoming_all):
            fid = match["fixture_id"]
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{len(upcoming_all)}] {match['home_team']} vs {match['away_team']}...")

            try:
                data = api_get("odds", {
                    "fixture": fid,
                    "bookmaker": 4,
                })
                resp = data.get("response", [])
                if resp:
                    bookmakers = resp[0].get("bookmakers", [])
                    for bk in bookmakers:
                        if bk["id"] == 4:  # Pinnacle
                            for bet in bk.get("bets", []):
                                if bet["name"] == "Match Winner":
                                    odd_home = odd_draw = odd_away = None
                                    for val in bet["values"]:
                                        if val["value"] == "Home":
                                            odd_home = float(val["odd"])
                                        elif val["value"] == "Draw":
                                            odd_draw = float(val["odd"])
                                        elif val["value"] == "Away":
                                            odd_away = float(val["odd"])
                                    if odd_home and odd_draw and odd_away:
                                        odds_rows.append({
                                            "fixture_id": fid,
                                            "odd_home": odd_home,
                                            "odd_draw": odd_draw,
                                            "odd_away": odd_away,
                                        })
                                    break
                            break
            except Exception as e:
                print(f"    FAIL odds {fid}: {e}")

        print(f"  Odds obtenidos: {len(odds_rows)} de {len(upcoming_all)} partidos")
    else:
        print("  No hay partidos pendientes — sin odds que descargar.")

    # ── Save CSVs ────────────────────────────────────────────────────────
    banner("GUARDAR DATOS")

    # Upcoming matches
    upcoming_df = pd.DataFrame(upcoming_all)
    if len(upcoming_df) > 0:
        upcoming_df = upcoming_df.sort_values("date")
    upcoming_path = PROCESSED_DIR / "upcoming_matches.csv"
    upcoming_df.to_csv(upcoming_path, index=False)
    print(f"  Guardado: {upcoming_path} ({len(upcoming_df)} partidos)")

    # Recent results
    recent_df = pd.DataFrame(recent_all)
    if len(recent_df) > 0:
        recent_df = recent_df.sort_values("date", ascending=False)
    recent_path = PROCESSED_DIR / "recent_results.csv"
    recent_df.to_csv(recent_path, index=False)
    print(f"  Guardado: {recent_path} ({len(recent_df)} partidos)")

    # Odds
    odds_df = pd.DataFrame(odds_rows)
    odds_path = PROCESSED_DIR / "odds.csv"
    odds_df.to_csv(odds_path, index=False)
    print(f"  Guardado: {odds_path} ({len(odds_df)} odds)")

    # ── Summary ──────────────────────────────────────────────────────────
    banner("RESUMEN")

    for league_name in LEAGUES.values():
        up_count = len(upcoming_df[upcoming_df["league_name"] == league_name]) if len(upcoming_df) > 0 else 0
        rec_count = len(recent_df[recent_df["league_name"] == league_name]) if len(recent_df) > 0 else 0
        print(f"  {league_name:<25} Proximos: {up_count:<5} Recientes: {rec_count}")

    # Show upcoming rounds
    if len(upcoming_df) > 0:
        print("\n  Proximas fases:")
        for league_name in LEAGUES.values():
            sub = upcoming_df[upcoming_df["league_name"] == league_name]
            if len(sub) > 0:
                rounds = sub["round"].unique()
                first_date = pd.to_datetime(sub["date"].min()).strftime("%d %b %Y")
                print(f"    {league_name}: {', '.join(rounds)} (desde {first_date})")

    total_calls = len(LEAGUES) * 2 + len(upcoming_all)
    print(f"\n  API calls realizadas: ~{total_calls}")
    print(f"\n{'█' * 70}")
    print("ACTUALIZACIÓN COMPLETADA")
    print(f"{'█' * 70}")


if __name__ == "__main__":
    main()
