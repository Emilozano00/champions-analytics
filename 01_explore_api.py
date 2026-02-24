#!/usr/bin/env python3
"""
01_explore_api.py — Exploración multi-temporada de la API-Football para Champions League.

Seasons: 2023, 2024, 2025
League ID: 2 (UEFA Champions League)
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# ── Configuración ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_FOOTBALL_KEY")
if not API_KEY:
    sys.exit("ERROR: La variable de entorno API_FOOTBALL_KEY no está definida.")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 2
SEASONS = [2023, 2024, 2025]
RAW_DIR = Path.home() / "champions-analytics" / "data" / "raw" / "exploration"

SLEEP = 1  # segundos entre llamadas para respetar rate limits


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(title: str, char: str = "═", width: int = 90) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def sub_sep(title: str, char: str = "─", width: int = 70) -> None:
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}")


def api_get(endpoint: str, params: dict | None = None) -> dict:
    """Llama a la API, respeta rate limit, devuelve JSON."""
    url = f"{BASE_URL}{endpoint}"
    time.sleep(SLEEP)
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  ⚠  Error en {endpoint} con params={params}: {e}")
        return {"response": [], "errors": str(e)}


def save_json(data: dict, season: int, filename: str) -> None:
    out_dir = RAW_DIR / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → Guardado: {path}")


def safe_response(data: dict) -> list:
    """Extrae data['response'] de forma segura."""
    if isinstance(data, dict):
        r = data.get("response")
        if isinstance(r, list):
            return r
    return []


# ── SECCIÓN 1: Descarga de datos base por temporada ──────────────────────────

def fetch_season_data() -> dict:
    """Descarga leagues, teams, fixtures, rounds para cada temporada."""
    results = {}
    for season in SEASONS:
        sep(f"DESCARGANDO TEMPORADA {season}")
        s = {}

        # Leagues
        print(f"  [1/4] GET /leagues?id={LEAGUE_ID}&season={season}")
        s["leagues"] = api_get("/leagues", {"id": LEAGUE_ID, "season": season})
        save_json(s["leagues"], season, "leagues.json")

        # Teams
        print(f"  [2/4] GET /teams?league={LEAGUE_ID}&season={season}")
        s["teams"] = api_get("/teams", {"league": LEAGUE_ID, "season": season})
        save_json(s["teams"], season, "teams.json")

        # Fixtures
        print(f"  [3/4] GET /fixtures?league={LEAGUE_ID}&season={season}")
        s["fixtures"] = api_get("/fixtures", {"league": LEAGUE_ID, "season": season})
        save_json(s["fixtures"], season, "fixtures.json")

        # Rounds
        print(f"  [4/4] GET /fixtures/rounds?league={LEAGUE_ID}&season={season}")
        s["rounds"] = api_get("/fixtures/rounds", {"league": LEAGUE_ID, "season": season})
        save_json(s["rounds"], season, "rounds.json")

        results[season] = s
    return results


# ── SECCIÓN 2: Comparativo entre temporadas ──────────────────────────────────

def print_season_comparison(data: dict) -> dict:
    """Imprime comparativo y devuelve fixture_ids seleccionados."""
    sep("COMPARATIVO ENTRE TEMPORADAS", "█")

    total_finished = 0
    fixture_picks = {}  # season -> fixture_id de un partido finalizado

    for season in SEASONS:
        sub_sep(f"Temporada {season}")
        s = data[season]

        # Equipos
        teams = safe_response(s["teams"])
        print(f"  Equipos participantes: {len(teams)}")

        # Fixtures
        fixtures = safe_response(s["fixtures"])
        total = len(fixtures)
        finished = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "FT"]
        not_started = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "NS"]
        other = total - len(finished) - len(not_started)

        print(f"  Partidos totales:      {total}")
        print(f"  Partidos finalizados:  {len(finished)}")
        print(f"  Partidos pendientes:   {len(not_started)}")
        if other > 0:
            print(f"  Otros estados:         {other}")

        total_finished += len(finished)

        # Guardar un fixture finalizado para la sección 3
        if finished:
            pick = finished[len(finished) // 2]  # tomar uno del medio
            fid = pick["fixture"]["id"]
            home = pick["teams"]["home"]["name"]
            away = pick["teams"]["away"]["name"]
            fixture_picks[season] = {"id": fid, "label": f"{home} vs {away}"}
            print(f"  Fixture seleccionado:  {fid} ({home} vs {away})")

        # Rondas
        rounds = safe_response(s["rounds"])
        print(f"  Rondas/fases ({len(rounds)}):")
        for r in rounds:
            print(f"    • {r}")

    sub_sep("RESUMEN ACUMULADO")
    print(f"  Total partidos finalizados (todas las temporadas): {total_finished}")
    print(f"  → Disponibles para entrenar modelo predictivo")

    return fixture_picks


# ── SECCIÓN 3: Detalle de fixtures individuales ──────────────────────────────

def fetch_fixture_details(fixture_picks: dict) -> dict:
    """Descarga stats, events, players, odds para fixtures de 2023 y 2024."""
    sep("DESCARGA DE DETALLE POR FIXTURE (2023 vs 2024)", "█")

    details = {}
    for season in [2023, 2024]:
        if season not in fixture_picks:
            print(f"  ⚠  No hay fixture finalizado para {season}, saltando.")
            continue

        pick = fixture_picks[season]
        fid = pick["id"]
        sub_sep(f"Temporada {season} — Fixture {fid} ({pick['label']})")

        d = {}

        # Statistics
        print(f"  [1/4] GET /fixtures/statistics?fixture={fid}")
        d["statistics"] = api_get("/fixtures/statistics", {"fixture": fid})
        save_json(d["statistics"], season, f"fixture_{fid}_statistics.json")

        # Events
        print(f"  [2/4] GET /fixtures/events?fixture={fid}")
        d["events"] = api_get("/fixtures/events", {"fixture": fid})
        save_json(d["events"], season, f"fixture_{fid}_events.json")

        # Players
        print(f"  [3/4] GET /fixtures/players?fixture={fid}")
        d["players"] = api_get("/fixtures/players", {"fixture": fid})
        save_json(d["players"], season, f"fixture_{fid}_players.json")

        # Odds (Pinnacle = bookmaker 8)
        print(f"  [4/4] GET /odds?fixture={fid}&bookmaker=8")
        d["odds"] = api_get("/odds", {"fixture": fid, "bookmaker": 8})
        save_json(d["odds"], season, f"fixture_{fid}_odds_pinnacle.json")

        details[season] = d

    return details


# ── SECCIÓN 4: Comparativo de datos disponibles ─────────────────────────────

def compare_fixture_details(details: dict, fixture_picks: dict) -> None:
    sep("COMPARATIVO DE DATOS DISPONIBLES: 2023 vs 2024", "█")

    # ── 4a. Statistics keys ──
    sub_sep("STATISTICS: Keys disponibles")
    stats_keys = {}
    for season in [2023, 2024]:
        if season not in details:
            stats_keys[season] = set()
            continue
        stat_resp = safe_response(details[season]["statistics"])
        keys = set()
        for team_block in stat_resp:
            for stat in team_block.get("statistics", []):
                if stat.get("type"):
                    keys.add(stat["type"])
        stats_keys[season] = keys
        print(f"\n  Temporada {season} ({len(keys)} stats):")
        for k in sorted(keys):
            marker = " ⭐" if "expected" in k.lower() or "xg" in k.lower() else ""
            print(f"    • {k}{marker}")

    only_2023 = stats_keys.get(2023, set()) - stats_keys.get(2024, set())
    only_2024 = stats_keys.get(2024, set()) - stats_keys.get(2023, set())
    if only_2023:
        print(f"\n  Solo en 2023: {sorted(only_2023)}")
    if only_2024:
        print(f"\n  Solo en 2024: {sorted(only_2024)}")
    if not only_2023 and not only_2024:
        print("\n  ✓ Mismas keys de statistics en ambas temporadas.")

    # ── 4b. Player stats keys ──
    sub_sep("PLAYERS: Keys de estadísticas de jugador")
    player_keys = {}
    for season in [2023, 2024]:
        if season not in details:
            player_keys[season] = set()
            continue
        players_resp = safe_response(details[season]["players"])
        keys = set()
        for team_block in players_resp:
            for player_entry in team_block.get("players", []):
                for stat_block in player_entry.get("statistics", []):
                    for category, val in stat_block.items():
                        if isinstance(val, dict):
                            for k in val:
                                keys.add(f"{category}.{k}")
                        else:
                            keys.add(category)
        player_keys[season] = keys
        print(f"\n  Temporada {season} ({len(keys)} keys):")
        for k in sorted(keys):
            marker = " ⭐" if "expected" in k.lower() or "xg" in k.lower() else ""
            print(f"    • {k}{marker}")

    only_2023_p = player_keys.get(2023, set()) - player_keys.get(2024, set())
    only_2024_p = player_keys.get(2024, set()) - player_keys.get(2023, set())
    if only_2023_p:
        print(f"\n  Solo en 2023: {sorted(only_2023_p)}")
    if only_2024_p:
        print(f"\n  Solo en 2024: {sorted(only_2024_p)}")
    if not only_2023_p and not only_2024_p:
        print("\n  ✓ Mismas keys de player stats en ambas temporadas.")

    # ── 4c. xG highlight ──
    sub_sep("xG (Expected Goals) — Búsqueda explícita")
    for season in [2023, 2024]:
        found_xg = False
        if season in details:
            # En statistics
            for team_block in safe_response(details[season]["statistics"]):
                for stat in team_block.get("statistics", []):
                    if "expected" in stat.get("type", "").lower():
                        print(f"  ⭐ Temporada {season} — STATISTICS tiene: {stat['type']} = {stat['value']}")
                        found_xg = True
            # En players
            for team_block in safe_response(details[season]["players"]):
                for player_entry in team_block.get("players", []):
                    pname = player_entry.get("player", {}).get("name", "?")
                    for stat_block in player_entry.get("statistics", []):
                        for category, val in stat_block.items():
                            if isinstance(val, dict):
                                for k, v in val.items():
                                    if "expected" in k.lower() and v is not None:
                                        print(f"  ⭐ Temporada {season} — PLAYER {pname}: {category}.{k} = {v}")
                                        found_xg = True
        if not found_xg:
            print(f"  ✗ Temporada {season} — No se encontró xG")

    # ── 4d. Odds de Pinnacle ──
    sub_sep("ODDS de Pinnacle (bookmaker=8)")
    for season in [2023, 2024]:
        if season not in details:
            print(f"  Temporada {season}: sin datos de fixture")
            continue
        odds_resp = safe_response(details[season]["odds"])
        if odds_resp:
            print(f"  ✓ Temporada {season}: Odds disponibles ({len(odds_resp)} registro(s))")
            for entry in odds_resp[:1]:  # solo el primero como ejemplo
                for bm in entry.get("bookmakers", []):
                    print(f"    Bookmaker: {bm.get('name')}")
                    for bet in bm.get("bets", []):
                        print(f"      Mercado: {bet.get('name')}")
                        for val in bet.get("values", []):
                            print(f"        {val.get('value')}: {val.get('odd')}")
        else:
            print(f"  ✗ Temporada {season}: Sin odds de Pinnacle para este fixture")

    # ── 4e. Valores reales de un partido 2024 ──
    if 2024 in details and 2024 in fixture_picks:
        sub_sep(f"VALORES REALES — Partido ejemplo 2024: {fixture_picks[2024]['label']}")

        stat_resp = safe_response(details[2024]["statistics"])
        for team_block in stat_resp:
            team_name = team_block.get("team", {}).get("name", "?")
            print(f"\n  📊 {team_name}:")
            for stat in team_block.get("statistics", []):
                marker = " ⭐" if "expected" in stat.get("type", "").lower() else ""
                print(f"    {stat['type']:.<40s} {stat['value']}{marker}")


# ── SECCIÓN 5: Verificar temporada 2025 ─────────────────────────────────────

def check_2025(data: dict, fixture_picks: dict) -> None:
    sep("VERIFICACIÓN TEMPORADA 2025 (actual 2025-26)", "█")

    s = data.get(2025)
    if not s:
        print("  ⚠  No se descargó data de 2025.")
        return

    fixtures = safe_response(s["fixtures"])
    teams = safe_response(s["teams"])
    rounds = safe_response(s["rounds"])

    print(f"  Equipos:              {len(teams)}")
    print(f"  Partidos totales:     {len(fixtures)}")

    if fixtures:
        finished = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "FT"]
        scheduled = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "NS"]
        print(f"  Partidos finalizados: {len(finished)}")
        print(f"  Partidos pendientes:  {len(scheduled)}")

        statuses = {}
        for f in fixtures:
            st = f.get("fixture", {}).get("status", {}).get("long", "Unknown")
            statuses[st] = statuses.get(st, 0) + 1
        print("  Desglose por estado:")
        for st, count in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"    • {st}: {count}")
    else:
        print("  → No hay fixtures aún para 2025.")

    if rounds:
        print(f"  Rondas/fases ({len(rounds)}):")
        for r in rounds:
            print(f"    • {r}")
    else:
        print("  → No hay rondas definidas aún para 2025.")

    # Si hay un fixture finalizado en 2025, descargar su detalle también
    if 2025 in fixture_picks:
        fid = fixture_picks[2025]["id"]
        label = fixture_picks[2025]["label"]
        sub_sep(f"Detalle fixture 2025: {fid} ({label})")

        stats = api_get("/fixtures/statistics", {"fixture": fid})
        save_json(stats, 2025, f"fixture_{fid}_statistics.json")
        stat_resp = safe_response(stats)
        if stat_resp:
            for team_block in stat_resp:
                team_name = team_block.get("team", {}).get("name", "?")
                print(f"\n  📊 {team_name}:")
                for stat in team_block.get("statistics", []):
                    marker = " ⭐" if "expected" in stat.get("type", "").lower() else ""
                    print(f"    {stat['type']:.<40s} {stat['value']}{marker}")
        else:
            print("  → Sin statistics para este fixture de 2025.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sep("API-FOOTBALL EXPLORER — Champions League Multi-Temporada", "█")
    print(f"  League ID: {LEAGUE_ID}")
    print(f"  Seasons:   {SEASONS}")
    print(f"  Output:    {RAW_DIR}")
    print(f"  API Key:   {API_KEY[:6]}...{API_KEY[-4:]}" if len(API_KEY) > 10 else f"  API Key: {API_KEY[:4]}...")

    # Sección 1
    data = fetch_season_data()

    # Sección 2
    fixture_picks = print_season_comparison(data)

    # Sección 3
    details = fetch_fixture_details(fixture_picks)

    # Sección 4
    compare_fixture_details(details, fixture_picks)

    # Sección 5
    check_2025(data, fixture_picks)

    sep("EXPLORACIÓN COMPLETA", "█")
    print("  Archivos raw guardados en:")
    for season in SEASONS:
        d = RAW_DIR / str(season)
        files = sorted(d.glob("*.json")) if d.exists() else []
        print(f"    {d}/ ({len(files)} archivos)")
    print()


if __name__ == "__main__":
    main()
