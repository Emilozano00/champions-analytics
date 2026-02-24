#!/usr/bin/env python3
"""
02_download_all_data.py — Descarga masiva de statistics, players y odds
para todos los fixtures de Champions League (2023, 2024, 2025).

Checkpoint/resume: si un archivo ya existe en disco, se salta.
Rate limit: 6.5s entre llamadas (plan Pro = 10 req/min).
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
SLEEP = 6.5  # segundos entre llamadas

BASE_DIR = Path.home() / "champions-analytics"
EXPLORATION_DIR = BASE_DIR / "data" / "raw" / "exploration"
STATS_DIR = BASE_DIR / "data" / "raw" / "statistics"
PLAYERS_DIR = BASE_DIR / "data" / "raw" / "players"
ODDS_DIR = BASE_DIR / "data" / "raw" / "odds"


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            return None
        return data
    except requests.RequestException:
        return None


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_exists_and_valid(path: Path) -> bool:
    """True si el archivo existe, no está vacío y tiene response."""
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


# ── PASO 1: Cargar fixtures desde exploración ────────────────────────────────

def load_fixtures() -> tuple[list[dict], list[dict]]:
    """Carga fixtures finalizados y pendientes de los JSONs de exploración."""
    sep("PASO 1: Cargando fixtures desde exploración")

    finished = []
    pending = []

    for season in SEASONS:
        fixtures_path = EXPLORATION_DIR / str(season) / "fixtures.json"
        if not fixtures_path.exists():
            print(f"  ⚠  No encontrado: {fixtures_path}")
            continue

        with open(fixtures_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fixtures = data.get("response", [])
        season_ft = 0
        season_pending = 0

        for fix in fixtures:
            fix_info = fix.get("fixture", {})
            status = fix_info.get("status", {}).get("short", "")
            fix_id = fix_info.get("id")
            home = fix.get("teams", {}).get("home", {}).get("name", "?")
            away = fix.get("teams", {}).get("away", {}).get("name", "?")

            entry = {
                "id": fix_id,
                "season": season,
                "home": home,
                "away": away,
                "status": status,
                "label": f"{home} vs {away}",
            }

            if status == "FT":
                finished.append(entry)
                season_ft += 1
            else:
                pending.append(entry)
                season_pending += 1

        print(f"  Temporada {season}: {season_ft} finalizados, {season_pending} otros")

    print(f"\n  TOTAL: {len(finished)} finalizados, {len(pending)} pendientes/otros")
    return finished, pending


# ── PASO 2 y 3: Descarga con checkpoint ──────────────────────────────────────

def download_with_checkpoint(
    fixtures: list[dict],
    endpoint: str,
    param_key: str,
    output_dir: Path,
    label: str,
) -> dict:
    """Descarga datos para cada fixture con checkpoint/resume.

    Returns dict con conteos: downloaded, skipped, failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(fixtures)
    downloaded = 0
    skipped = 0
    failed = 0
    failed_ids = []

    for i, fix in enumerate(fixtures, 1):
        fix_id = fix["id"]
        fix_label = fix["label"]
        out_path = output_dir / f"{fix_id}.json"

        pct = (i / total) * 100
        prefix = f"  {label} {i}/{total} ({pct:5.1f}%)"

        # Checkpoint: si ya existe, saltar
        if file_exists_and_valid(out_path):
            skipped += 1
            continue

        print(f"{prefix} — {fix_label}...", end="", flush=True)

        # Rate limit antes de la llamada
        time.sleep(SLEEP)

        data = api_get(endpoint, {param_key: fix_id})
        if data is not None:
            save_json(data, out_path)
            downloaded += 1
            print(" ✓")
        else:
            failed += 1
            failed_ids.append(fix_id)
            print(" ✗ FAIL")

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failed_ids": failed_ids,
        "total": total,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    sep("DESCARGA MASIVA — Champions League Multi-Temporada", "█")

    # Paso 1
    finished, pending = load_fixtures()

    # Filtrar pendientes de 2025 para odds
    pending_2025 = [f for f in pending if f["season"] == 2025 and f["status"] == "NS"]

    # Contar cuántas llamadas necesitamos realmente (sin las ya descargadas)
    calls_stats = sum(
        1 for f in finished
        if not file_exists_and_valid(STATS_DIR / f"{f['id']}.json")
    )
    calls_players = sum(
        1 for f in finished
        if not file_exists_and_valid(PLAYERS_DIR / f"{f['id']}.json")
    )
    calls_odds = sum(
        1 for f in pending_2025
        if not file_exists_and_valid(ODDS_DIR / f"{f['id']}.json")
    )
    total_calls = calls_stats + calls_players + calls_odds
    est_seconds = total_calls * SLEEP
    already_done = (len(finished) * 2 + len(pending_2025)) - total_calls

    sep("ESTIMACIÓN DE TIEMPO")
    print(f"  Fixtures finalizados:    {len(finished)}")
    print(f"  Pendientes 2025 (odds):  {len(pending_2025)}")
    print(f"  Llamadas ya completadas: {already_done} (checkpoint)")
    print(f"  Llamadas pendientes:     {total_calls}")
    print(f"    - statistics:          {calls_stats}")
    print(f"    - players:             {calls_players}")
    print(f"    - odds:                {calls_odds}")
    print(f"  Sleep entre llamadas:    {SLEEP}s")
    print(f"  Tiempo estimado:         {fmt_time(est_seconds)}")

    if total_calls == 0:
        sep("TODO YA DESCARGADO")
        print("  No hay llamadas pendientes. Todos los archivos ya existen.")
        return

    print(f"\n  Iniciando descarga...")

    # Paso 2a: Statistics
    sep("PASO 2a: Descargando statistics de fixtures finalizados")
    stats_result = download_with_checkpoint(
        finished, "/fixtures/statistics", "fixture", STATS_DIR, "stats"
    )

    # Paso 2b: Players
    sep("PASO 2b: Descargando players de fixtures finalizados")
    players_result = download_with_checkpoint(
        finished, "/fixtures/players", "fixture", PLAYERS_DIR, "players"
    )

    # Paso 3: Odds de partidos pendientes 2025
    sep("PASO 3: Descargando odds (Pinnacle) de fixtures pendientes 2025")
    if pending_2025:
        odds_result = download_with_checkpoint(
            pending_2025, "/odds", "fixture", ODDS_DIR, "odds"
        )
    else:
        print("  No hay fixtures pendientes en 2025.")
        odds_result = {"downloaded": 0, "skipped": 0, "failed": 0, "failed_ids": [], "total": 0}

    # Resumen final
    t_end = time.time()
    elapsed = t_end - t_start

    sep("RESUMEN FINAL", "█")

    for name, res in [("Statistics", stats_result), ("Players", players_result), ("Odds", odds_result)]:
        ok = res["skipped"] + res["downloaded"]
        print(f"\n  {name}:")
        print(f"    Total fixtures:  {res['total']}")
        print(f"    Ya existían:     {res['skipped']}")
        print(f"    Descargados hoy: {res['downloaded']}")
        print(f"    Fallidos:        {res['failed']}")
        print(f"    Completos:       {ok}/{res['total']}")
        if res["failed_ids"]:
            print(f"    IDs fallidos:    {res['failed_ids'][:20]}")
            if len(res["failed_ids"]) > 20:
                print(f"                     ... y {len(res['failed_ids']) - 20} más")

    print(f"\n  Tiempo total: {fmt_time(elapsed)}")

    total_failed = stats_result["failed"] + players_result["failed"] + odds_result["failed"]
    if total_failed > 0:
        print(f"\n  ⚠  {total_failed} llamadas fallaron. Re-ejecuta el script para reintentar")
        print(f"     (el checkpoint saltará las ya descargadas).")
    else:
        print(f"\n  ✓  Descarga completa sin errores.")

    print()


if __name__ == "__main__":
    main()
