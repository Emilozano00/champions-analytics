#!/usr/bin/env python3
"""
07_elo_ratings.py — Scrape ELO ratings from clubelo.com and integrate into model.

Uses date snapshot endpoint: http://api.clubelo.com/YYYY-MM-DD
One call per unique match date (~127 calls, ~4 minutes).
"""

import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ELO_DIR = BASE_DIR / "data" / "raw" / "elo"
MODELS_DIR = BASE_DIR / "models"
SLEEP = 2

# ---------------------------------------------------------------------------
# Name mapping: our team names → clubelo names
# ---------------------------------------------------------------------------

NAME_MAP = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle": "Newcastle",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Tottenham": "Tottenham",
    "Aston Villa": "Aston Villa",
    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico",
    "Real Sociedad": "Sociedad",
    "Sevilla": "Sevilla",
    "Villarreal": "Villarreal",
    "Girona": "Girona",
    "Athletic Club": "Bilbao",
    # Bundesliga
    "Bayern München": "Bayern",
    "Bayern Munich": "Bayern",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Frankfurt",
    "Bayer Leverkusen": "Leverkusen",
    "Union Berlin": "Union Berlin",
    "VfB Stuttgart": "Stuttgart",
    # Serie A
    "Inter": "Inter",
    "AC Milan": "AC Milan",
    "Napoli": "Napoli",
    "Juventus": "Juventus",
    "Atalanta": "Atalanta",
    "Lazio": "Lazio",
    "Bologna": "Bologna",
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Marseille": "Marseille",
    "Lens": "Lens",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Nice": "Nice",
    "Stade Brestois 29": "Brest",
    # Portugal
    "Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting",
    "SC Braga": "Braga",
    # Netherlands
    "PSV Eindhoven": "PSV",
    "Ajax": "Ajax",
    "Feyenoord": "Feyenoord",
    "Twente": "Twente",
    # Belgium
    "Club Brugge KV": "Brugge",
    "Antwerp": "Antwerp",
    "Genk": "Genk",
    "Union St. Gilloise": "Union SG",
    # Turkey
    "Galatasaray": "Galatasaray",
    "Fenerbahçe": "Fenerbahce",
    # Scotland
    "Celtic": "Celtic",
    "Rangers": "Rangers",
    # Austria
    "Red Bull Salzburg": "Salzburg",
    "Sturm Graz": "Sturm Graz",
    # Switzerland
    "BSC Young Boys": "Young Boys",
    "FC Basel 1893": "Basel",
    "FC Lugano": "Lugano",
    "Servette FC": "Servette",
    # Czech Republic
    "Sparta Praha": "Sparta Praha",
    "Slavia Praha": "Slavia Praha",
    "Plzen": "Viktoria Plzen",
    # Denmark
    "FC Copenhagen": "Koebenhavn",
    "FC Midtjylland": "Midtjylland",
    # Sweden
    "Malmo FF": "Malmoe",
    "BK Hacken": "Hacken",
    # Norway
    "Molde": "Molde",
    "Bodo/Glimt": "Bodoe Glimt",
    "Brann": "Brann",
    # Greece
    "Olympiakos Piraeus": "Olympiakos",
    "PAOK": "PAOK",
    "Panathinaikos": "Panathinaikos",
    "AEK Athens FC": "AEK",
    "Aris": "Aris",
    # Croatia
    "Dinamo Zagreb": "Dinamo Zagreb",
    "HNK Rijeka": "Rijeka",
    # Serbia
    "FK Crvena Zvezda": "Crvena Zvezda",
    "FK Partizan": "Partizan",
    "TSC Backa Topola": "Backa Topola",
    # Ukraine
    "Shakhtar Donetsk": "Shakhtar",
    "Dynamo Kyiv": "Dynamo Kyiv",
    "Dnipro-1": "Dnipro",
    # Romania
    "FCSB": "FCSB",
    "Farul Constanta": "Farul",
    # Hungary
    "Ferencvarosi TC": "Ferencvaros",
    # Cyprus
    "Apoel Nicosia": "APOEL",
    "Pafos": "Paphos",
    # Poland
    "Lech Poznan": "Lech",
    "Raków Częstochowa": "Rakow",
    "Jagiellonia": "Jagiellonia",
    # Bulgaria
    "Ludogorets": "Razgrad",
    # Israel
    "Maccabi Haifa": "Maccabi Haifa",
    "Maccabi Tel Aviv": "Maccabi Tel-Aviv",
    # Slovakia
    "Slovan Bratislava": "Slovan Bratislava",
    # Kazakhstan
    "FC Astana": "FK Astana",
    "Ordabasy": "Ordabasy",
    "Kairat Almaty": "Kairat",
    # Moldova
    "Sheriff Tiraspol": "Sheriff Tiraspol",
    "Milsami Orhei": "Milsami Orhei",
    "Petrocub": "Petrocub",
    # Finland
    "HJK helsinki": "HJK Helsinki",
    "KuPS": "Kuopio",
    # Ireland
    "Shamrock Rovers": "Shamrock Rovers",
    "Shelbourne": "Shelbourne",
    # Northern Ireland
    "Linfield": "Linfield",
    "Larne": "Larne",
    # Iceland
    "Breidablik": "Breidablik",
    "Vikingur Reykjavik": "Vikingur",
    # Faroe Islands
    "KI Klaksvik": "Klaksvik",
    "Vikingur Gota": "Vikingur Gota",
    # North Macedonia
    "Shkendija": "Shkendija",
    "Struga": "Struga",
    # Albania
    "Partizani": "Partizani Tirana",
    "Egnatia Rrogozhinë": "Egnatia",
    # Kosovo
    "Drita": "Drita",
    "Ballkani": "Ballkani",
    # Bosnia
    "Zrinjski": "Zrinjski Mostar",
    "Borac Banja Luka": "Borac Banja Luka",
    # Lithuania
    "FK Zalgiris Vilnius": "Zalgiris",
    "Panevėžys": "Panevezys",
    # Estonia
    "Flora Tallinn": "Flora Tallinn",
    "FC Levadia Tallinn": "Levadia",
    # Latvia
    "Rīgas FS": "Rigas",
    "Valmiera / BSS": "Valmiera",
    # Georgia
    "Dinamo Tbilisi": "Dinamo Tbilisi",
    "Dinamo Batumi": "Batumi",
    "Saburtalo": "Saburtalo",
    "FC Noah": "Noah",
    # Armenia
    "Pyunik Yerevan": "Pyunik",
    "FC Urartu": "Urartu",
    # Montenegro
    "Buducnost Podgorica": "Buducnost",
    "Dečić": "Decic",
    # Belarus
    "Bate Borisov": "BATE",
    "Dinamo Minsk": "Dinamo Minsk",
    # Slovenia
    "Olimpija Ljubljana": "Olimpija Ljubljana",
    "Celje": "Celje",
    # San Marino
    "Tre Penne": "Tre Penne",
    "Virtus": "SS Virtus",
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
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ---------------------------------------------------------------------------
# Phase 1: Download ELO snapshots by date
# ---------------------------------------------------------------------------

def download_elo_snapshots(dates: list) -> dict:
    """Download ELO snapshots for each unique date. Returns {date_str: DataFrame}."""
    sep("FASE 1-2: Descargar ELO ratings por fecha")
    ELO_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = {}
    needed = []
    for d in sorted(dates):
        date_str = d.strftime("%Y-%m-%d")
        path = ELO_DIR / f"{date_str}.csv"
        if path.exists() and path.stat().st_size > 50:
            df = pd.read_csv(path)
            snapshots[date_str] = df
        else:
            needed.append((d, date_str, path))

    print(f"  Fechas únicas: {len(dates)}")
    print(f"  Ya descargadas: {len(dates) - len(needed)}")
    print(f"  Por descargar: {len(needed)}")
    if needed:
        print(f"  Tiempo estimado: {len(needed) * SLEEP / 60:.1f} minutos")

    for i, (d, date_str, path) in enumerate(needed):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(needed)}] {date_str}...", flush=True)

        time.sleep(SLEEP)
        try:
            resp = requests.get(f"http://api.clubelo.com/{date_str}", timeout=15)
            resp.raise_for_status()
            text = resp.text.strip()
            lines = text.split("\n")
            if len(lines) > 1:
                with open(path, "w") as f:
                    f.write(text)
                df = pd.read_csv(path)
                snapshots[date_str] = df
            else:
                print(f"    Empty response for {date_str}")
        except Exception as e:
            print(f"    Error for {date_str}: {e}")

    print(f"  Total snapshots cargados: {len(snapshots)}")
    return snapshots


# ---------------------------------------------------------------------------
# Phase 3: Assign ELO to each match
# ---------------------------------------------------------------------------

def assign_elo(features_df: pd.DataFrame, snapshots: dict) -> pd.DataFrame:
    """For each match, look up ELO for both teams on the match date."""
    sep("FASE 3: Asignar ELO a cada partido")

    features_df = features_df.copy()
    features_df["elo_home"] = np.nan
    features_df["elo_away"] = np.nan

    matched = 0
    unmatched_teams = set()

    for idx, row in features_df.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        home_name = row["home_team_name"]
        away_name = row["away_team_name"]

        if date_str not in snapshots:
            continue

        snapshot = snapshots[date_str]
        elo_lookup = dict(zip(snapshot["Club"], snapshot["Elo"]))

        # Look up home team
        elo_name_home = NAME_MAP.get(home_name)
        if elo_name_home and elo_name_home in elo_lookup:
            features_df.at[idx, "elo_home"] = elo_lookup[elo_name_home]
        else:
            unmatched_teams.add(home_name)

        # Look up away team
        elo_name_away = NAME_MAP.get(away_name)
        if elo_name_away and elo_name_away in elo_lookup:
            features_df.at[idx, "elo_away"] = elo_lookup[elo_name_away]
        else:
            unmatched_teams.add(away_name)

        if not pd.isna(features_df.at[idx, "elo_home"]) and not pd.isna(features_df.at[idx, "elo_away"]):
            matched += 1

    # Derived features
    features_df["elo_diff"] = features_df["elo_home"] - features_df["elo_away"]
    features_df["elo_expected_home"] = 1 / (1 + 10 ** ((features_df["elo_away"] - features_df["elo_home"]) / 400))

    both_elo = features_df["elo_home"].notna() & features_df["elo_away"].notna()
    home_only = features_df["elo_home"].notna().sum()
    away_only = features_df["elo_away"].notna().sum()

    print(f"  Partidos con ELO completo (ambos equipos): {matched}/{len(features_df)} ({matched/len(features_df)*100:.1f}%)")
    print(f"  Home ELO encontrado: {home_only}/{len(features_df)}")
    print(f"  Away ELO encontrado: {away_only}/{len(features_df)}")

    if unmatched_teams:
        print(f"\n  Equipos sin ELO ({len(unmatched_teams)}):")
        for t in sorted(unmatched_teams):
            mapped = NAME_MAP.get(t, "NO MAPPING")
            print(f"    {t} -> {mapped}")

    # Quick correlation check
    result_map = {"H": 1, "D": 0, "A": -1}
    features_df["result_numeric"] = features_df["result"].map(result_map)
    for col in ["elo_home", "elo_away", "elo_diff", "elo_expected_home"]:
        corr = features_df[col].corr(features_df["result_numeric"])
        print(f"  Correlación {col} vs resultado: {corr:+.4f}")
    features_df.drop(columns=["result_numeric"], inplace=True)

    # ELO stats
    print(f"\n  ELO stats:")
    print(f"    Home ELO: mean={features_df['elo_home'].mean():.0f}, std={features_df['elo_home'].std():.0f}")
    print(f"    Away ELO: mean={features_df['elo_away'].mean():.0f}, std={features_df['elo_away'].std():.0f}")
    print(f"    ELO diff: mean={features_df['elo_diff'].mean():.0f}, std={features_df['elo_diff'].std():.0f}")

    return features_df


# ---------------------------------------------------------------------------
# Phase 4: Evaluate with walk-forward
# ---------------------------------------------------------------------------

ORIGINAL_FEATURES = [
    "home_rolling_xg_for", "home_rolling_xg_against", "home_rolling_xg_diff",
    "home_rolling_goals_for", "home_rolling_goals_against",
    "home_rolling_xg_overperformance",
    "home_rolling_shots_on_goal", "home_rolling_shots_accuracy",
    "home_rolling_possession", "home_rolling_pass_accuracy",
    "home_rolling_corners", "home_rolling_avg_rating",
    "home_rolling_key_passes", "home_rolling_duels_won_pct",
    "home_rolling_dribbles_success", "home_rolling_points",
    "away_rolling_xg_for", "away_rolling_xg_against", "away_rolling_xg_diff",
    "away_rolling_goals_for", "away_rolling_goals_against",
    "away_rolling_xg_overperformance",
    "away_rolling_shots_on_goal", "away_rolling_shots_accuracy",
    "away_rolling_possession", "away_rolling_pass_accuracy",
    "away_rolling_corners", "away_rolling_avg_rating",
    "away_rolling_key_passes", "away_rolling_duels_won_pct",
    "away_rolling_dribbles_success", "away_rolling_points",
    "diff_rolling_xg", "diff_rolling_goals", "diff_rolling_form",
    "is_knockout", "home_days_since_last", "away_days_since_last",
]

ELO_FEATURES = ["elo_home", "elo_away", "elo_diff", "elo_expected_home"]

# Top 5 domestic features from 06_retrain (by permutation importance)
TOP_DOMESTIC = [
    "away_domestic_away_goals_avg",
    "home_domestic_home_goals_avg",
    "away_domestic_xg_against_last5",
    "home_domestic_goals_against_last5",
    "home_domestic_league_position",
]

CLASSES = ["A", "D", "H"]


def build_splits(df):
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


def make_models():
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, min_samples_leaf=10,
                learning_rate=0.05, random_state=42,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42, C=1.0,
            )),
        ]),
    }


def run_walk_forward(df, features, le):
    splits = build_splits(df)
    model_results = {}
    for model_name in ["Random Forest", "HistGradientBoosting", "Logistic Regression"]:
        split_metrics = []
        for split_name, train, val in splits:
            pipeline = make_models()[model_name]
            pipeline.fit(train[features].values, le.transform(train["result"]))
            y_pred = le.inverse_transform(pipeline.predict(val[features].values))
            y_proba = pipeline.predict_proba(val[features].values)
            y_true = val["result"].values
            split_metrics.append({
                "accuracy": accuracy_score(y_true, y_pred),
                "log_loss": log_loss(y_true, y_proba, labels=le.classes_),
                "f1_macro": f1_score(y_true, y_pred, average="macro", labels=le.classes_),
            })
        model_results[model_name] = {
            "avg_accuracy": np.mean([m["accuracy"] for m in split_metrics]),
            "avg_log_loss": np.mean([m["log_loss"] for m in split_metrics]),
            "avg_f1_macro": np.mean([m["f1_macro"] for m in split_metrics]),
            "per_split": split_metrics,
        }
    return model_results


def print_comparison(variants: dict):
    """Print side-by-side comparison table."""
    variant_names = list(variants.keys())
    model_names = list(variants[variant_names[0]].keys())

    # Header
    header = f"  {'Model':<25s}"
    for vn in variant_names:
        header += f" {vn:>14s}"
    if len(variant_names) > 1:
        header += f" {'Delta vs A':>12s}"
    print(header)
    print(f"  {'-'*25}" + f" {'-'*14}" * len(variant_names) + (f" {'-'*12}" if len(variant_names) > 1 else ""))

    for model_name in model_names:
        for metric, key in [("Accuracy", "avg_accuracy"), ("Log Loss", "avg_log_loss"), ("F1 Macro", "avg_f1_macro")]:
            row = f"  {model_name + ' ' + metric:<25s}"
            vals = []
            for vn in variant_names:
                v = variants[vn][model_name][key]
                vals.append(v)
                row += f" {v:>14.4f}"
            if len(variant_names) > 1:
                delta = vals[-1] - vals[0]
                better = (delta > 0 and key != "avg_log_loss") or (delta < 0 and key == "avg_log_loss")
                marker = " *" if better else ""
                row += f" {delta:>+11.4f}{marker}"
            print(row)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep("ELO RATINGS PIPELINE")

    # Load data
    features_df = pd.read_csv(PROCESSED_DIR / "features_v2.csv")
    features_df["date"] = pd.to_datetime(features_df["date"])
    le = LabelEncoder()
    le.fit(CLASSES)

    print(f"  Dataset: {features_df.shape}")

    # Phase 1-2: Download ELO snapshots
    unique_dates = features_df["date"].dt.date.unique()
    snapshots = download_elo_snapshots(list(unique_dates))

    # Phase 3: Assign ELO
    features_df = assign_elo(features_df, snapshots)

    # Save features_v3
    v3_path = PROCESSED_DIR / "features_v3.csv"
    features_df.to_csv(v3_path, index=False)
    print(f"\n  Guardado: {v3_path} ({features_df.shape})")

    # Phase 4: Walk-forward comparison
    sep("FASE 4: Walk-Forward Validation")

    feat_a = ORIGINAL_FEATURES
    feat_b = ORIGINAL_FEATURES + ELO_FEATURES
    feat_c = ORIGINAL_FEATURES + ELO_FEATURES + TOP_DOMESTIC

    print(f"\n  Variante A: {len(feat_a)} features (originales)")
    print(f"  Variante B: {len(feat_b)} features (originales + ELO)")
    print(f"  Variante C: {len(feat_c)} features (originales + ELO + top 5 domestic)")

    print("\n  Training...")
    results_a = run_walk_forward(features_df, feat_a, le)
    results_b = run_walk_forward(features_df, feat_b, le)
    results_c = run_walk_forward(features_df, feat_c, le)

    sep("RESULTADOS: Comparación de Variantes")
    print_comparison({"A (original)": results_a, "B (+ELO)": results_b, "C (+ELO+dom)": results_c})

    # Per-split detail for Random Forest
    sep("DETALLE POR SPLIT (Random Forest)")
    splits_names = ["Split 1", "Split 2", "Split 3"]
    print(f"\n  {'Split':<12s} {'A Acc':>8s} {'B Acc':>8s} {'C Acc':>8s} {'A LL':>8s} {'B LL':>8s} {'C LL':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, sname in enumerate(splits_names):
        sa = results_a["Random Forest"]["per_split"][i]
        sb = results_b["Random Forest"]["per_split"][i]
        sc = results_c["Random Forest"]["per_split"][i]
        print(f"  {sname:<12s} {sa['accuracy']:>8.4f} {sb['accuracy']:>8.4f} {sc['accuracy']:>8.4f} {sa['log_loss']:>8.4f} {sb['log_loss']:>8.4f} {sc['log_loss']:>8.4f}")

    # Feature importance
    sep("FEATURE IMPORTANCE (mejor variante, full dataset)")

    # Determine best variant
    best_variants = {}
    for name, results, feats in [("A", results_a, feat_a), ("B", results_b, feat_b), ("C", results_c, feat_c)]:
        best_model = max(results, key=lambda m: results[m]["avg_accuracy"])
        best_variants[name] = {
            "model": best_model,
            "accuracy": results[best_model]["avg_accuracy"],
            "log_loss": results[best_model]["avg_log_loss"],
            "f1_macro": results[best_model]["avg_f1_macro"],
            "features": feats,
        }

    # Pick overall best by accuracy
    overall_best = max(best_variants, key=lambda v: best_variants[v]["accuracy"])
    best_info = best_variants[overall_best]
    best_feats = best_info["features"]

    print(f"\n  Mejor variante: {overall_best} ({best_info['model']}, Acc={best_info['accuracy']:.4f})")

    # Train for feature importance
    pipeline = make_models()[best_info["model"]]
    X_all = features_df[best_feats].values
    y_all = le.transform(features_df["result"])
    pipeline.fit(X_all, y_all)

    result = permutation_importance(pipeline, X_all, y_all, n_repeats=10, random_state=42, scoring="accuracy")
    feat_imp = sorted(zip(best_feats, result.importances_mean), key=lambda x: x[1], reverse=True)

    max_imp = max(imp for _, imp in feat_imp[:15]) if feat_imp else 1
    print(f"\n  Top 15 features:")
    for i, (fname, imp) in enumerate(feat_imp[:15], 1):
        marker = ""
        if "elo" in fname:
            marker = " ** ELO"
        elif "domestic" in fname:
            marker = " ** DOMESTIC"
        bar = "█" * int(imp / max_imp * 25) if max_imp > 0 else ""
        print(f"    {i:2d}. {fname:<45s} {imp:.4f}  {bar}{marker}")

    # ELO features rank
    elo_in_list = [(f, imp) for f, imp in feat_imp if "elo" in f]
    print(f"\n  ELO features:")
    for f, imp in elo_in_list:
        rank = [fn for fn, _ in feat_imp].index(f) + 1
        print(f"    {f:<45s} {imp:.4f} (rank #{rank})")

    # Decision
    sep("DECISION")

    baseline_acc = best_variants["A"]["accuracy"]
    baseline_ll = best_variants["A"]["log_loss"]
    best_acc = best_variants[overall_best]["accuracy"]
    best_ll = best_variants[overall_best]["log_loss"]
    acc_delta = best_acc - baseline_acc
    ll_delta = best_ll - baseline_ll

    print(f"\n  Variante A (baseline): {best_variants['A']['model']} Acc={baseline_acc:.4f} LL={baseline_ll:.4f}")
    print(f"  Variante B (+ELO):     {best_variants['B']['model']} Acc={best_variants['B']['accuracy']:.4f} LL={best_variants['B']['log_loss']:.4f}")
    print(f"  Variante C (+ELO+dom): {best_variants['C']['model']} Acc={best_variants['C']['accuracy']:.4f} LL={best_variants['C']['log_loss']:.4f}")
    print(f"\n  Mejor variante: {overall_best}")
    print(f"  Delta accuracy vs baseline: {acc_delta:+.4f}")
    print(f"  Delta log loss vs baseline: {ll_delta:+.4f} ({'better' if ll_delta < 0 else 'worse'})")

    improved = acc_delta > 0 or (acc_delta >= -0.005 and ll_delta < -0.02)

    if improved:
        print(f"\n  MEJORA: +{acc_delta:.4f} accuracy, {ll_delta:+.4f} log loss")
        print(f"  Guardando modelo v3...")

        # Train final model on all data
        final_pipeline = make_models()[best_info["model"]]
        final_pipeline.fit(features_df[best_feats].values, le.transform(features_df["result"]))
        train_acc = accuracy_score(le.transform(features_df["result"]), final_pipeline.predict(features_df[best_feats].values))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / "champion_model_v3.pkl"
        joblib.dump(final_pipeline, model_path)
        print(f"  Saved: {model_path}")

        feat_path = MODELS_DIR / "feature_list_v3.json"
        with open(feat_path, "w") as f:
            json.dump(best_feats, f, indent=2)
        print(f"  Saved: {feat_path}")

        metrics_v3 = {
            "model": best_info["model"],
            "variant": overall_best,
            "n_features": len(best_feats),
            "accuracy": round(best_acc, 4),
            "log_loss": round(best_ll, 4),
            "f1_macro": round(best_info["f1_macro"], 4),
            "delta_accuracy_vs_v1": round(acc_delta, 4),
            "delta_log_loss_vs_v1": round(ll_delta, 4),
            "training_accuracy": round(train_acc, 4),
            "feature_importance_top15": [
                {"feature": f, "importance": round(float(imp), 6)}
                for f, imp in feat_imp[:15]
            ],
            "per_variant": {
                name: {
                    "model": info["model"],
                    "accuracy": round(info["accuracy"], 4),
                    "log_loss": round(info["log_loss"], 4),
                }
                for name, info in best_variants.items()
            },
        }
        metrics_path = MODELS_DIR / "metrics_v3.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_v3, f, indent=2)
        print(f"  Saved: {metrics_path}")
    else:
        print(f"\n  NO MEJORO: ELO no aporta suficiente mejora en walk-forward validation.")
        print(f"  El modelo original sigue siendo el mejor.")

        metrics_v3 = {
            "improved": False,
            "per_variant": {
                name: {
                    "model": info["model"],
                    "accuracy": round(info["accuracy"], 4),
                    "log_loss": round(info["log_loss"], 4),
                }
                for name, info in best_variants.items()
            },
        }
        metrics_path = MODELS_DIR / "metrics_v3.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_v3, f, indent=2)
        print(f"  Metrics saved for reference: {metrics_path}")

    print()


if __name__ == "__main__":
    main()
