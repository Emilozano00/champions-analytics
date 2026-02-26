#!/usr/bin/env python3
"""
10_europa_league_model.py — Train independent models for Europa League & Conference League.

Uses the unified dataset from features_all.csv (2,471 matches: CL + EL + ECL).
Reuses the same 20-feature set from the Champions League final model.

Phases:
  1. Dataset analysis by competition
  2. Train Europa League model (3 variants × 3 algorithms)
  3. Save best Europa League model
  4. Train Conference League model
  5. Summary table across all competitions
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "features_all.csv"
FEAT_JSON = BASE / "models" / "feature_list_final.json"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def banner(text):
    print(f"\n{'=' * 70}")
    print(f"{text}")
    print(f"{'=' * 70}")


def build_splits(df, eval_filter=None):
    """Walk-forward splits identical to Champions pipeline."""
    df = df.sort_values("date").reset_index(drop=True)
    s2023 = df[df["season"] == 2023]
    s2024 = df[df["season"] == 2024].sort_values("date")
    s2025 = df[df["season"] == 2025]
    mid = len(s2024) // 2

    splits = [
        ("Split 1", s2023, s2024.iloc[:mid]),
        ("Split 2", pd.concat([s2023, s2024.iloc[:mid]]), s2024.iloc[mid:]),
        ("Split 3", pd.concat([s2023, s2024]), s2025),
    ]

    if eval_filter is not None:
        splits = [
            (name, tr, va[eval_filter(va)]) for name, tr, va in splits
        ]
        splits = [(n, tr, va) for n, tr, va in splits if len(va) > 0]

    return splits


def make_models():
    """Return dict of model name → pipeline."""
    return {
        "LogReg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.01, penalty="l2", max_iter=2000,
                class_weight="balanced", random_state=42,
            )),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=10, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "HGB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=500, max_depth=5, learning_rate=0.05,
                min_samples_leaf=10, random_state=42,
            )),
        ]),
    }


def evaluate_variant(train_df, splits, features, le):
    """Train each model type on the given splits, return results."""
    results = {}
    for model_name, pipe in make_models().items():
        accs, lls, f1s = [], [], []
        for split_name, tr, va in splits:
            if len(tr) < 10 or len(va) < 10:
                continue
            X_tr, y_tr = tr[features].values, le.transform(tr["result"])
            X_va, y_va = va[features].values, le.transform(va["result"])

            pipe.fit(X_tr, y_tr)

            # Calibrate
            try:
                cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
                cal.fit(X_tr, y_tr)
                proba = cal.predict_proba(X_va)
                preds = cal.predict(X_va)
            except Exception:
                proba = pipe.predict_proba(X_va)
                preds = pipe.predict(X_va)

            acc = accuracy_score(y_va, preds)
            ll = log_loss(y_va, proba, labels=[0, 1, 2])
            f1 = f1_score(y_va, preds, average="macro")
            accs.append(acc)
            lls.append(ll)
            f1s.append(f1)
            print(f"      {split_name}: Acc={acc:.4f} LL={ll:.4f} "
                  f"(train={len(tr)}, val={len(va)})")

        if accs:
            results[model_name] = {
                "accuracy": round(np.mean(accs), 4),
                "log_loss": round(np.mean(lls), 4),
                "f1_macro": round(np.mean(f1s), 4),
                "n_splits": len(accs),
            }
            print(f"    {model_name} AVG: Acc={results[model_name]['accuracy']:.4f} "
                  f"LL={results[model_name]['log_loss']:.4f} "
                  f"F1={results[model_name]['f1_macro']:.4f}")
        else:
            print(f"    {model_name}: insufficient data for evaluation")
    return results


def train_and_save_best(df, features, le, comp_name, comp_flag, all_results):
    """Find best model across all variants, retrain on full data, save."""
    best_acc = 0
    best_key = None
    for variant, models in all_results.items():
        for model_name, metrics in models.items():
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_key = (variant, model_name, metrics)

    if best_key is None:
        print(f"\n  No valid model found for {comp_name}.")
        return None

    variant, model_name, metrics = best_key
    print(f"\n  Best: {variant} / {model_name} → "
          f"Acc={metrics['accuracy']:.4f} LL={metrics['log_loss']:.4f}")

    # Retrain on all available data
    pipe = make_models()[model_name]
    X_all = df[features].values
    y_all = le.transform(df["result"])
    pipe.fit(X_all, y_all)

    try:
        cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
        cal.fit(X_all, y_all)
        final_model = cal
    except Exception:
        final_model = pipe

    # Save
    prefix = comp_name.lower().replace(" ", "_")
    model_path = MODELS_DIR / f"{prefix}_model.pkl"
    feat_path = MODELS_DIR / f"{prefix}_feature_list.json"
    metrics_path = MODELS_DIR / f"{prefix}_metrics.json"

    joblib.dump(final_model, model_path)
    with open(feat_path, "w") as f:
        json.dump(features, f, indent=2)

    save_metrics = {
        "competition": comp_name,
        "best_variant": variant,
        "best_model": model_name,
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "f1_macro": metrics["f1_macro"],
        "n_matches_train": len(df),
        "n_features": len(features),
        "all_results": {
            var: {m: r for m, r in models.items()}
            for var, models in all_results.items()
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)

    print(f"  Saved: {model_path}")
    print(f"  Saved: {feat_path}")
    print(f"  Saved: {metrics_path}")

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("█" * 70)
    print("10 — EUROPA LEAGUE & CONFERENCE LEAGUE MODELS")
    print("█" * 70)

    # Load data
    df = pd.read_csv(DATA, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n  Dataset total: {len(df)} partidos")

    # Load feature list
    with open(FEAT_JSON) as f:
        CL_FEATURES = json.load(f)
    print(f"  Features base (CL): {len(CL_FEATURES)}")

    # Label encoder
    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    # ── FASE 1: Análisis ─────────────────────────────────────────────────
    banner("FASE 1: Análisis del Dataset por Competición")

    # Matches by competition and season
    print("\n  Partidos por competición y temporada:")
    print(f"  {'Competición':<25} {'2023':>8} {'2024':>8} {'2025':>8} {'Total':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    comp_map = {
        "is_champions": "Champions League",
        "is_europa": "Europa League",
        "is_conference": "Conference League",
    }

    comp_counts = {}
    for flag, name in comp_map.items():
        sub = df[df[flag] == 1]
        comp_counts[name] = len(sub)
        by_season = sub.groupby("season").size()
        s23 = by_season.get(2023, 0)
        s24 = by_season.get(2024, 0)
        s25 = by_season.get(2025, 0)
        print(f"  {name:<25} {s23:>8} {s24:>8} {s25:>8} {len(sub):>8}")

    # Target distribution by competition
    print("\n  Distribución del resultado (H/D/A) por competición:")
    print(f"  {'Competición':<25} {'H%':>8} {'D%':>8} {'A%':>8} {'Home Adv':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for flag, name in comp_map.items():
        sub = df[df[flag] == 1]
        if len(sub) == 0:
            continue
        dist = sub["result"].value_counts(normalize=True)
        h = dist.get("H", 0) * 100
        d = dist.get("D", 0) * 100
        a = dist.get("A", 0) * 100
        adv = h - a
        print(f"  {name:<25} {h:>7.1f}% {d:>7.1f}% {a:>7.1f}% {adv:>+9.1f}%")

    # Feature coverage by competition
    print("\n  Cobertura de features clave por competición:")
    key_features = ["elo_diff", "elo_home", "home_domestic_home_goals_avg",
                    "home_rolling_points", "home_rolling_avg_rating"]
    print(f"  {'Feature':<35} {'CL%':>8} {'EL%':>8} {'ECL%':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")

    for feat in key_features:
        if feat not in df.columns:
            continue
        cl_cov = df[df["is_champions"] == 1][feat].notna().mean() * 100
        el_cov = df[df["is_europa"] == 1][feat].notna().mean() * 100 if comp_counts.get("Europa League", 0) > 0 else 0
        ecl_cov = df[df["is_conference"] == 1][feat].notna().mean() * 100 if comp_counts.get("Conference League", 0) > 0 else 0
        print(f"  {feat:<35} {cl_cov:>7.1f}% {el_cov:>7.1f}% {ecl_cov:>7.1f}%")

    # Stats comparison
    print("\n  Medias de stats clave por competición:")
    stat_cols = ["home_rolling_xg_for", "home_rolling_shots_accuracy",
                 "home_rolling_pass_accuracy", "home_rolling_avg_rating"]
    available_stats = [c for c in stat_cols if c in df.columns]

    if available_stats:
        print(f"  {'Stat':<35} {'CL':>10} {'EL':>10} {'ECL':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for stat in available_stats:
            cl_mean = df[df["is_champions"] == 1][stat].mean()
            el_mean = df[df["is_europa"] == 1][stat].mean() if comp_counts.get("Europa League", 0) > 0 else 0
            ecl_mean = df[df["is_conference"] == 1][stat].mean() if comp_counts.get("Conference League", 0) > 0 else 0
            print(f"  {stat:<35} {cl_mean:>10.3f} {el_mean:>10.3f} {ecl_mean:>10.3f}")

    # Check feature availability for EL
    print("\n  Verificando features del modelo CL para Europa League...")
    el_df = df[df["is_europa"] == 1]
    ecl_df = df[df["is_conference"] == 1]

    features = []
    dropped = []
    for feat in CL_FEATURES:
        if feat not in df.columns:
            dropped.append((feat, "no existe"))
            continue
        el_cov = el_df[feat].notna().mean() if len(el_df) > 0 else 0
        if el_cov < 0.10:
            dropped.append((feat, f"cobertura EL {el_cov:.1%}"))
        else:
            features.append(feat)

    if dropped:
        print(f"  Features eliminadas ({len(dropped)}):")
        for feat, reason in dropped:
            print(f"    - {feat}: {reason}")
    print(f"  Features finales: {len(features)}")

    # ── FASE 2: Modelo Europa League ─────────────────────────────────────
    banner("FASE 2: Modelo Europa League")

    if comp_counts.get("Europa League", 0) < 50:
        print("  Insuficientes partidos de Europa League. Saltando.")
        el_best = None
    else:
        el_results = {}

        # Variant A: Solo Europa League
        print("\n  Variante A: Solo Europa League")
        el_only = df[df["is_europa"] == 1].copy()
        splits_a = build_splits(el_only)
        el_results["A: Solo EL"] = evaluate_variant(el_only, splits_a, features, le)

        # Variant B: Europa + Conference
        print("\n  Variante B: Europa + Conference League")
        el_ecl = df[(df["is_europa"] == 1) | (df["is_conference"] == 1)].copy()
        splits_b = build_splits(el_ecl, eval_filter=lambda v: v["is_europa"] == 1)
        el_results["B: EL+ECL→eval EL"] = evaluate_variant(el_ecl, splits_b, features, le)

        # Variant C: All competitions, eval on EL
        print("\n  Variante C: CL + EL + ECL → eval solo EL (transfer)")
        splits_c = build_splits(df, eval_filter=lambda v: v["is_europa"] == 1)
        el_results["C: ALL→eval EL"] = evaluate_variant(df, splits_c, features, le)

        # Print comparison
        print(f"\n  {'Variante':<30} {'Accuracy':>10} {'Log Loss':>10} {'F1':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for variant, models in el_results.items():
            for model_name, m in models.items():
                label = f"{variant}/{model_name}"
                print(f"  {label:<30} {m['accuracy']:>10.4f} {m['log_loss']:>10.4f} {m['f1_macro']:>10.4f}")

    # ── FASE 3: Guardar mejor modelo EL ──────────────────────────────────
    banner("FASE 3: Guardar Mejor Modelo Europa League")

    if comp_counts.get("Europa League", 0) >= 50:
        el_best = train_and_save_best(
            el_only, features, le, "europa", "is_europa", el_results
        )
    else:
        el_best = None
        print("  Saltado — insuficientes datos.")

    # ── FASE 4: Modelo Conference League ─────────────────────────────────
    banner("FASE 4: Modelo Conference League")

    if comp_counts.get("Conference League", 0) < 50:
        print("  Insuficientes partidos de Conference League. Saltando.")
        ecl_best = None
    else:
        ecl_results = {}

        # Variant A: Solo Conference League
        print("\n  Variante A: Solo Conference League")
        ecl_only = df[df["is_conference"] == 1].copy()
        splits_a = build_splits(ecl_only)
        ecl_results["A: Solo ECL"] = evaluate_variant(ecl_only, splits_a, features, le)

        # Variant B: Conference + Europa
        print("\n  Variante B: Conference + Europa League")
        ecl_el = df[(df["is_conference"] == 1) | (df["is_europa"] == 1)].copy()
        splits_b = build_splits(ecl_el, eval_filter=lambda v: v["is_conference"] == 1)
        ecl_results["B: ECL+EL→eval ECL"] = evaluate_variant(ecl_el, splits_b, features, le)

        # Variant C: All competitions, eval on ECL
        print("\n  Variante C: CL + EL + ECL → eval solo ECL (transfer)")
        splits_c = build_splits(df, eval_filter=lambda v: v["is_conference"] == 1)
        ecl_results["C: ALL→eval ECL"] = evaluate_variant(df, splits_c, features, le)

        # Print comparison
        print(f"\n  {'Variante':<30} {'Accuracy':>10} {'Log Loss':>10} {'F1':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for variant, models in ecl_results.items():
            for model_name, m in models.items():
                label = f"{variant}/{model_name}"
                print(f"  {label:<30} {m['accuracy']:>10.4f} {m['log_loss']:>10.4f} {m['f1_macro']:>10.4f}")

        # Save best
        banner("Guardar Mejor Modelo Conference League")
        ecl_best = train_and_save_best(
            ecl_only, features, le, "conference", "is_conference", ecl_results
        )

    # ── FASE 5: Tabla resumen ────────────────────────────────────────────
    banner("FASE 5: Tabla Resumen — Todos los Modelos")

    print(f"\n  {'Competición':<20} {'Partidos':>10} {'Accuracy':>10} {'Log Loss':>10} {'Mejor Modelo':<25}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*25}")

    # Champions (reference)
    print(f"  {'Champions League':<20} {709:>10} {0.5721:>10.4f} {0.9331:>10.4f} {'LogReg calibrado':<25}")

    # Europa
    if el_best:
        # Find best variant/model name from el_results
        best_label = "—"
        best_acc = 0
        for var, models in el_results.items():
            for mname, m in models.items():
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    best_label = f"{mname} ({var.split(':')[0]})"
        print(f"  {'Europa League':<20} {comp_counts.get('Europa League', 0):>10} "
              f"{el_best['accuracy']:>10.4f} {el_best['log_loss']:>10.4f} {best_label:<25}")
    else:
        print(f"  {'Europa League':<20} {comp_counts.get('Europa League', 0):>10} "
              f"{'—':>10} {'—':>10} {'sin modelo':<25}")

    # Conference
    if ecl_best:
        best_label = "—"
        best_acc = 0
        for var, models in ecl_results.items():
            for mname, m in models.items():
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    best_label = f"{mname} ({var.split(':')[0]})"
        print(f"  {'Conference League':<20} {comp_counts.get('Conference League', 0):>10} "
              f"{ecl_best['accuracy']:>10.4f} {ecl_best['log_loss']:>10.4f} {best_label:<25}")
    else:
        print(f"  {'Conference League':<20} {comp_counts.get('Conference League', 0):>10} "
              f"{'—':>10} {'—':>10} {'sin modelo':<25}")

    # Recommendation for Streamlit
    print("\n" + "=" * 70)
    print("RECOMENDACIÓN PARA STREAMLIT")
    print("=" * 70)

    viable = []
    if el_best and el_best["accuracy"] > 0.45:
        viable.append("Europa League")
    if ecl_best and ecl_best["accuracy"] > 0.45:
        viable.append("Conference League")

    if viable:
        print(f"\n  Modelos viables para agregar a la app: {', '.join(viable)}")
        print("  Sugerencia: agregar selector de competición en Streamlit")
        print("  que cargue el modelo correspondiente (CL/EL/ECL).")
    else:
        print("\n  Ningún modelo adicional supera el umbral mínimo (45%).")
        print("  Mantener solo el modelo de Champions League en la app.")

    if el_best or ecl_best:
        print("\n  Nota: accuracy >50% en 3-way classification (H/D/A)")
        print("  es mejor que random (33%). El baseline informado")
        print("  (siempre predecir Home) suele dar ~45-50%.")

    print(f"\n{'█' * 70}")
    print("PIPELINE COMPLETADO")
    print(f"{'█' * 70}")


if __name__ == "__main__":
    main()
