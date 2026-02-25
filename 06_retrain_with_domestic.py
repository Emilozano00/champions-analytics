"""
06_retrain_with_domestic.py

Compare model performance WITH vs WITHOUT domestic league features.
Uses identical walk-forward validation to 04_train_model.py for fair comparison.
"""

import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Feature sets
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

DOMESTIC_FEATURES = [
    "home_domestic_points_last5", "home_domestic_goals_for_last5",
    "home_domestic_goals_against_last5", "home_domestic_xg_for_last5",
    "home_domestic_xg_against_last5", "home_domestic_league_position",
    "home_domestic_ppg", "home_domestic_win_rate",
    "home_domestic_home_goals_avg", "home_domestic_away_goals_avg",
    "away_domestic_points_last5", "away_domestic_goals_for_last5",
    "away_domestic_goals_against_last5", "away_domestic_xg_for_last5",
    "away_domestic_xg_against_last5", "away_domestic_league_position",
    "away_domestic_ppg", "away_domestic_win_rate",
    "away_domestic_home_goals_avg", "away_domestic_away_goals_avg",
    "diff_domestic_position",
]

ALL_FEATURES_V2 = ORIGINAL_FEATURES + DOMESTIC_FEATURES

TARGET = "result"
CLASSES = ["A", "D", "H"]


# ---------------------------------------------------------------------------
# Walk-forward splits (identical to 04_train_model.py)
# ---------------------------------------------------------------------------

def build_splits(df):
    df = df.sort_values("date").reset_index(drop=True)
    s2023 = df[df["season"] == 2023]
    s2024 = df[df["season"] == 2024].sort_values("date")
    s2025 = df[df["season"] == 2025]
    mid = len(s2024) // 2
    s2024_first = s2024.iloc[:mid]
    s2024_second = s2024.iloc[mid:]
    return [
        ("Split 1: 2023 → early 2024", s2023, s2024_first),
        ("Split 2: 2023+early2024 → late 2024", pd.concat([s2023, s2024_first]), s2024_second),
        ("Split 3: 2023+2024 → 2025", pd.concat([s2023, s2024]), s2025),
    ]


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, y_proba, le):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=le.classes_),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=le.classes_),
    }


def run_walk_forward(df, features, le):
    """Run walk-forward for all models. Returns {model: {metric: avg_value}}."""
    splits = build_splits(df)
    model_results = {}

    for model_name in ["Random Forest", "HistGradientBoosting", "Logistic Regression"]:
        split_metrics = []
        for split_name, train, val in splits:
            pipeline = make_models()[model_name]
            X_train = train[features].values
            y_train = le.transform(train[TARGET])
            X_val = val[features].values
            y_val = val[TARGET].values

            pipeline.fit(X_train, y_train)
            y_pred = le.inverse_transform(pipeline.predict(X_val))
            y_proba = pipeline.predict_proba(X_val)
            split_metrics.append(evaluate(y_val, y_pred, y_proba, le))

        model_results[model_name] = {
            "avg_accuracy": np.mean([m["accuracy"] for m in split_metrics]),
            "avg_log_loss": np.mean([m["log_loss"] for m in split_metrics]),
            "avg_f1_macro": np.mean([m["f1_macro"] for m in split_metrics]),
            "per_split": split_metrics,
        }

    return model_results


def run_walk_forward_by_phase(df, features, le):
    """Run walk-forward split 3 only, return accuracy by round type."""
    splits = build_splits(df)
    _, train, val = splits[2]  # 2023+2024 → 2025

    pipeline = make_models()["Random Forest"]
    pipeline.fit(train[features].values, le.transform(train[TARGET]))

    val = val.copy()
    val["y_pred"] = le.inverse_transform(pipeline.predict(val[features].values))

    results = {}
    patterns = {
        "Qualifying": r"(Qualifying|Preliminary)",
        "Group/League Stage": r"(Group|League Stage)",
        "Knockouts": r"(Round of|Quarter|Semi|Final|Play-offs|Knockout)",
    }
    for phase, pat in patterns.items():
        mask = val["round"].str.contains(pat, case=False, na=False)
        sub = val[mask]
        if len(sub) >= 5:
            results[phase] = {
                "n": len(sub),
                "accuracy": accuracy_score(sub[TARGET], sub["y_pred"]),
                "f1_macro": f1_score(sub[TARGET], sub["y_pred"], average="macro", labels=CLASSES),
            }
    return results


def run_walk_forward_draw_analysis(df, features, le):
    """Check if domestic features help predict draws specifically."""
    splits = build_splits(df)
    _, train, val = splits[2]

    pipeline = make_models()["Random Forest"]
    pipeline.fit(train[features].values, le.transform(train[TARGET]))

    y_pred = le.inverse_transform(pipeline.predict(val[features].values))
    y_true = val[TARGET].values

    # Draw-specific metrics
    draw_mask_true = y_true == "D"
    draw_mask_pred = y_pred == "D"

    n_actual_draws = draw_mask_true.sum()
    n_predicted_draws = draw_mask_pred.sum()
    correct_draws = ((y_true == "D") & (y_pred == "D")).sum()
    draw_recall = correct_draws / n_actual_draws if n_actual_draws > 0 else 0
    draw_precision = correct_draws / n_predicted_draws if n_predicted_draws > 0 else 0

    return {
        "actual_draws": int(n_actual_draws),
        "predicted_draws": int(n_predicted_draws),
        "correct_draws": int(correct_draws),
        "draw_recall": draw_recall,
        "draw_precision": draw_precision,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = lambda t: print(f"\n{'='*70}\n{t}\n{'='*70}")

    sep("RETRAIN WITH DOMESTIC FEATURES")

    df = pd.read_csv(PROCESSED_DIR / "features_v2.csv")
    df["date"] = pd.to_datetime(df["date"])
    le = LabelEncoder()
    le.fit(CLASSES)

    print(f"\nDataset: {df.shape[0]} matches, {df.shape[1]} columns")
    print(f"Original features: {len(ORIGINAL_FEATURES)}")
    print(f"Domestic features: {len(DOMESTIC_FEATURES)}")
    print(f"Combined features: {len(ALL_FEATURES_V2)}")

    # ── Run both variants ────────────────────────────────────────────────
    sep("WALK-FORWARD VALIDATION: Variante A (original features)")
    results_a = run_walk_forward(df, ORIGINAL_FEATURES, le)
    for model_name, res in results_a.items():
        print(f"  {model_name:<25s}  Acc={res['avg_accuracy']:.4f}  LL={res['avg_log_loss']:.4f}  F1={res['avg_f1_macro']:.4f}")

    sep("WALK-FORWARD VALIDATION: Variante B (original + domestic)")
    results_b = run_walk_forward(df, ALL_FEATURES_V2, le)
    for model_name, res in results_b.items():
        print(f"  {model_name:<25s}  Acc={res['avg_accuracy']:.4f}  LL={res['avg_log_loss']:.4f}  F1={res['avg_f1_macro']:.4f}")

    # ── Comparison table ─────────────────────────────────────────────────
    sep("COMPARISON: Sin Domésticas vs Con Domésticas")

    print(f"\n  {'Model':<25s} {'Metric':<12s} {'Original':>10s} {'+ Domestic':>10s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    best_model_a = None
    best_model_b = None
    best_acc_a = -1
    best_acc_b = -1

    for model_name in ["Random Forest", "HistGradientBoosting", "Logistic Regression"]:
        a = results_a[model_name]
        b = results_b[model_name]

        if a["avg_accuracy"] > best_acc_a:
            best_acc_a = a["avg_accuracy"]
            best_model_a = model_name
        if b["avg_accuracy"] > best_acc_b:
            best_acc_b = b["avg_accuracy"]
            best_model_b = model_name

        for metric, key in [("Accuracy", "avg_accuracy"), ("Log Loss", "avg_log_loss"), ("F1 Macro", "avg_f1_macro")]:
            va = a[key]
            vb = b[key]
            delta = vb - va
            direction = "+" if delta > 0 else ""
            # For log loss, lower is better
            if key == "avg_log_loss":
                symbol = " (better)" if delta < 0 else " (worse)" if delta > 0 else ""
            else:
                symbol = " (better)" if delta > 0 else " (worse)" if delta < 0 else ""
            print(f"  {model_name:<25s} {metric:<12s} {va:>10.4f} {vb:>10.4f} {direction}{delta:>9.4f}{symbol}")
        print()

    # ── Per-split detail for best model ──────────────────────────────────
    sep("PER-SPLIT DETAIL (Random Forest)")

    splits_names = ["Split 1: 2023→early2024", "Split 2: +early2024→late2024", "Split 3: 2023+2024→2025"]
    print(f"\n  {'Split':<35s} {'Orig Acc':>10s} {'+Dom Acc':>10s} {'Delta':>8s} {'Orig LL':>10s} {'+Dom LL':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    for i, sname in enumerate(splits_names):
        sa = results_a["Random Forest"]["per_split"][i]
        sb = results_b["Random Forest"]["per_split"][i]
        d = sb["accuracy"] - sa["accuracy"]
        print(f"  {sname:<35s} {sa['accuracy']:>10.4f} {sb['accuracy']:>10.4f} {d:>+8.4f} {sa['log_loss']:>10.4f} {sb['log_loss']:>10.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    sep("FEATURE IMPORTANCE (Random Forest + Domestic, full dataset)")

    # Train RF on full data with all features for importance
    pipeline = make_models()["Random Forest"]
    X_all = df[ALL_FEATURES_V2].values
    y_all = le.transform(df[TARGET])
    pipeline.fit(X_all, y_all)

    # Use permutation importance
    result = permutation_importance(
        pipeline, X_all, y_all, n_repeats=10, random_state=42, scoring="accuracy",
    )
    importances = result.importances_mean
    feat_imp = sorted(zip(ALL_FEATURES_V2, importances), key=lambda x: x[1], reverse=True)

    max_imp = max(imp for _, imp in feat_imp[:15]) if feat_imp else 1
    print(f"\n  Top 15 features:")
    for i, (fname, imp) in enumerate(feat_imp[:15], 1):
        marker = " ** DOMESTIC" if "domestic" in fname else ""
        bar = "█" * int(imp / max_imp * 25) if max_imp > 0 else ""
        print(f"    {i:2d}. {fname:<45s} {imp:.4f}  {bar}{marker}")

    # Count domestic in top 15
    domestic_in_top15 = sum(1 for f, _ in feat_imp[:15] if "domestic" in f)
    print(f"\n  Domestic features in top 15: {domestic_in_top15}/15")

    # Top domestic features specifically
    domestic_imp = [(f, imp) for f, imp in feat_imp if "domestic" in f]
    print(f"\n  All domestic features ranked:")
    for i, (fname, imp) in enumerate(domestic_imp, 1):
        overall_rank = [f for f, _ in feat_imp].index(fname) + 1
        print(f"    {i:2d}. {fname:<45s} {imp:.4f}  (overall rank: #{overall_rank})")

    # ── Phase analysis ───────────────────────────────────────────────────
    sep("ANALYSIS BY PHASE (Split 3: 2025 validation)")

    phase_a = run_walk_forward_by_phase(df, ORIGINAL_FEATURES, le)
    phase_b = run_walk_forward_by_phase(df, ALL_FEATURES_V2, le)

    print(f"\n  {'Phase':<25s} {'N':>5s} {'Orig Acc':>10s} {'+Dom Acc':>10s} {'Delta':>8s}")
    print(f"  {'-'*25} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
    for phase in ["Qualifying", "Group/League Stage", "Knockouts"]:
        if phase in phase_a and phase in phase_b:
            a = phase_a[phase]
            b = phase_b[phase]
            d = b["accuracy"] - a["accuracy"]
            print(f"  {phase:<25s} {a['n']:>5d} {a['accuracy']:>10.4f} {b['accuracy']:>10.4f} {d:>+8.4f}")

    # ── Draw analysis ────────────────────────────────────────────────────
    sep("DRAW PREDICTION ANALYSIS (Split 3)")

    draw_a = run_walk_forward_draw_analysis(df, ORIGINAL_FEATURES, le)
    draw_b = run_walk_forward_draw_analysis(df, ALL_FEATURES_V2, le)

    print(f"\n  {'Metric':<25s} {'Original':>12s} {'+ Domestic':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Actual draws':<25s} {draw_a['actual_draws']:>12d} {draw_b['actual_draws']:>12d}")
    print(f"  {'Predicted draws':<25s} {draw_a['predicted_draws']:>12d} {draw_b['predicted_draws']:>12d}")
    print(f"  {'Correct draws':<25s} {draw_a['correct_draws']:>12d} {draw_b['correct_draws']:>12d}")
    print(f"  {'Draw recall':<25s} {draw_a['draw_recall']:>12.4f} {draw_b['draw_recall']:>12.4f}")
    print(f"  {'Draw precision':<25s} {draw_a['draw_precision']:>12.4f} {draw_b['draw_precision']:>12.4f}")

    # ── Decision: save or not ────────────────────────────────────────────
    sep("DECISION")

    best_a_acc = results_a[best_model_a]["avg_accuracy"]
    best_a_ll = results_a[best_model_a]["avg_log_loss"]
    best_b_acc = results_b[best_model_b]["avg_accuracy"]
    best_b_ll = results_b[best_model_b]["avg_log_loss"]

    acc_delta = best_b_acc - best_a_acc
    ll_delta = best_b_ll - best_a_ll

    print(f"\n  Best original model: {best_model_a} (Acc={best_a_acc:.4f}, LL={best_a_ll:.4f})")
    print(f"  Best v2 model:       {best_model_b} (Acc={best_b_acc:.4f}, LL={best_b_ll:.4f})")
    print(f"  Accuracy delta:      {acc_delta:+.4f}")
    print(f"  Log Loss delta:      {ll_delta:+.4f} ({'better' if ll_delta < 0 else 'worse'})")

    # Save if accuracy improved OR if log loss improved meaningfully
    improved = acc_delta > 0 or (acc_delta >= -0.005 and ll_delta < -0.01)

    if improved:
        print(f"\n  MEJORA: +{acc_delta:.4f} accuracy, {ll_delta:+.4f} log loss")
        print(f"  Guardando modelo v2...")

        # Determine which features to use for the saved model
        save_features = ALL_FEATURES_V2
        save_model_name = best_model_b

        # Train final model on ALL data
        final_pipeline = make_models()[save_model_name]
        X_full = df[save_features].values
        y_full = le.transform(df[TARGET])
        final_pipeline.fit(X_full, y_full)

        train_acc = accuracy_score(y_full, final_pipeline.predict(X_full))

        # Save artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / "champion_model_v2.pkl"
        joblib.dump(final_pipeline, model_path)
        print(f"  Saved: {model_path}")

        feat_path = MODELS_DIR / "feature_list_v2.json"
        with open(feat_path, "w") as f:
            json.dump(save_features, f, indent=2)
        print(f"  Saved: {feat_path}")

        metrics_v2 = {
            "model": save_model_name,
            "features_used": save_features,
            "n_features": len(save_features),
            "n_matches": len(df),
            "walk_forward_averages": {
                "original": {
                    "model": best_model_a,
                    "accuracy": round(best_a_acc, 4),
                    "log_loss": round(best_a_ll, 4),
                },
                "with_domestic": {
                    "model": best_model_b,
                    "accuracy": round(best_b_acc, 4),
                    "log_loss": round(best_b_ll, 4),
                },
                "delta_accuracy": round(acc_delta, 4),
                "delta_log_loss": round(ll_delta, 4),
            },
            "per_model_comparison": {},
            "training_accuracy": round(train_acc, 4),
            "feature_importance_top15": [
                {"feature": f, "importance": round(float(imp), 6)}
                for f, imp in feat_imp[:15]
            ],
        }
        for mn in ["Random Forest", "HistGradientBoosting", "Logistic Regression"]:
            metrics_v2["per_model_comparison"][mn] = {
                "original": {
                    "accuracy": round(results_a[mn]["avg_accuracy"], 4),
                    "log_loss": round(results_a[mn]["avg_log_loss"], 4),
                    "f1_macro": round(results_a[mn]["avg_f1_macro"], 4),
                },
                "with_domestic": {
                    "accuracy": round(results_b[mn]["avg_accuracy"], 4),
                    "log_loss": round(results_b[mn]["avg_log_loss"], 4),
                    "f1_macro": round(results_b[mn]["avg_f1_macro"], 4),
                },
            }

        metrics_path = MODELS_DIR / "metrics_v2.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_v2, f, indent=2)
        print(f"  Saved: {metrics_path}")

    else:
        print(f"\n  NO MEJORO: las features domesticas no aportan suficiente senal")
        print(f"  El modelo original sigue siendo el mejor.")
        print(f"  No se guarda modelo v2.")

        # Still save metrics for reference
        metrics_v2 = {
            "improved": False,
            "per_model_comparison": {},
        }
        for mn in ["Random Forest", "HistGradientBoosting", "Logistic Regression"]:
            metrics_v2["per_model_comparison"][mn] = {
                "original": {
                    "accuracy": round(results_a[mn]["avg_accuracy"], 4),
                    "log_loss": round(results_a[mn]["avg_log_loss"], 4),
                    "f1_macro": round(results_a[mn]["avg_f1_macro"], 4),
                },
                "with_domestic": {
                    "accuracy": round(results_b[mn]["avg_accuracy"], 4),
                    "log_loss": round(results_b[mn]["avg_log_loss"], 4),
                    "f1_macro": round(results_b[mn]["avg_f1_macro"], 4),
                },
            }
        metrics_path = MODELS_DIR / "metrics_v2.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_v2, f, indent=2)
        print(f"  Metrics saved for reference: {metrics_path}")

    print()


if __name__ == "__main__":
    main()
