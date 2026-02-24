"""
04_train_model.py

Train and evaluate Champions League match outcome prediction models
using walk-forward validation on features.csv.

Models: Baseline (always H), Random Forest, HistGradientBoosting, Logistic Regression
Validation: 3 walk-forward splits simulating production deployment
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

ROLLING_FEATURES = [
    "home_rolling_xg_for", "home_rolling_xg_against", "home_rolling_xg_diff",
    "home_rolling_goals_for", "home_rolling_goals_against",
    "home_rolling_xg_overperformance",
    "home_rolling_shots_on_goal", "home_rolling_shots_accuracy",
    "home_rolling_possession", "home_rolling_pass_accuracy",
    "home_rolling_corners",
    "home_rolling_avg_rating",
    "home_rolling_key_passes", "home_rolling_duels_won_pct",
    "home_rolling_dribbles_success", "home_rolling_points",
    "away_rolling_xg_for", "away_rolling_xg_against", "away_rolling_xg_diff",
    "away_rolling_goals_for", "away_rolling_goals_against",
    "away_rolling_xg_overperformance",
    "away_rolling_shots_on_goal", "away_rolling_shots_accuracy",
    "away_rolling_possession", "away_rolling_pass_accuracy",
    "away_rolling_corners",
    "away_rolling_avg_rating",
    "away_rolling_key_passes", "away_rolling_duels_won_pct",
    "away_rolling_dribbles_success", "away_rolling_points",
]

DERIVED_FEATURES = [
    "diff_rolling_xg", "diff_rolling_goals", "diff_rolling_form",
]

OTHER_FEATURES = [
    "is_knockout", "home_days_since_last", "away_days_since_last",
]

ALL_FEATURES = ROLLING_FEATURES + DERIVED_FEATURES + OTHER_FEATURES

# Features that include avg_rating (for leakage analysis)
RATING_FEATURES = ["home_rolling_avg_rating", "away_rolling_avg_rating"]
FEATURES_NO_RATING = [f for f in ALL_FEATURES if f not in RATING_FEATURES]

TARGET = "result"
CLASSES = ["A", "D", "H"]  # alphabetical for sklearn


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def build_walk_forward_splits(df: pd.DataFrame) -> list[dict]:
    """Build 3 walk-forward temporal splits."""
    df = df.sort_values("date").reset_index(drop=True)

    s2023 = df[df["season"] == 2023]
    s2024 = df[df["season"] == 2024].sort_values("date")
    s2025 = df[df["season"] == 2025]

    mid_2024 = len(s2024) // 2
    s2024_first = s2024.iloc[:mid_2024]
    s2024_second = s2024.iloc[mid_2024:]

    splits = [
        {
            "name": "Split 1: Train 2023 → Val early 2024",
            "train": s2023,
            "val": s2024_first,
        },
        {
            "name": "Split 2: Train 2023+early2024 → Val late 2024",
            "train": pd.concat([s2023, s2024_first]),
            "val": s2024_second,
        },
        {
            "name": "Split 3: Train 2023+2024 → Val 2025",
            "train": pd.concat([s2023, s2024]),
            "val": s2025,
        },
    ]
    return splits


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """Return dict of model_name -> (pipeline, handles_nan)."""
    return {
        "Random Forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=12, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ]),
            False,
        ),
        "HistGradientBoosting": (
            Pipeline([
                ("clf", HistGradientBoostingClassifier(
                    max_iter=300, max_depth=6, min_samples_leaf=10,
                    learning_rate=0.05, random_state=42,
                )),
            ]),
            True,
        ),
        "Logistic Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced", random_state=42, C=1.0,
                )),
            ]),
            False,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, y_proba, label_encoder) -> dict:
    """Compute all metrics for a single evaluation."""
    classes = label_encoder.classes_
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba, labels=classes)
    f1 = f1_score(y_true, y_pred, average="macro", labels=classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Calibration: predicted prob vs actual frequency per class
    calibration = {}
    for i, cls in enumerate(classes):
        avg_pred_prob = y_proba[:, i].mean()
        actual_freq = (y_true == cls).mean()
        calibration[cls] = {
            "predicted_prob": round(float(avg_pred_prob), 4),
            "actual_freq": round(float(actual_freq), 4),
        }

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "f1_macro": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "calibration": calibration,
    }


def baseline_predictions(y_true, label_encoder) -> dict:
    """Baseline: always predict Home win."""
    classes = label_encoder.classes_
    h_idx = list(classes).index("H")

    y_pred = np.full(len(y_true), "H")
    y_proba = np.zeros((len(y_true), len(classes)))
    y_proba[:, h_idx] = 1.0

    return evaluate_model(y_true, y_pred, y_proba, label_encoder)


def print_metrics(metrics: dict, name: str):
    """Pretty-print metrics for a model."""
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Log Loss:  {metrics['log_loss']:.4f}")
    print(f"    F1 Macro:  {metrics['f1_macro']:.4f}")

    cm = np.array(metrics["confusion_matrix"])
    classes = CLASSES
    print(f"    Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             {'  '.join(f'{c:>5}' for c in classes)}")
    for i, cls in enumerate(classes):
        row_str = "  ".join(f"{v:5d}" for v in cm[i])
        print(f"      {cls:>5}  {row_str}")

    print(f"    Calibration:")
    for cls, cal in metrics["calibration"].items():
        diff = cal["predicted_prob"] - cal["actual_freq"]
        print(f"      {cls}: pred={cal['predicted_prob']:.4f}  actual={cal['actual_freq']:.4f}  diff={diff:+.4f}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_walk_forward(df: pd.DataFrame, features: list, label_encoder: LabelEncoder):
    """Run walk-forward validation for all models. Returns per-split results."""
    splits = build_walk_forward_splits(df)
    all_results = {}

    for split in splits:
        print(f"\n  {split['name']}")
        print(f"    Train: {len(split['train'])} rows  |  Val: {len(split['val'])} rows")

        X_train = split["train"][features].values
        y_train = label_encoder.transform(split["train"][TARGET])
        X_val = split["val"][features].values
        y_val_encoded = label_encoder.transform(split["val"][TARGET])
        y_val_labels = split["val"][TARGET].values

        # Baseline
        split_results = {}
        split_results["Baseline (always H)"] = baseline_predictions(
            y_val_labels, label_encoder
        )
        print(f"\n    --- Baseline (always H) ---")
        print_metrics(split_results["Baseline (always H)"], "Baseline")

        # Trained models
        models = build_models()
        for model_name, (pipeline, handles_nan) in models.items():
            pipeline.fit(X_train, y_train)

            y_pred_encoded = pipeline.predict(X_val)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_proba = pipeline.predict_proba(X_val)

            metrics = evaluate_model(y_val_labels, y_pred, y_proba, label_encoder)
            split_results[model_name] = metrics

            print(f"\n    --- {model_name} ---")
            print_metrics(metrics, model_name)

        all_results[split["name"]] = split_results

    return all_results


def compute_averages(all_results: dict) -> dict:
    """Compute average metrics across splits for each model."""
    model_names = list(next(iter(all_results.values())).keys())
    averages = {}
    for model_name in model_names:
        accs, lls, f1s = [], [], []
        for split_name, split_results in all_results.items():
            m = split_results[model_name]
            accs.append(m["accuracy"])
            lls.append(m["log_loss"])
            f1s.append(m["f1_macro"])
        averages[model_name] = {
            "avg_accuracy": round(np.mean(accs), 4),
            "avg_log_loss": round(np.mean(lls), 4),
            "avg_f1_macro": round(np.mean(f1s), 4),
        }
    return averages


def feature_importance_analysis(df: pd.DataFrame, features: list,
                                label_encoder: LabelEncoder):
    """Train best model on full data and report feature importances via permutation."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (HistGradientBoosting — permutation importance)")
    print("=" * 60)

    X = df[features].values
    y = label_encoder.transform(df[TARGET])

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=10,
        learning_rate=0.05, random_state=42,
    )
    model.fit(X, y)

    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, scoring="accuracy",
    )
    importances = result.importances_mean
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    max_imp = max(imp for _, imp in feat_imp[:15]) if feat_imp else 1
    print("\n  Top 15 features:")
    for i, (fname, imp) in enumerate(feat_imp[:15], 1):
        bar = "█" * int(imp / max_imp * 30) if max_imp > 0 else ""
        print(f"    {i:2d}. {fname:<40s} {imp:.4f}  {bar}")

    return feat_imp


def rating_leakage_analysis(df: pd.DataFrame, label_encoder: LabelEncoder):
    """Compare model accuracy WITH vs WITHOUT avg_rating features."""
    print("\n" + "=" * 60)
    print("LEAKAGE ANALYSIS: avg_rating impact")
    print("=" * 60)

    splits = build_walk_forward_splits(df)

    for feature_set, feat_name in [(ALL_FEATURES, "WITH rating"), (FEATURES_NO_RATING, "WITHOUT rating")]:
        accs, lls = [], []
        for split in splits:
            X_train = split["train"][feature_set].values
            y_train = label_encoder.transform(split["train"][TARGET])
            X_val = split["val"][feature_set].values
            y_val = split["val"][TARGET].values

            model = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, min_samples_leaf=10,
                learning_rate=0.05, random_state=42,
            )
            model.fit(X_train, y_train)

            y_pred = label_encoder.inverse_transform(model.predict(X_val))
            y_proba = model.predict_proba(X_val)
            accs.append(accuracy_score(y_val, y_pred))
            lls.append(log_loss(y_val, y_proba, labels=label_encoder.classes_))

        avg_acc = np.mean(accs)
        avg_ll = np.mean(lls)
        print(f"\n  {feat_name} ({len(feature_set)} features):")
        print(f"    Avg Accuracy: {avg_acc:.4f}")
        print(f"    Avg Log Loss: {avg_ll:.4f}")
        for i, split in enumerate(splits):
            print(f"    Split {i+1} Accuracy: {accs[i]:.4f}  Log Loss: {lls[i]:.4f}")


def phase_analysis(df: pd.DataFrame, label_encoder: LabelEncoder):
    """Compare model performance on group/league stage vs knockout rounds."""
    print("\n" + "=" * 60)
    print("ANALYSIS BY PHASE: Group/League Stage vs Knockouts")
    print("=" * 60)

    # Use the last split (train 2023+2024, val 2025) for this analysis
    splits = build_walk_forward_splits(df)
    split = splits[2]  # 2023+2024 train, 2025 val

    X_train = split["train"][ALL_FEATURES].values
    y_train = label_encoder.transform(split["train"][TARGET])

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=10,
        learning_rate=0.05, random_state=42,
    )
    model.fit(X_train, y_train)

    val = split["val"].copy()
    val["y_pred"] = label_encoder.inverse_transform(model.predict(val[ALL_FEATURES].values))

    # Classify rounds
    knockout_pattern = r"(Round of|Quarter|Semi|Final|Play-offs|Knockout)"
    group_pattern = r"(Group|League Stage)"
    qualifying_pattern = r"(Qualifying|Preliminary)"

    for phase_name, pattern in [("Group/League Stage", group_pattern),
                                 ("Knockouts", knockout_pattern),
                                 ("Qualifying", qualifying_pattern)]:
        mask = val["round"].str.contains(pattern, case=False, na=False)
        subset = val[mask]
        if len(subset) == 0:
            print(f"\n  {phase_name}: no matches in validation set")
            continue

        acc = accuracy_score(subset[TARGET], subset["y_pred"])
        f1 = f1_score(subset[TARGET], subset["y_pred"], average="macro", labels=CLASSES)
        dist = subset[TARGET].value_counts()
        print(f"\n  {phase_name} ({len(subset)} matches):")
        print(f"    Accuracy: {acc:.4f}  F1 Macro: {f1:.4f}")
        print(f"    Actual distribution: {dict(dist)}")
        pred_dist = pd.Series(subset["y_pred"]).value_counts()
        print(f"    Predicted distribution: {dict(pred_dist)}")


def train_final_model(df: pd.DataFrame, features: list,
                      label_encoder: LabelEncoder):
    """Train final model on ALL data and save to disk."""
    print("\n" + "=" * 60)
    print("FINAL MODEL: Training on full dataset (2023+2024+2025)")
    print("=" * 60)

    X = df[features].values
    y = label_encoder.transform(df[TARGET])

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=10,
        learning_rate=0.05, random_state=42,
    )
    model.fit(X, y)

    # Save model
    model_path = MODELS_DIR / "champion_model.pkl"
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    # Save feature list
    feature_list_path = MODELS_DIR / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(features, f, indent=2)
    print(f"  Feature list saved: {feature_list_path}")

    # Save label encoder
    le_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, le_path)
    print(f"  Label encoder saved: {le_path}")

    # Training accuracy (sanity check, not real eval)
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    print(f"  Training accuracy (sanity check): {train_acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CHAMPIONS LEAGUE MATCH PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # Load data
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\nDataset: {df.shape[0]} matches, {df.shape[1]} columns")
    print(f"Seasons: {sorted(df['season'].unique())}")

    # Label encoder
    le = LabelEncoder()
    le.fit(CLASSES)
    print(f"Classes: {list(le.classes_)}")

    # NaN report
    print(f"\n{'='*60}")
    print("NaN REPORT")
    print(f"{'='*60}")
    nan_counts = df[ALL_FEATURES].isna().sum()
    nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)
    rows_with_any_nan = df[ALL_FEATURES].isna().any(axis=1).sum()
    print(f"\n  Rows with at least one NaN: {rows_with_any_nan}/{len(df)} ({rows_with_any_nan/len(df)*100:.1f}%)")
    print(f"  Features with NaN ({len(nan_features)}/{len(ALL_FEATURES)}):")
    for feat, cnt in nan_features.items():
        print(f"    {feat:<45s} {cnt:>4d} ({cnt/len(df)*100:.1f}%)")

    # Walk-forward validation
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*60}")
    all_results = run_walk_forward(df, ALL_FEATURES, le)

    # Averages
    averages = compute_averages(all_results)
    print(f"\n{'='*60}")
    print("AVERAGE METRICS ACROSS ALL SPLITS")
    print(f"{'='*60}")
    print(f"\n  {'Model':<30s} {'Accuracy':>10s} {'Log Loss':>10s} {'F1 Macro':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    for model_name, avg in averages.items():
        print(f"  {model_name:<30s} {avg['avg_accuracy']:>10.4f} {avg['avg_log_loss']:>10.4f} {avg['avg_f1_macro']:>10.4f}")

    # Extra analyses
    feat_imp = feature_importance_analysis(df, ALL_FEATURES, le)
    rating_leakage_analysis(df, le)
    phase_analysis(df, le)

    # Train and save final model
    final_model = train_final_model(df, ALL_FEATURES, le)

    # Save metrics JSON
    metrics_output = {
        "walk_forward_results": {},
        "averages": averages,
        "feature_importance_top15": [
            {"feature": f, "importance": round(float(imp), 6)}
            for f, imp in feat_imp[:15]
        ],
        "features_used": ALL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "n_matches": len(df),
    }
    # Convert walk-forward results (make serializable)
    for split_name, split_results in all_results.items():
        metrics_output["walk_forward_results"][split_name] = split_results

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
