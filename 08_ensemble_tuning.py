#!/usr/bin/env python3
"""
08_ensemble_tuning.py — Feature selection, hyperparameter tuning, stacking ensemble.

Evaluates improvements over model v3:
  - Random Forest, 47 features
  - Accuracy: 0.537, Log loss: 0.967

Phases:
  1. Feature selection (permutation importance, top-N evaluation)
  2. Hyperparameter tuning (GridSearchCV within walk-forward)
  3. Stacking & Voting ensemble
  4. Probability calibration
  5. Comparison table, save if improved
"""

import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
CLASSES = ["A", "D", "H"]

V3_ACCURACY = 0.5369
V3_LOG_LOSS = 0.9665
V3_F1_MACRO = 0.4499
BASELINE_ACC = 0.4984  # Always predict H


# ===========================================================================
# Helpers
# ===========================================================================

def sep(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def build_splits(df):
    """Walk-forward splits identical to all previous scripts."""
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


def eval_metrics(y_true, y_pred, y_proba, le):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=le.classes_),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=le.classes_),
    }


def walk_forward_eval(df, features, le, make_pipe):
    """Run walk-forward with a pipeline factory. Returns metrics + predictions."""
    splits = build_splits(df)
    split_metrics = []
    all_true, all_pred = [], []

    for _, train, val in splits:
        pipe = make_pipe()
        X_tr, y_tr = train[features].values, le.transform(train["result"])
        X_val, y_val = val[features].values, val["result"].values

        pipe.fit(X_tr, y_tr)
        y_pred = le.inverse_transform(pipe.predict(X_val))
        y_proba = pipe.predict_proba(X_val)

        split_metrics.append(eval_metrics(y_val, y_pred, y_proba, le))
        all_true.extend(y_val)
        all_pred.extend(y_pred)

    return {
        "avg_accuracy": np.mean([m["accuracy"] for m in split_metrics]),
        "avg_log_loss": np.mean([m["log_loss"] for m in split_metrics]),
        "avg_f1_macro": np.mean([m["f1_macro"] for m in split_metrics]),
        "per_split": split_metrics,
        "all_true": all_true,
        "all_pred": all_pred,
    }


def classify_phase(round_str):
    r = str(round_str).lower()
    if any(x in r for x in ["qualifying", "preliminary"]):
        return "Qualifying"
    if any(x in r for x in ["round of", "quarter", "semi-final", "final",
                               "play-off", "knockout", "8th"]):
        return "Knockouts"
    return "Group/League"


# ===========================================================================
# Phase 1: Feature Selection
# ===========================================================================

def phase1(df, all_features, le):
    sep("FASE 1: Feature Selection")

    # Rank on Split 3 train (largest set, avoids using 2025 val data)
    splits = build_splits(df)
    train = splits[2][1]
    print(f"  Ranking {len(all_features)} features on Split 3 train ({len(train)} rows)...")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    X_tr, y_tr = train[all_features].values, le.transform(train["result"])
    pipe.fit(X_tr, y_tr)

    result = permutation_importance(
        pipe, X_tr, y_tr, n_repeats=10, random_state=42,
        scoring="accuracy", n_jobs=-1)

    ranked = sorted(zip(all_features, result.importances_mean),
                    key=lambda x: x[1], reverse=True)

    top_imp = ranked[0][1] if ranked[0][1] > 0 else 1
    print(f"\n  Feature ranking:")
    for i, (f, imp) in enumerate(ranked, 1):
        tag = ""
        if "elo" in f:
            tag = " [ELO]"
        elif "domestic" in f:
            tag = " [DOM]"
        bar = "█" * int(imp / top_imp * 20)
        print(f"    {i:2d}. {f:<45s} {imp:>7.4f} {bar}{tag}")

    # Evaluate subsets
    print(f"\n  Evaluating feature subsets in walk-forward:")

    def make_rf():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ])

    subset_results = {}
    for n in [15, 20, 25, 30, 47]:
        top_n = [f for f, _ in ranked[:n]]
        res = walk_forward_eval(df, top_n, le, make_rf)
        subset_results[n] = res
        tag = " ◀ v3 actual" if n == 47 else ""
        print(f"    Top {n:2d}: Acc={res['avg_accuracy']:.4f}  "
              f"LL={res['avg_log_loss']:.4f}  F1={res['avg_f1_macro']:.4f}{tag}")

    # Best subset — prefer smaller if within 0.5% of best
    best_n_by_acc = max(subset_results, key=lambda n: subset_results[n]["avg_accuracy"])
    best_acc = subset_results[best_n_by_acc]["avg_accuracy"]
    best_n = best_n_by_acc
    for n in sorted(subset_results.keys()):
        if subset_results[n]["avg_accuracy"] >= best_acc - 0.005:
            best_n = n
            break

    selected = [f for f, _ in ranked[:best_n]]
    print(f"\n  ➜ Seleccionado: top {best_n} features "
          f"(Acc={subset_results[best_n]['avg_accuracy']:.4f})")

    return selected, ranked, subset_results


# ===========================================================================
# Phase 2: Hyperparameter Tuning
# ===========================================================================

def phase2(df, features, le):
    sep("FASE 2: Hyperparameter Tuning")
    print(f"  Features: {len(features)}")
    print(f"  Método: GridSearchCV(cv=3) dentro de cada walk-forward split\n")

    splits = build_splits(df)

    configs = [
        ("Random Forest",
         lambda: Pipeline([
             ("imputer", SimpleImputer(strategy="median")),
             ("clf", RandomForestClassifier(random_state=42, n_jobs=1)),
         ]),
         {"clf__n_estimators": [200, 500, 1000],
          "clf__max_depth": [5, 10, 15, None],
          "clf__min_samples_leaf": [3, 5, 10],
          "clf__class_weight": [None, "balanced"]}),

        ("HistGradientBoosting",
         lambda: Pipeline([
             ("clf", HistGradientBoostingClassifier(random_state=42)),
         ]),
         {"clf__max_iter": [200, 500, 1000],
          "clf__max_depth": [3, 5, 7],
          "clf__learning_rate": [0.01, 0.05, 0.1],
          "clf__min_samples_leaf": [5, 10, 20]}),

        ("Logistic Regression",
         lambda: Pipeline([
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler()),
             ("clf", LogisticRegression(
                 max_iter=2000, solver="saga", random_state=42)),
         ]),
         {"clf__C": [0.01, 0.1, 1, 10],
          "clf__penalty": ["l1", "l2"],
          "clf__class_weight": [None, "balanced"]}),
    ]

    results = {}
    best_params = {}

    for model_name, make_pipe, grid in configs:
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"  {model_name} ({n_combos} combos × 3-fold × 3 splits):")

        split_metrics, split_params = [], []
        all_true, all_pred = [], []
        t0 = time.time()

        for split_name, train, val in splits:
            X_tr, y_tr = train[features].values, le.transform(train["result"])
            X_val, y_val = val[features].values, val["result"].values

            gs = GridSearchCV(
                make_pipe(), grid, cv=3, scoring="accuracy",
                n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)

            y_pred = le.inverse_transform(gs.predict(X_val))
            y_proba = gs.predict_proba(X_val)
            m = eval_metrics(y_val, y_pred, y_proba, le)

            split_metrics.append(m)
            split_params.append(gs.best_params_)
            all_true.extend(y_val)
            all_pred.extend(y_pred)

            print(f"    {split_name}: Acc={m['accuracy']:.4f} LL={m['log_loss']:.4f} "
                  f"(CV best={gs.best_score_:.4f})")

        elapsed = time.time() - t0
        avg_acc = np.mean([m["accuracy"] for m in split_metrics])
        avg_ll = np.mean([m["log_loss"] for m in split_metrics])
        avg_f1 = np.mean([m["f1_macro"] for m in split_metrics])
        delta = avg_acc - V3_ACCURACY

        results[model_name] = {
            "avg_accuracy": avg_acc, "avg_log_loss": avg_ll,
            "avg_f1_macro": avg_f1, "per_split": split_metrics,
            "all_true": all_true, "all_pred": all_pred,
        }
        best_params[model_name] = split_params[2]  # Split 3 params

        p_str = ", ".join(f"{k.split('__')[1]}={v}" for k, v in split_params[2].items())
        print(f"    AVG: Acc={avg_acc:.4f} LL={avg_ll:.4f} F1={avg_f1:.4f} "
              f"({elapsed:.0f}s) [Δ={delta:+.4f}]")
        print(f"    Split 3 params: {p_str}\n")

    return results, best_params


# ===========================================================================
# Phase 3: Stacking & Voting
# ===========================================================================

def phase3(df, features, le, best_params):
    sep("FASE 3: Stacking & Voting Ensemble")

    def clf_p(params):
        return {k.replace("clf__", ""): v for k, v in params.items()}

    rf_p = clf_p(best_params["Random Forest"])
    hgb_p = clf_p(best_params["HistGradientBoosting"])
    lr_p = clf_p(best_params["Logistic Regression"])

    def make_rf():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**rf_p, random_state=42, n_jobs=1)),
        ])

    def make_hgb():
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(**hgb_p, random_state=42)),
        ])

    def make_lr():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**lr_p, max_iter=2000, solver="saga",
                                       random_state=42)),
        ])

    # --- Stacking ---
    print("  StackingClassifier (meta=LogReg, cv=3)...")
    t0 = time.time()

    def make_stacking():
        return StackingClassifier(
            estimators=[("rf", make_rf()), ("hgb", make_hgb()), ("lr", make_lr())],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=3, stack_method="predict_proba", passthrough=False, n_jobs=-1,
        )

    stacking_res = walk_forward_eval(df, features, le, make_stacking)
    d = stacking_res["avg_accuracy"] - V3_ACCURACY
    print(f"    Acc={stacking_res['avg_accuracy']:.4f} "
          f"LL={stacking_res['avg_log_loss']:.4f} "
          f"F1={stacking_res['avg_f1_macro']:.4f} ({time.time()-t0:.0f}s) [Δ={d:+.4f}]")

    # --- Voting ---
    print("  VotingClassifier (soft)...")
    t0 = time.time()

    def make_voting():
        return VotingClassifier(
            estimators=[("rf", make_rf()), ("hgb", make_hgb()), ("lr", make_lr())],
            voting="soft", n_jobs=-1,
        )

    voting_res = walk_forward_eval(df, features, le, make_voting)
    d = voting_res["avg_accuracy"] - V3_ACCURACY
    print(f"    Acc={voting_res['avg_accuracy']:.4f} "
          f"LL={voting_res['avg_log_loss']:.4f} "
          f"F1={voting_res['avg_f1_macro']:.4f} ({time.time()-t0:.0f}s) [Δ={d:+.4f}]")

    factories = {
        "rf": make_rf, "hgb": make_hgb, "lr": make_lr,
        "stacking": make_stacking, "voting": make_voting,
    }
    return stacking_res, voting_res, factories


# ===========================================================================
# Phase 4: Calibration
# ===========================================================================

def phase4(df, features, le, best_name, make_best):
    sep("FASE 4: Calibración de Probabilidades")
    print(f"  Modelo a calibrar: {best_name}")

    results = {}
    for method in ["sigmoid", "isotonic"]:
        def make_cal(m=method):
            return CalibratedClassifierCV(make_best(), cv=3, method=m)

        res = walk_forward_eval(df, features, le, make_cal)
        results[method] = res
        d = res["avg_accuracy"] - V3_ACCURACY
        print(f"    {method:<10s}: Acc={res['avg_accuracy']:.4f} "
              f"LL={res['avg_log_loss']:.4f} F1={res['avg_f1_macro']:.4f} [Δ={d:+.4f}]")

    # Pick by log_loss (calibration primarily improves probability quality)
    best_method = min(results, key=lambda m: results[m]["avg_log_loss"])
    print(f"  ➜ Mejor calibración: {best_method}")

    def make_cal_final(m=best_method):
        return CalibratedClassifierCV(make_best(), cv=3, method=m)

    return results[best_method], make_cal_final, best_method


# ===========================================================================
# Phase 5: Comparison Table
# ===========================================================================

def phase5(all_results, n_selected):
    sep("FASE 5: Tabla Comparativa Final")

    hdr = (f"  {'Modelo':<35s} {'Feat':>5s} {'Accuracy':>9s} "
           f"{'Log Loss':>9s} {'F1 Macro':>9s} {'Δ vs v3':>9s}")
    print(hdr)
    print(f"  {'-'*35} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

    best_name, best_acc = None, 0
    for name, res in all_results.items():
        n = res.get("n_features", n_selected)
        acc, ll, f1 = res["avg_accuracy"], res["avg_log_loss"], res["avg_f1_macro"]
        delta = acc - V3_ACCURACY
        print(f"  {name:<35s} {n:>5d} {acc:>9.4f} {ll:>9.4f} {f1:>9.4f} {delta:>+9.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = name

    # Mark best
    best_res = all_results[best_name]
    print(f"\n  MEJOR MODELO: {best_name}")
    print(f"    Accuracy:       {best_res['avg_accuracy']:.4f}")
    print(f"    Log Loss:       {best_res['avg_log_loss']:.4f}")
    print(f"    F1 Macro:       {best_res['avg_f1_macro']:.4f}")
    print(f"    vs v3 accuracy: {best_res['avg_accuracy'] - V3_ACCURACY:+.4f}")
    print(f"    vs v3 log loss: {best_res['avg_log_loss'] - V3_LOG_LOSS:+.4f}")
    print(f"    vs baseline:    +{best_res['avg_accuracy'] - BASELINE_ACC:.4f}")

    return best_name, best_res


# ===========================================================================
# Analysis helpers
# ===========================================================================

def phase_analysis(best_res, df):
    """Performance by match phase."""
    sep("ANÁLISIS POR FASE")

    y_true = np.array(best_res["all_true"])
    y_pred = np.array(best_res["all_pred"])

    splits = build_splits(df)
    val_rows = pd.concat([val for _, _, val in splits]).reset_index(drop=True)
    val_rows["phase"] = val_rows["round"].apply(classify_phase)

    for phase in ["Qualifying", "Group/League", "Knockouts"]:
        mask = (val_rows["phase"] == phase).values[:len(y_true)]
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        print(f"  {phase:<15s}: {acc:.4f} ({mask.sum()} partidos)")


def draw_analysis(best_res):
    """Draw prediction quality."""
    sep("ANÁLISIS DE DRAWS")

    y_true = np.array(best_res["all_true"])
    y_pred = np.array(best_res["all_pred"])

    cm = confusion_matrix(y_true, y_pred, labels=["A", "D", "H"])
    # cm rows = actual, cols = predicted. Index 1 = Draw
    draw_correct = cm[1, 1]
    draw_actual = cm[1, :].sum()
    draw_predicted = cm[:, 1].sum()

    recall = draw_correct / draw_actual if draw_actual > 0 else 0
    precision = draw_correct / draw_predicted if draw_predicted > 0 else 0

    print(f"  Draws reales:    {draw_actual}/{len(y_true)} "
          f"({draw_actual/len(y_true)*100:.1f}%)")
    print(f"  Draws predichos: {draw_predicted}/{len(y_pred)} "
          f"({draw_predicted/len(y_pred)*100:.1f}%)")
    print(f"  Draw Recall:     {recall:.4f} "
          f"(de los draws reales, ¿cuántos predijimos?)")
    print(f"  Draw Precision:  {precision:.4f} "
          f"(de los draws predichos, ¿cuántos fueron reales?)")

    if recall < 0.15:
        print(f"\n  HONESTAMENTE: El modelo sigue sin poder predecir empates de forma fiable.")
        print(f"  Esto es normal — los empates son el resultado más difícil en fútbol.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    sep("08 — ENSEMBLE TUNING PIPELINE")
    t_start = time.time()

    # Load data
    df = pd.read_csv(PROCESSED_DIR / "features_v3.csv")
    df["date"] = pd.to_datetime(df["date"])
    with open(MODELS_DIR / "feature_list_v3.json") as f:
        all_features = json.load(f)
    le = LabelEncoder()
    le.fit(CLASSES)

    print(f"  Dataset: {df.shape}")
    print(f"  Features v3: {len(all_features)}")
    print(f"  Target: v3 Acc={V3_ACCURACY} LL={V3_LOG_LOSS}")

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    selected, ranked, subset_results = phase1(df, all_features, le)

    # -----------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------
    tuning_results, best_params = phase2(df, selected, le)

    # -----------------------------------------------------------------------
    # Phase 3
    # -----------------------------------------------------------------------
    stacking_res, voting_res, factories = phase3(df, selected, le, best_params)

    # -----------------------------------------------------------------------
    # Collect results so far
    # -----------------------------------------------------------------------
    n_sel = len(selected)
    all_results = {}

    # v3 baseline (re-verified in walk-forward)
    all_results["v3 baseline (RF, 47 feat)"] = {
        **subset_results[47], "n_features": 47,
    }

    # Tuned individual models
    name_map = {"Random Forest": "RF tuned", "HistGradientBoosting": "HGB tuned",
                "Logistic Regression": "LogReg tuned"}
    for orig, short in name_map.items():
        all_results[short] = {**tuning_results[orig], "n_features": n_sel}

    all_results["Stacking"] = {**stacking_res, "n_features": n_sel}
    all_results["Voting"] = {**voting_res, "n_features": n_sel}

    # -----------------------------------------------------------------------
    # Phase 4: calibrate the best non-baseline model
    # -----------------------------------------------------------------------
    candidates = {k: v for k, v in all_results.items() if "baseline" not in k}
    best_pre_cal = max(candidates, key=lambda n: candidates[n]["avg_accuracy"])

    factory_map = {
        "RF tuned": factories["rf"],
        "HGB tuned": factories["hgb"],
        "LogReg tuned": factories["lr"],
        "Stacking": factories["stacking"],
        "Voting": factories["voting"],
    }
    cal_res, make_cal, cal_method = phase4(
        df, selected, le, best_pre_cal, factory_map[best_pre_cal])
    all_results[f"{best_pre_cal} + cal({cal_method})"] = {
        **cal_res, "n_features": n_sel,
    }

    # -----------------------------------------------------------------------
    # Phase 5: comparison
    # -----------------------------------------------------------------------
    best_name, best_res = phase5(all_results, n_sel)
    phase_analysis(best_res, df)
    draw_analysis(best_res)

    # -----------------------------------------------------------------------
    # Save decision
    # -----------------------------------------------------------------------
    sep("DECISIÓN")

    improved = (best_res["avg_accuracy"] > V3_ACCURACY or
                (best_res["avg_accuracy"] >= V3_ACCURACY - 0.005
                 and best_res["avg_log_loss"] < V3_LOG_LOSS - 0.02))

    if improved and "baseline" not in best_name:
        print(f"\n  MEJORA CONFIRMADA: {best_name}")
        print(f"  Guardando modelo final...")

        # Determine the factory for final model
        save_name = best_name.replace(f" + cal({cal_method})", "")
        if save_name in factory_map:
            if "cal" in best_name:
                make_final = make_cal
            else:
                make_final = factory_map[save_name]
        else:
            make_final = factory_map.get(best_pre_cal, factories["rf"])

        # Train on all data
        final_pipe = make_final()
        final_pipe.fit(df[selected].values, le.transform(df["result"]))
        train_acc = accuracy_score(
            le.transform(df["result"]),
            final_pipe.predict(df[selected].values))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / "champion_model_final.pkl"
        joblib.dump(final_pipe, model_path)
        print(f"  Saved: {model_path}")

        feat_path = MODELS_DIR / "feature_list_final.json"
        with open(feat_path, "w") as f:
            json.dump(selected, f, indent=2)
        print(f"  Saved: {feat_path}")

        metrics_final = {
            "model": best_name,
            "n_features": len(selected),
            "features": selected,
            "accuracy": round(best_res["avg_accuracy"], 4),
            "log_loss": round(best_res["avg_log_loss"], 4),
            "f1_macro": round(best_res["avg_f1_macro"], 4),
            "delta_accuracy_vs_v3": round(best_res["avg_accuracy"] - V3_ACCURACY, 4),
            "delta_log_loss_vs_v3": round(best_res["avg_log_loss"] - V3_LOG_LOSS, 4),
            "training_accuracy": round(train_acc, 4),
            "best_params": {k: str(v) for k, v in best_params.items()},
            "per_split": best_res.get("per_split", []),
        }
        met_path = MODELS_DIR / "metrics_final.json"
        with open(met_path, "w") as f:
            json.dump(metrics_final, f, indent=2, default=str)
        print(f"  Saved: {met_path}")
    else:
        print(f"\n  NO MEJORO lo suficiente.")
        print(f"  Mejor: {best_name} Acc={best_res['avg_accuracy']:.4f} "
              f"(v3={V3_ACCURACY})")
        print(f"  El modelo v3 sigue siendo el mejor para producción.")

        # Save comparison metrics for reference
        metrics_final = {
            "improved": False,
            "best_model": best_name,
            "best_accuracy": round(best_res["avg_accuracy"], 4),
            "v3_accuracy": V3_ACCURACY,
            "all_models": {
                name: {
                    "accuracy": round(r["avg_accuracy"], 4),
                    "log_loss": round(r["avg_log_loss"], 4),
                }
                for name, r in all_results.items()
            },
        }
        met_path = MODELS_DIR / "metrics_final.json"
        with open(met_path, "w") as f:
            json.dump(metrics_final, f, indent=2)
        print(f"  Saved for reference: {met_path}")

    elapsed = time.time() - t_start
    print(f"\n  Tiempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
