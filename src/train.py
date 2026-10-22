"""
FootyPredict: Training Pipeline & Evaluation Benchmark
Trains Stage 1 Classifiers, Stage 2 Poisson Regressors, and saves models & metrics.
"""

import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from data_loader import load_data, prepare_features, split_and_scale, split_chronological
from model import DualStageFootballPredictor

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_FIGS = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

def run_training_pipeline(use_chronological=True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_FIGS, exist_ok=True)

    print("[1/4] Loading and preparing dataset...")
    df = load_data()

    if use_chronological:
        print("  -> Using CHRONOLOGICAL split (prevents temporal leakage from rolling features)")
        data = split_chronological(df, test_fraction=0.2)
        print(f"     Train period: {data['train_date_range'][0]} to {data['train_date_range'][1]}")
        print(f"     Test  period: {data['test_date_range'][0]} to {data['test_date_range'][1]}")
    else:
        print("  -> Using RANDOM stratified split (legacy mode)")
        X, y_out, y_hg, y_ag = prepare_features(df)
        data = split_and_scale(X, y_out, y_hg, y_ag, test_size=0.2, random_state=42)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_out_train"], data["y_out_test"]
    y_hg_train, y_hg_test = data["y_hg_train"], data["y_hg_test"]
    y_ag_train, y_ag_test = data["y_ag_train"], data["y_ag_test"]

    print("[2/4] Benchmarking Classifiers on Test Set...")
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=500, random_state=42)
    }

    metrics_summary = {}
    fitted_models = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metrics_summary[name] = {
            "Accuracy": round(acc * 100, 2),
            "Precision_Macro": round(prec_macro * 100, 2),
            "Recall_Macro": round(rec_macro * 100, 2),
            "F1_Macro": round(f1_mac * 100, 2),
            "F1_Weighted": round(f1_wt * 100, 2)
        }
        fitted_models[name] = clf
        print(f"  -> {name:<20}: Accuracy={acc*100:5.2f}% | F1-Macro={f1_mac*100:5.2f}% | F1-Weighted={f1_wt*100:5.2f}%")

    print("[3/4] Fitting Dual-Stage Predictor (GBDT + Poisson)...")
    dual_predictor = DualStageFootballPredictor(classifier_type="gradient_boosting")
    dual_predictor.fit(X_train, y_train, y_hg_train, y_ag_train)

    # Blend predictions for final test evaluation
    clf_probs = dual_predictor.predict_outcome_proba(X_test)
    lam_h_all, lam_a_all = dual_predictor.predict_expected_goals(X_test)

    blended_preds = []
    alpha = 0.55  # 55% direct classifier, 45% Poisson matrix outcome integration
    for i in range(len(X_test)):
        c_p = clf_probs[i]
        mat = dual_predictor.compute_scoreline_matrix(lam_h_all[i], lam_a_all[i])
        p_home_pois = np.tril(mat, -1).sum()  # home goals > away goals
        p_draw_pois = np.trace(mat)           # home goals == away goals
        p_away_pois = np.triu(mat, 1).sum()   # home goals < away goals

        b_home = alpha * c_p[0] + (1 - alpha) * p_home_pois
        b_draw = alpha * c_p[1] + (1 - alpha) * p_draw_pois
        b_away = alpha * c_p[2] + (1 - alpha) * p_away_pois
        blended_preds.append(np.argmax([b_home, b_draw, b_away]))

    blended_preds = np.array(blended_preds)
    blend_acc = accuracy_score(y_test, blended_preds)
    blend_f1_mac = f1_score(y_test, blended_preds, average="macro", zero_division=0)
    blend_f1_wt = f1_score(y_test, blended_preds, average="weighted", zero_division=0)
    blend_prec = precision_score(y_test, blended_preds, average="macro", zero_division=0)
    blend_rec = recall_score(y_test, blended_preds, average="macro", zero_division=0)

    metrics_summary["FootyPredict (Dual-Stage Ensemble)"] = {
        "Accuracy": round(blend_acc * 100, 2),
        "Precision_Macro": round(blend_prec * 100, 2),
        "Recall_Macro": round(blend_rec * 100, 2),
        "F1_Macro": round(blend_f1_mac * 100, 2),
        "F1_Weighted": round(blend_f1_wt * 100, 2)
    }
    print(f"  -> {'FootyPredict Dual-Stage':<20}: Accuracy={blend_acc*100:5.2f}% | F1-Macro={blend_f1_mac*100:5.2f}% | F1-Weighted={blend_f1_wt*100:5.2f}%")

    # Confusion matrix for FootyPredict
    cm = confusion_matrix(y_test, blended_preds)

    print("[4/4] Persisting models and evaluation metrics...")
    joblib.dump(dual_predictor, os.path.join(MODELS_DIR, "dual_stage_model.pkl"))
    joblib.dump(data["scaler"], os.path.join(MODELS_DIR, "scaler.pkl"))

    results_payload = {
        "metrics_summary": metrics_summary,
        "confusion_matrix": cm.tolist(),
        "classes": ["Home Win", "Draw", "Away Win"],
        "feature_names": data["feature_names"],
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in zip(data["feature_names"], fitted_models["Gradient Boosting"].feature_importances_)
        },
        "sample_size": {
            "total": len(df),
            "train": len(X_train),
            "test": len(X_test)
        },
        "split_method": data["split_method"]
    }

    # Include date ranges if chronological split was used
    if "train_date_range" in data:
        results_payload["train_date_range"] = data["train_date_range"]
        results_payload["test_date_range"] = data["test_date_range"]

    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"[SUCCESS] Models saved to {MODELS_DIR}")
    return results_payload

if __name__ == "__main__":
    use_random = "--random-split" in sys.argv
    run_training_pipeline(use_chronological=not use_random)
