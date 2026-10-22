"""
FootyPredict: Dual-Stage Machine Learning Architecture
Stage 1: Outcome Classifier (Home Win / Draw / Away Win)
Stage 2: Poisson Goal Expectancy Regressor & Scoreline Probability Matrix
"""

import math
import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DualStageFootballPredictor:
    """
    Unified dual-stage predictor combining:
    1. Multi-class outcome classification
    2. Bivariate Poisson scoreline and xG modeling
    """
    def __init__(self, classifier_type="gradient_boosting"):
        self.classifier_type = classifier_type
        self.clf = self._init_classifier(classifier_type)
        self.home_goal_model = PoissonRegressor(alpha=0.1, max_iter=500)
        self.away_goal_model = PoissonRegressor(alpha=0.1, max_iter=500)
        self.fitted = False

    def _init_classifier(self, clf_type):
        if clf_type == "logistic_regression":
            return LogisticRegression(C=0.6, max_iter=1000, random_state=42)
        elif clf_type == "random_forest":
            return RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=2, random_state=42)
        elif clf_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=130, learning_rate=0.035, max_depth=4, random_state=42)
        elif clf_type == "mlp":
            return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {clf_type}")

    def fit(self, X_train, y_outcome, y_home_goals, y_away_goals):
        """Fits both Stage 1 outcome classifier and Stage 2 goal expectancy models."""
        self.clf.fit(X_train, y_outcome)
        self.home_goal_model.fit(X_train, y_home_goals)
        self.away_goal_model.fit(X_train, y_away_goals)
        self.fitted = True
        return self

    def predict_outcome_proba(self, X):
        """Returns [P(Home Win), P(Draw), P(Away Win)]."""
        return self.clf.predict_proba(X)

    def predict_expected_goals(self, X):
        """Returns predicted lambda for home and away goals (xG)."""
        lam_h = np.clip(self.home_goal_model.predict(X), 0.2, 5.0)
        lam_a = np.clip(self.away_goal_model.predict(X), 0.2, 5.0)
        return lam_h, lam_a

    @staticmethod
    def compute_scoreline_matrix(lam_h, lam_a, max_goals=6):
        """
        Computes joint bivariate Poisson probability matrix:
        P(X=x, Y=y) = (lam_h^x * e^-lam_h / x!) * (lam_a^y * e^-lam_a / y!)
        """
        prob_matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            p_x = (lam_h ** x) * math.exp(-lam_h) / math.factorial(x)
            for y in range(max_goals):
                p_y = (lam_a ** y) * math.exp(-lam_a) / math.factorial(y)
                prob_matrix[x, y] = p_x * p_y

        # Normalize matrix to sum to 1.0
        prob_matrix /= prob_matrix.sum()
        return prob_matrix

    def predict_match(self, X_single, max_score=6):
        """
        Full inference pipeline for a single match feature vector:
        - Outcome probabilities
        - Expected goals (xG)
        - Top 5 most likely exact scorelines
        """
        if len(X_single.shape) == 1:
            X_single = X_single.reshape(1, -1)

        outcome_probs = self.predict_outcome_proba(X_single)[0]
        lam_h, lam_a = self.predict_expected_goals(X_single)
        lam_h, lam_a = float(lam_h[0]), float(lam_a[0])

        prob_matrix = self.compute_scoreline_matrix(lam_h, lam_a, max_goals=max_score)

        # Extract top scorelines
        flat_indices = np.argsort(prob_matrix.ravel())[::-1]
        top_scores = []
        for idx in flat_indices[:5]:
            h_g = idx // max_score
            a_g = idx % max_score
            top_scores.append({
                "score": f"{h_g} - {a_g}",
                "home_goals": int(h_g),
                "away_goals": int(a_g),
                "probability": float(prob_matrix[h_g, a_g])
            })

        return {
            "outcome_probs": {
                "Home Win": float(outcome_probs[0]),
                "Draw": float(outcome_probs[1]),
                "Away Win": float(outcome_probs[2])
            },
            "expected_goals": {
                "home_xG": round(lam_h, 2),
                "away_xG": round(lam_a, 2)
            },
            "top_scorelines": top_scores,
            "score_matrix": prob_matrix.tolist()
        }

def benchmark_all_classifiers(X_train, y_train, X_test, y_test):
    """
    Trains and compares multiple classifiers on the mandatory academic metrics:
    Accuracy, Precision (macro), Recall (macro), and F1-Score (macro & weighted).
    """
    models = {
        "Logistic Regression": LogisticRegression(C=0.6, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=130, learning_rate=0.035, max_depth=4, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=500, random_state=42)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "Accuracy": round(acc * 100, 2),
            "Precision_Macro": round(prec_macro * 100, 2),
            "Recall_Macro": round(rec_macro * 100, 2),
            "F1_Macro": round(f1_macro * 100, 2),
            "F1_Weighted": round(f1_weighted * 100, 2),
            "model_obj": m
        }

    return results

if __name__ == "__main__":
    from data_loader import load_data, prepare_features, split_and_scale
    df = load_data()
    X, y_out, y_hg, y_ag = prepare_features(df)
    data = split_and_scale(X, y_out, y_hg, y_ag)

    print("Benchmarking Stage 1 Classifiers...")
    results = benchmark_all_classifiers(data["X_train"], data["y_out_train"], data["X_test"], data["y_out_test"])
    for name, mets in results.items():
        print(f"[{name}] Acc: {mets['Accuracy']}% | F1-Macro: {mets['F1_Macro']}% | F1-Weighted: {mets['F1_Weighted']}%")

    print("\nTraining Full Dual-Stage Predictor...")
    dual_pred = DualStageFootballPredictor(classifier_type="gradient_boosting")
    dual_pred.fit(data["X_train"], data["y_out_train"], data["y_hg_train"], data["y_ag_train"])
    test_sample = data["X_test"][0]
    prediction = dual_pred.predict_match(test_sample)
    print("Sample Match Prediction:", prediction["outcome_probs"])
    print("Expected Goals:", prediction["expected_goals"])
    print("Top Scorelines:", prediction["top_scorelines"])
