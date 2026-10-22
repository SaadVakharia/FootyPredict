"""
FootyPredict: Test Suite
Tests for chronological splitting, Poisson scoreline math, probability outputs,
model loading, input validation, and data pipeline integrity.
"""

import os
import sys
import math
import pytest
import numpy as np
import pandas as pd

# Add src/ to path so imports work without package installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    load_data, prepare_features, split_and_scale,
    split_chronological, FEATURE_COLS, OUTCOME_MAPPING
)
from model import DualStageFootballPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal DataFrame mirroring the real matches.csv schema."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-08-01", periods=n, freq="3D")
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "HomeTeam": np.random.choice(["Arsenal", "Chelsea", "Liverpool"], n),
        "AwayTeam": np.random.choice(["Tottenham", "ManCity", "ManUtd"], n),
        "HomeElo": np.random.uniform(1700, 1950, n),
        "AwayElo": np.random.uniform(1700, 1950, n),
        "EloDiff": np.random.uniform(-200, 200, n),
        "HomeFormPts5": np.random.randint(0, 16, n),
        "AwayFormPts5": np.random.randint(0, 16, n),
        "HomeFormGD5": np.random.randint(-10, 10, n),
        "AwayFormGD5": np.random.randint(-10, 10, n),
        "RestDaysHome": np.random.randint(3, 14, n),
        "RestDaysAway": np.random.randint(3, 14, n),
        "HomeShots": np.random.randint(2, 25, n),
        "AwayShots": np.random.randint(2, 20, n),
        "HomeShotsOnTarget": np.random.randint(1, 12, n),
        "AwayShotsOnTarget": np.random.randint(1, 10, n),
        "HomeCorners": np.random.randint(1, 12, n),
        "AwayCorners": np.random.randint(1, 10, n),
        "FTHG": np.random.randint(0, 5, n),
        "FTAG": np.random.randint(0, 4, n),
        "FTR": np.random.choice(["H", "D", "A"], n, p=[0.45, 0.25, 0.30]),
    })
    return df


@pytest.fixture
def real_df():
    """Load the actual matches.csv from data/."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")
    if not os.path.exists(csv_path):
        pytest.skip("matches.csv not found")
    return pd.read_csv(csv_path)


@pytest.fixture
def trained_model(sample_df):
    """A DualStageFootballPredictor fitted on sample data."""
    data = split_chronological(sample_df, test_fraction=0.2)
    model = DualStageFootballPredictor(classifier_type="gradient_boosting")
    model.fit(
        data["X_train"], data["y_out_train"],
        data["y_hg_train"], data["y_ag_train"]
    )
    return model, data


# ---------------------------------------------------------------------------
# 1. Chronological Split Tests
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    """Verify the chronological split prevents temporal leakage."""

    def test_split_preserves_temporal_order(self, sample_df):
        """Train dates must all precede test dates."""
        data = split_chronological(sample_df, test_fraction=0.2)
        train_end = data["train_date_range"][1]
        test_start = data["test_date_range"][0]
        assert train_end <= test_start, (
            f"Temporal leakage: train ends at {train_end} but test starts at {test_start}"
        )

    def test_split_sizes(self, sample_df):
        """Train and test sizes should match the requested fraction."""
        data = split_chronological(sample_df, test_fraction=0.2)
        total = len(sample_df)
        expected_train = int(total * 0.8)
        assert data["X_train"].shape[0] == expected_train
        assert data["X_test"].shape[0] == total - expected_train

    def test_no_data_loss(self, sample_df):
        """All samples must appear in either train or test, with no overlap or loss."""
        data = split_chronological(sample_df, test_fraction=0.2)
        total_rows = data["X_train"].shape[0] + data["X_test"].shape[0]
        assert total_rows == len(sample_df)

    def test_output_keys_match_random_split(self, sample_df):
        """Chronological split dict must have at least all keys from random split."""
        data_chrono = split_chronological(sample_df, test_fraction=0.2)
        X, y_out, y_hg, y_ag = prepare_features(sample_df)
        data_random = split_and_scale(X, y_out, y_hg, y_ag)
        for key in data_random:
            assert key in data_chrono, f"Missing key in chronological split: {key}"

    def test_split_method_label(self, sample_df):
        """split_method should be 'chronological'."""
        data = split_chronological(sample_df, test_fraction=0.2)
        assert data["split_method"] == "chronological"

    def test_random_split_method_label(self, sample_df):
        """split_method should be 'random_stratified' for random split."""
        X, y_out, y_hg, y_ag = prepare_features(sample_df)
        data = split_and_scale(X, y_out, y_hg, y_ag)
        assert data["split_method"] == "random_stratified"

    def test_invalid_missing_date_column(self, sample_df):
        """Should raise ValueError when Date column is missing."""
        df_no_date = sample_df.drop(columns=["Date"])
        with pytest.raises(ValueError, match="Date"):
            split_chronological(df_no_date)

    def test_invalid_test_fraction_zero(self, sample_df):
        """test_fraction=0 should raise ValueError."""
        with pytest.raises(ValueError):
            split_chronological(sample_df, test_fraction=0.0)

    def test_invalid_test_fraction_one(self, sample_df):
        """test_fraction=1 should raise ValueError."""
        with pytest.raises(ValueError):
            split_chronological(sample_df, test_fraction=1.0)

    def test_chronological_on_real_data(self, real_df):
        """Chronological split works on the actual matches.csv dataset."""
        data = split_chronological(real_df, test_fraction=0.2)
        assert data["train_date_range"][1] <= data["test_date_range"][0]
        assert data["X_train"].shape[0] > 0
        assert data["X_test"].shape[0] > 0


# ---------------------------------------------------------------------------
# 2. Poisson Scoreline Math Tests
# ---------------------------------------------------------------------------

class TestPoissonScoreline:
    """Verify the Poisson probability matrix computation is mathematically correct."""

    def test_matrix_sums_to_one(self):
        """Normalized probability matrix should sum to ~1.0."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(1.5, 1.2, max_goals=6)
        assert abs(mat.sum() - 1.0) < 1e-9, f"Matrix sum is {mat.sum()}, expected ~1.0"

    def test_all_probabilities_non_negative(self):
        """Every cell must be >= 0."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(2.0, 0.8, max_goals=6)
        assert np.all(mat >= 0), "Negative probability found in score matrix"

    def test_all_probabilities_at_most_one(self):
        """Every cell must be <= 1."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(2.0, 0.8, max_goals=6)
        assert np.all(mat <= 1.0), "Probability > 1.0 found in score matrix"

    def test_matrix_shape(self):
        """Matrix shape must match max_goals."""
        for mg in [4, 6, 8]:
            mat = DualStageFootballPredictor.compute_scoreline_matrix(1.5, 1.0, max_goals=mg)
            assert mat.shape == (mg, mg), f"Expected shape ({mg},{mg}), got {mat.shape}"

    def test_known_poisson_value(self):
        """Verify P(0,0) against manual Poisson calculation for known lambdas."""
        lam_h, lam_a = 1.5, 1.0
        # P(0,0) = e^{-lam_h} * e^{-lam_a} (before normalization)
        expected_raw = math.exp(-lam_h) * math.exp(-lam_a)
        mat = DualStageFootballPredictor.compute_scoreline_matrix(lam_h, lam_a, max_goals=6)
        # After normalization, P(0,0) should be close to expected_raw / sum_raw
        # Since max_goals=6 covers most of the probability mass, the normalization
        # factor should be close to 1.0
        assert abs(mat[0, 0] - expected_raw) < 0.02, (
            f"P(0,0) = {mat[0, 0]}, expected ~{expected_raw:.6f}"
        )

    def test_home_draw_away_decomposition(self):
        """Poisson matrix should decompose into home/draw/away probabilities summing to 1."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(1.8, 1.2, max_goals=6)
        p_home = np.tril(mat, -1).sum()  # home goals > away
        p_draw = np.trace(mat)            # home goals == away
        p_away = np.triu(mat, 1).sum()    # home goals < away
        total = p_home + p_draw + p_away
        assert abs(total - 1.0) < 1e-9, f"H/D/A sum is {total}, expected 1.0"

    def test_symmetric_lambdas_equal_home_away(self):
        """When lambdas are equal, P(home win) should equal P(away win)."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(1.5, 1.5, max_goals=6)
        p_home = np.tril(mat, -1).sum()
        p_away = np.triu(mat, 1).sum()
        assert abs(p_home - p_away) < 1e-9, (
            f"With equal lambdas, P(home)={p_home:.6f} != P(away)={p_away:.6f}"
        )

    def test_extreme_lambda_high(self):
        """High lambda should still produce valid probability matrix."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(4.5, 0.3, max_goals=8)
        assert abs(mat.sum() - 1.0) < 1e-9
        assert np.all(mat >= 0)

    def test_extreme_lambda_low(self):
        """Very low lambda should still produce valid probability matrix."""
        mat = DualStageFootballPredictor.compute_scoreline_matrix(0.2, 0.2, max_goals=6)
        assert abs(mat.sum() - 1.0) < 1e-9
        assert np.all(mat >= 0)
        # With very low lambdas, 0-0 should be the most likely scoreline
        assert mat[0, 0] == mat.max()


# ---------------------------------------------------------------------------
# 3. Probability Output Tests
# ---------------------------------------------------------------------------

class TestProbabilityOutputs:
    """Verify model probability outputs are valid probability distributions."""

    def test_outcome_probs_sum_to_one(self, trained_model):
        """Outcome probabilities for each prediction should sum to ~1.0."""
        model, data = trained_model
        probs = model.predict_outcome_proba(data["X_test"])
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
            err_msg="Outcome probabilities do not sum to 1.0")

    def test_outcome_probs_in_range(self, trained_model):
        """Each probability should be in [0, 1]."""
        model, data = trained_model
        probs = model.predict_outcome_proba(data["X_test"])
        assert np.all(probs >= 0), "Negative probability in outcome predictions"
        assert np.all(probs <= 1), "Probability > 1.0 in outcome predictions"

    def test_outcome_probs_three_classes(self, trained_model):
        """Output should have exactly 3 columns (Home, Draw, Away)."""
        model, data = trained_model
        probs = model.predict_outcome_proba(data["X_test"])
        assert probs.shape[1] == 3, f"Expected 3 classes, got {probs.shape[1]}"

    def test_expected_goals_positive(self, trained_model):
        """Expected goals (xG) should always be positive."""
        model, data = trained_model
        lam_h, lam_a = model.predict_expected_goals(data["X_test"])
        assert np.all(lam_h > 0), "Home xG contains non-positive values"
        assert np.all(lam_a > 0), "Away xG contains non-positive values"

    def test_expected_goals_reasonable_range(self, trained_model):
        """xG values should be within a reasonable football range (0.2 to 5.0)."""
        model, data = trained_model
        lam_h, lam_a = model.predict_expected_goals(data["X_test"])
        assert np.all(lam_h >= 0.2), "Home xG below minimum clip threshold"
        assert np.all(lam_h <= 5.0), "Home xG above maximum clip threshold"
        assert np.all(lam_a >= 0.2), "Away xG below minimum clip threshold"
        assert np.all(lam_a <= 5.0), "Away xG above maximum clip threshold"


# ---------------------------------------------------------------------------
# 4. Full Predict Match Pipeline Tests
# ---------------------------------------------------------------------------

class TestPredictMatch:
    """Verify the predict_match() end-to-end inference pipeline."""

    def test_predict_match_structure(self, trained_model):
        """predict_match should return dict with expected keys."""
        model, data = trained_model
        result = model.predict_match(data["X_test"][0])
        assert "outcome_probs" in result
        assert "expected_goals" in result
        assert "top_scorelines" in result
        assert "score_matrix" in result

    def test_predict_match_outcome_probs_valid(self, trained_model):
        """outcome_probs values should be valid probabilities summing to ~1."""
        model, data = trained_model
        result = model.predict_match(data["X_test"][0])
        probs = result["outcome_probs"]
        assert set(probs.keys()) == {"Home Win", "Draw", "Away Win"}
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-4, f"Outcome probs sum to {total}"
        for label, p in probs.items():
            assert 0 <= p <= 1, f"{label} probability {p} out of [0,1]"

    def test_predict_match_top_scorelines(self, trained_model):
        """Top scorelines should have valid structure and probabilities."""
        model, data = trained_model
        result = model.predict_match(data["X_test"][0])
        top = result["top_scorelines"]
        assert len(top) == 5, f"Expected 5 top scorelines, got {len(top)}"
        for entry in top:
            assert "score" in entry
            assert "home_goals" in entry
            assert "away_goals" in entry
            assert "probability" in entry
            assert entry["home_goals"] >= 0
            assert entry["away_goals"] >= 0
            assert 0 < entry["probability"] <= 1

    def test_predict_match_scorelines_sorted(self, trained_model):
        """Top scorelines should be sorted by descending probability."""
        model, data = trained_model
        result = model.predict_match(data["X_test"][0])
        probs = [s["probability"] for s in result["top_scorelines"]]
        assert probs == sorted(probs, reverse=True), "Scorelines not sorted by probability"


# ---------------------------------------------------------------------------
# 5. Model Initialization & Invalid Input Tests
# ---------------------------------------------------------------------------

class TestModelValidation:
    """Test model initialization and error handling."""

    def test_unknown_classifier_type(self):
        """Unknown classifier type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown classifier"):
            DualStageFootballPredictor(classifier_type="xgboost_not_supported")

    def test_valid_classifier_types(self):
        """All documented classifier types should initialize without error."""
        for ctype in ["logistic_regression", "random_forest", "gradient_boosting", "mlp"]:
            model = DualStageFootballPredictor(classifier_type=ctype)
            assert model.classifier_type == ctype
            assert model.fitted is False

    def test_fitted_flag(self, trained_model):
        """After fitting, the fitted flag should be True."""
        model, _ = trained_model
        assert model.fitted is True


# ---------------------------------------------------------------------------
# 6. Data Loading & Feature Extraction Tests
# ---------------------------------------------------------------------------

class TestDataPipeline:
    """Test data loading and feature extraction."""

    def test_load_real_data(self, real_df):
        """matches.csv should load with expected columns."""
        for col in FEATURE_COLS:
            assert col in real_df.columns, f"Missing feature column: {col}"
        assert "FTR" in real_df.columns
        assert "FTHG" in real_df.columns
        assert "FTAG" in real_df.columns
        assert "Date" in real_df.columns

    def test_prepare_features_shapes(self, sample_df):
        """prepare_features should return arrays with correct shapes."""
        X, y_out, y_hg, y_ag = prepare_features(sample_df)
        n = len(sample_df)
        assert X.shape == (n, len(FEATURE_COLS))
        assert y_out.shape == (n,)
        assert y_hg.shape == (n,)
        assert y_ag.shape == (n,)

    def test_outcome_mapping_values(self, sample_df):
        """Outcome labels should only contain 0, 1, 2."""
        _, y_out, _, _ = prepare_features(sample_df)
        unique_vals = set(y_out)
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected outcome values: {unique_vals}"

    def test_feature_cols_no_nans(self, real_df):
        """Feature columns in the real dataset should not contain NaN values."""
        X = real_df[FEATURE_COLS]
        nan_counts = X.isna().sum()
        assert nan_counts.sum() == 0, f"NaN values found in features:\n{nan_counts[nan_counts > 0]}"


# ---------------------------------------------------------------------------
# 7. Model Persistence Tests
# ---------------------------------------------------------------------------

class TestModelPersistence:
    """Test that saved models can be loaded and produce valid predictions."""

    def test_saved_model_loads(self):
        """The saved dual_stage_model.pkl should load successfully."""
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "dual_stage_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("dual_stage_model.pkl not found (run train.py first)")
        import joblib
        model = joblib.load(model_path)
        assert isinstance(model, DualStageFootballPredictor)
        assert model.fitted is True

    def test_saved_scaler_loads(self):
        """The saved scaler.pkl should load successfully."""
        scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
        if not os.path.exists(scaler_path):
            pytest.skip("scaler.pkl not found (run train.py first)")
        import joblib
        from sklearn.preprocessing import StandardScaler
        scaler = joblib.load(scaler_path)
        assert isinstance(scaler, StandardScaler)

    def test_evaluation_results_valid(self):
        """evaluation_results.json should parse and contain expected keys."""
        results_path = os.path.join(os.path.dirname(__file__), "..", "models", "evaluation_results.json")
        if not os.path.exists(results_path):
            pytest.skip("evaluation_results.json not found (run train.py first)")
        import json
        with open(results_path, "r") as f:
            data = json.load(f)
        assert "metrics_summary" in data
        assert "confusion_matrix" in data
        assert "classes" in data
        assert len(data["classes"]) == 3
