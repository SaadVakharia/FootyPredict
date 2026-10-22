"""
FootyPredict: Data Loader & Feature Engineering Pipeline
Handles loading match records, feature extraction, scaling, and train-test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    "EloDiff",
    "HomeElo",
    "AwayElo",
    "HomeFormPts5",
    "AwayFormPts5",
    "HomeFormGD5",
    "AwayFormGD5",
    "RestDaysHome",
    "RestDaysAway",
    "HomeShots",
    "AwayShots",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeCorners",
    "AwayCorners"
]

OUTCOME_MAPPING = {"H": 0, "D": 1, "A": 2}
INV_OUTCOME_MAPPING = {0: "Home Win", 1: "Draw", 2: "Away Win"}

def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    """
    Extracts numerical feature matrix X, outcome labels y_outcome,
    and goal targets y_home_goals, y_away_goals.
    """
    X = df[FEATURE_COLS].copy()
    y_outcome = df["FTR"].map(OUTCOME_MAPPING).values
    y_home_goals = df["FTHG"].values
    y_away_goals = df["FTAG"].values
    return X, y_outcome, y_home_goals, y_away_goals

def split_and_scale(X, y_outcome, y_home_goals, y_away_goals, test_size=0.2, random_state=42):
    """
    Stratified random split on outcome, with feature standardization.

    WARNING: This performs a random split which can cause temporal data leakage
    when features include rolling statistics (e.g., HomeFormPts5). Use
    split_chronological() for temporally valid evaluation.
    """
    X_train, X_test, y_out_train, y_out_test, y_hg_train, y_hg_test, y_ag_train, y_ag_test = train_test_split(
        X, y_outcome, y_home_goals, y_away_goals,
        test_size=test_size,
        random_state=random_state,
        stratify=y_outcome
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_out_train": y_out_train,
        "y_out_test": y_out_test,
        "y_hg_train": y_hg_train,
        "y_hg_test": y_hg_test,
        "y_ag_train": y_ag_train,
        "y_ag_test": y_ag_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "split_method": "random_stratified"
    }

def split_chronological(df, test_fraction=0.2):
    """
    Chronological train/test split that prevents temporal data leakage.

    Sorts matches by date, then assigns the earliest (1 - test_fraction) as
    training data and the most recent test_fraction as test data. This ensures
    rolling features (e.g., HomeFormPts5, HomeFormGD5) computed from past
    matches never leak future information into the training set.

    Args:
        df: DataFrame with a 'Date' column and all feature/target columns.
        test_fraction: Fraction of most recent matches to hold out (default 0.2).

    Returns:
        dict with the same keys as split_and_scale(), plus 'split_method',
        'train_date_range', and 'test_date_range'.

    Raises:
        ValueError: If 'Date' column is missing or test_fraction is invalid.
    """
    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column for chronological splitting.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be between 0 and 1 exclusive, got {test_fraction}.")

    df_sorted = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))

    if split_idx == 0 or split_idx == len(df_sorted):
        raise ValueError(
            f"Chronological split produced an empty set: "
            f"split_idx={split_idx}, total={len(df_sorted)}, test_fraction={test_fraction}"
        )

    df_train = df_sorted.iloc[:split_idx]
    df_test = df_sorted.iloc[split_idx:]

    X_train = df_train[FEATURE_COLS].values
    X_test = df_test[FEATURE_COLS].values
    y_out_train = df_train["FTR"].map(OUTCOME_MAPPING).values
    y_out_test = df_test["FTR"].map(OUTCOME_MAPPING).values
    y_hg_train = df_train["FTHG"].values
    y_hg_test = df_test["FTHG"].values
    y_ag_train = df_train["FTAG"].values
    y_ag_test = df_test["FTAG"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": pd.DataFrame(X_train, columns=FEATURE_COLS),
        "X_test_raw": pd.DataFrame(X_test, columns=FEATURE_COLS),
        "y_out_train": y_out_train,
        "y_out_test": y_out_test,
        "y_hg_train": y_hg_train,
        "y_hg_test": y_hg_test,
        "y_ag_train": y_ag_train,
        "y_ag_test": y_ag_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "split_method": "chronological",
        "train_date_range": (str(df_train["Date"].iloc[0]), str(df_train["Date"].iloc[-1])),
        "test_date_range": (str(df_test["Date"].iloc[0]), str(df_test["Date"].iloc[-1])),
    }

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} matches, {df.shape[1]} columns.")
    X, y_out, y_hg, y_ag = prepare_features(df)
    data = split_and_scale(X, y_out, y_hg, y_ag)
    print(f"Train samples: {data['X_train'].shape[0]}, Test samples: {data['X_test'].shape[0]}")
