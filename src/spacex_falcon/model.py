"""Leakage safe model training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["FlightNumber", "PayloadMass", "Flights", "Block", "ReusedCount"]
BOOLEAN_FEATURES = ["GridFins", "Reused", "Legs"]
CATEGORICAL_FEATURES = ["Orbit", "LaunchSite", "BoosterVersion"]
MODEL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ModelReport:
    """Metrics from an untouched, later launch holdout set."""

    train_samples: int
    test_samples: int
    train_max_flight: int
    test_min_flight: int
    baseline_accuracy: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _to_float(values):
    """Convert boolean transformer input to a numeric array."""
    return values.astype(float)


def build_pipeline() -> Pipeline:
    """Create preprocessing and classification as one fitted unit."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    boolean = Pipeline(
        steps=[
            ("to_float", FunctionTransformer(_to_float)),
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("boolean", boolean, BOOLEAN_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def train_temporal_model(
    data: pd.DataFrame, test_fraction: float = 0.20
) -> tuple[Pipeline, ModelReport]:
    """Evaluate on later launches, then fit the delivered model on all records."""
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4")
    if len(data) < 20:
        raise ValueError("At least 20 records are required for temporal evaluation")

    ordered = data.sort_values("FlightNumber").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_index]
    test = ordered.iloc[split_index:]
    if train["Class"].nunique() < 2:
        raise ValueError("Training records must contain both target classes")

    evaluation_model = build_pipeline()
    evaluation_model.fit(train[MODEL_FEATURES], train["Class"])
    prediction = evaluation_model.predict(test[MODEL_FEATURES])
    probability = evaluation_model.predict_proba(test[MODEL_FEATURES])[:, 1]
    baseline_class = int(train["Class"].mode().iloc[0])
    baseline_prediction = [baseline_class] * len(test)
    roc_auc = (
        float(roc_auc_score(test["Class"], probability)) if test["Class"].nunique() == 2 else None
    )
    report = ModelReport(
        train_samples=len(train),
        test_samples=len(test),
        train_max_flight=int(train["FlightNumber"].max()),
        test_min_flight=int(test["FlightNumber"].min()),
        baseline_accuracy=float(accuracy_score(test["Class"], baseline_prediction)),
        accuracy=float(accuracy_score(test["Class"], prediction)),
        balanced_accuracy=float(balanced_accuracy_score(test["Class"], prediction)),
        precision=float(precision_score(test["Class"], prediction, zero_division=0)),
        recall=float(recall_score(test["Class"], prediction, zero_division=0)),
        f1=float(f1_score(test["Class"], prediction, zero_division=0)),
        roc_auc=roc_auc,
    )

    final_model = clone(evaluation_model)
    final_model.fit(ordered[MODEL_FEATURES], ordered["Class"])
    return final_model, report


def prediction_record(
    data: pd.DataFrame,
    *,
    flight_number: float,
    payload_mass: float,
    orbit: str,
    launch_site: str,
    grid_fins: bool,
    reused: bool,
    legs: bool,
) -> pd.DataFrame:
    """Create one inference row and fill less useful controls from recent data."""
    latest = data.sort_values("FlightNumber").iloc[-1]
    record = {
        "FlightNumber": flight_number,
        "PayloadMass": payload_mass,
        "Flights": latest["Flights"],
        "Block": latest["Block"],
        "ReusedCount": latest["ReusedCount"],
        "GridFins": grid_fins,
        "Reused": reused,
        "Legs": legs,
        "Orbit": orbit,
        "LaunchSite": launch_site,
        "BoosterVersion": latest["BoosterVersion"],
    }
    return pd.DataFrame([record], columns=MODEL_FEATURES)
