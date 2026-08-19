"""Deployment model training and explicit prediction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.base import BaseEstimator

from spacex_falcon.evaluation import backtest_model, build_model, summarize_backtest
from spacex_falcon.features import DEPLOYMENT_FEATURES

MODEL_FEATURES = DEPLOYMENT_FEATURES.columns


@dataclass(frozen=True)
class ModelReport:
    """Aggregate metrics from expanding window temporal evaluation."""

    folds: int
    evaluation_samples: int
    accuracy: float
    balanced_accuracy: float
    balanced_accuracy_ci_low: float
    balanced_accuracy_ci_high: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier: float
    log_loss: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def train_temporal_model(data: pd.DataFrame) -> tuple[BaseEstimator, ModelReport]:
    """Backtest logistic probabilities, then fit the deployment model on all records."""
    folds, predictions = backtest_model(data, "logistic_regression", DEPLOYMENT_FEATURES)
    summary = summarize_backtest(folds, predictions)
    report = ModelReport(
        folds=int(summary["Folds"]),
        evaluation_samples=int(summary["EvaluationSamples"]),
        accuracy=float(summary["Accuracy"]),
        balanced_accuracy=float(summary["BalancedAccuracy"]),
        balanced_accuracy_ci_low=float(summary["BalancedAccuracyCILow"]),
        balanced_accuracy_ci_high=float(summary["BalancedAccuracyCIHigh"]),
        precision=float(summary["Precision"]),
        recall=float(summary["Recall"]),
        f1=float(summary["F1"]),
        roc_auc=float(summary["ROCAUC"]),
        brier=float(summary["Brier"]),
        log_loss=float(summary["LogLoss"]),
    )
    model = build_model("logistic_regression", DEPLOYMENT_FEATURES)
    model.fit(data[MODEL_FEATURES], data["Class"])
    return model, report


def prediction_record(
    *,
    flight_number: float,
    payload_mass: float,
    booster_prior_flights: float,
    orbit: str,
    launch_site: str,
    grid_fins: bool,
    reused: bool,
    legs: bool,
) -> pd.DataFrame:
    """Create a fully explicit feature vector for interactive inference."""
    record = {
        "FlightNumber": flight_number,
        "PayloadMass": payload_mass,
        "Flights": booster_prior_flights,
        "GridFins": grid_fins,
        "Reused": reused,
        "Legs": legs,
        "Orbit": orbit,
        "LaunchSite": launch_site,
    }
    return pd.DataFrame([record], columns=MODEL_FEATURES)
