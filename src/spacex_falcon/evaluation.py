"""Temporal experiments, uncertainty estimates, and diagnostic analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from spacex_falcon.features import (
    ABLATION_FEATURES,
    DEPLOYMENT_FEATURES,
    TIME_ONLY,
    FeatureSpec,
)


@dataclass(frozen=True)
class ExperimentResults:
    """Tables produced by the complete temporal study."""

    comparison: pd.DataFrame
    folds: pd.DataFrame
    predictions: pd.DataFrame
    ablations: pd.DataFrame
    calibration: pd.DataFrame
    calibration_predictions: pd.DataFrame


MODEL_LABELS = {
    "historical_rate": "Historical rate baseline",
    "time_only_logistic": "Time only logistic",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "calibrated_logistic": "Calibrated logistic",
}


def _to_float(values):
    return values.astype(float)


def build_preprocessor(features: FeatureSpec) -> ColumnTransformer:
    """Build typed preprocessing for one feature specification."""
    transformers = []
    if features.numeric:
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, list(features.numeric)))
    if features.boolean:
        boolean = Pipeline(
            [
                ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("boolean", boolean, list(features.boolean)))
    if features.categorical:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, list(features.categorical)))
    return ColumnTransformer(transformers=transformers, sparse_threshold=0)


def build_model(model_name: str, features: FeatureSpec = DEPLOYMENT_FEATURES) -> BaseEstimator:
    """Construct a deliberately small set of contrasting estimators."""
    if model_name == "time_only_logistic":
        features = TIME_ONLY
        model_name = "logistic_regression"

    preprocessor = build_preprocessor(features)
    if model_name in {"logistic_regression", "calibrated_logistic"}:
        classifier: BaseEstimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    if model_name == "calibrated_logistic":
        return CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3),
        )
    return pipeline


def expanding_windows(
    rows: int, initial_train_size: int = 40, test_size: int = 10
) -> list[tuple[int, int]]:
    """Return expanding train boundaries followed by fixed test windows."""
    if initial_train_size < 20 or test_size < 5:
        raise ValueError("Temporal windows are too small for this study")
    windows = []
    train_stop = initial_train_size
    while train_stop < rows:
        test_stop = min(train_stop + test_size, rows)
        if test_stop - train_stop < 5:
            break
        windows.append((train_stop, test_stop))
        train_stop = test_stop
    if not windows:
        raise ValueError("No temporal evaluation windows can be created")
    return windows


def _safe_roc_auc(actual: pd.Series, probability: np.ndarray) -> float:
    return float(roc_auc_score(actual, probability)) if actual.nunique() == 2 else float("nan")


def _metric_row(actual: pd.Series, predicted: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "BalancedAccuracy": (
            float(balanced_accuracy_score(actual, predicted))
            if actual.nunique() == 2
            else float("nan")
        ),
        "ROCAUC": _safe_roc_auc(actual, probability),
        "Brier": float(brier_score_loss(actual, probability)),
        "LogLoss": float(log_loss(actual, probability, labels=[0, 1])),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
    }


def backtest_model(
    data: pd.DataFrame,
    model_name: str,
    features: FeatureSpec = DEPLOYMENT_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model through expanding temporal windows."""
    ordered = data.sort_values("FlightNumber").reset_index(drop=True)
    predictions = []
    fold_metrics = []
    for fold_number, (train_stop, test_stop) in enumerate(expanding_windows(len(ordered)), 1):
        train = ordered.iloc[:train_stop]
        test = ordered.iloc[train_stop:test_stop]
        if model_name == "historical_rate":
            probability = np.repeat(float(train["Class"].mean()), len(test))
        else:
            estimator = build_model(model_name, features)
            estimator.fit(train[features.columns], train["Class"])
            probability = estimator.predict_proba(test[features.columns])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        metrics = _metric_row(test["Class"], predicted, probability)
        metrics.update(
            {
                "Model": MODEL_LABELS.get(model_name, model_name),
                "Fold": fold_number,
                "TrainEnd": int(train["FlightNumber"].max()),
                "TestStart": int(test["FlightNumber"].min()),
                "TestEnd": int(test["FlightNumber"].max()),
                "TestSamples": len(test),
            }
        )
        fold_metrics.append(metrics)
        for row_index, (_, row) in enumerate(test.iterrows()):
            predictions.append(
                {
                    "Model": MODEL_LABELS.get(model_name, model_name),
                    "Fold": fold_number,
                    "FlightNumber": int(row["FlightNumber"]),
                    "Actual": int(row["Class"]),
                    "Predicted": int(predicted[row_index]),
                    "Probability": float(probability[row_index]),
                }
            )
    return pd.DataFrame(fold_metrics), pd.DataFrame(predictions)


def bootstrap_interval(
    predictions: pd.DataFrame,
    metric: Callable[[pd.Series, pd.Series], float] = balanced_accuracy_score,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval over out of period predictions."""
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indices = generator.integers(0, len(predictions), len(predictions))
        sample = predictions.iloc[indices]
        if sample["Actual"].nunique() < 2:
            continue
        values.append(metric(sample["Actual"], sample["Predicted"]))
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def summarize_backtest(folds: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Aggregate all out of period predictions and fold variation."""
    metrics = _metric_row(
        predictions["Actual"],
        predictions["Predicted"].to_numpy(),
        predictions["Probability"].to_numpy(),
    )
    interval_low, interval_high = bootstrap_interval(predictions)
    valid_folds = folds["BalancedAccuracy"].dropna()
    return {
        "Model": predictions["Model"].iloc[0],
        **metrics,
        "BalancedAccuracyMean": float(valid_folds.mean()),
        "BalancedAccuracyStd": float(valid_folds.std(ddof=0)),
        "BalancedAccuracyCILow": interval_low,
        "BalancedAccuracyCIHigh": interval_high,
        "Folds": int(folds["Fold"].nunique()),
        "EvaluationSamples": len(predictions),
    }


def run_model_comparison(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare baseline, linear, and tree models on identical windows."""
    summaries = []
    all_folds = []
    all_predictions = []
    for model_name in (
        "historical_rate",
        "time_only_logistic",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ):
        features = TIME_ONLY if model_name == "time_only_logistic" else DEPLOYMENT_FEATURES
        folds, predictions = backtest_model(data, model_name, features)
        summaries.append(summarize_backtest(folds, predictions))
        all_folds.append(folds)
        all_predictions.append(predictions)
    comparison = pd.DataFrame(summaries).sort_values("BalancedAccuracy", ascending=False)
    return comparison.reset_index(drop=True), pd.concat(all_folds), pd.concat(all_predictions)


def run_ablation_study(data: pd.DataFrame) -> pd.DataFrame:
    """Measure how feature groups affect temporal logistic performance."""
    rows = []
    for label, features in ABLATION_FEATURES.items():
        folds, predictions = backtest_model(data, "logistic_regression", features)
        summary = summarize_backtest(folds, predictions)
        summary["FeatureSet"] = label
        rows.append(summary)
    return (
        pd.DataFrame(rows).sort_values("BalancedAccuracy", ascending=False).reset_index(drop=True)
    )


def run_calibration_study(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw and sigmoid calibrated logistic probabilities."""
    summaries = []
    predictions = []
    for model_name in ("logistic_regression", "calibrated_logistic"):
        folds, model_predictions = backtest_model(data, model_name, DEPLOYMENT_FEATURES)
        summaries.append(summarize_backtest(folds, model_predictions))
        predictions.append(model_predictions)
    return pd.DataFrame(summaries), pd.concat(predictions).reset_index(drop=True)


def calibration_table(predictions: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Return observed versus predicted rates for each model."""
    rows = []
    for model_name, group in predictions.groupby("Model"):
        observed, predicted = calibration_curve(
            group["Actual"], group["Probability"], n_bins=bins, strategy="quantile"
        )
        for bin_number, (mean_prediction, observed_rate) in enumerate(
            zip(predicted, observed, strict=True), 1
        ):
            rows.append(
                {
                    "Model": model_name,
                    "Bin": bin_number,
                    "MeanPredicted": float(mean_prediction),
                    "ObservedRate": float(observed_rate),
                }
            )
    return pd.DataFrame(rows)


def run_experiments(data: pd.DataFrame) -> ExperimentResults:
    """Run the complete reproducible experiment suite."""
    comparison, folds, predictions = run_model_comparison(data)
    ablations = run_ablation_study(data)
    calibration, calibration_predictions = run_calibration_study(data)
    return ExperimentResults(
        comparison=comparison,
        folds=folds,
        predictions=predictions,
        ablations=ablations,
        calibration=calibration,
        calibration_predictions=calibration_predictions,
    )


def error_analysis(data: pd.DataFrame, predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Join mistakes to mission context for qualitative review."""
    selected = predictions.loc[predictions["Model"] == model]
    errors = selected.loc[selected["Actual"] != selected["Predicted"]]
    columns = [
        "FlightNumber",
        "Date",
        "PayloadMass",
        "Orbit",
        "LaunchSite",
        "Flights",
        "Reused",
        "Actual",
        "Predicted",
        "Probability",
    ]
    return errors.merge(data, on="FlightNumber", how="left", suffixes=("", "_Source"))[columns]


def feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """Compute raw feature permutation importance on a two class temporal window."""
    ordered = data.sort_values("FlightNumber").reset_index(drop=True)
    train = ordered.iloc[:70]
    test = ordered.iloc[70:80]
    model = build_model("logistic_regression", DEPLOYMENT_FEATURES)
    model.fit(train[DEPLOYMENT_FEATURES.columns], train["Class"])
    result = permutation_importance(
        model,
        test[DEPLOYMENT_FEATURES.columns],
        test["Class"],
        scoring="balanced_accuracy",
        n_repeats=40,
        random_state=42,
    )
    return pd.DataFrame(
        {
            "Feature": DEPLOYMENT_FEATURES.columns,
            "Importance": result.importances_mean,
            "ImportanceStd": result.importances_std,
        }
    ).sort_values("Importance", ascending=False)


def logistic_coefficients(data: pd.DataFrame) -> pd.DataFrame:
    """Return fitted coefficient direction and magnitude for interpretation."""
    model = build_model("logistic_regression", DEPLOYMENT_FEATURES)
    model.fit(data[DEPLOYMENT_FEATURES.columns], data["Class"])
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    coefficients = classifier.coef_[0]
    return pd.DataFrame(
        {
            "Feature": names,
            "Coefficient": coefficients,
            "AbsoluteCoefficient": np.abs(coefficients),
        }
    ).sort_values("AbsoluteCoefficient", ascending=False)
