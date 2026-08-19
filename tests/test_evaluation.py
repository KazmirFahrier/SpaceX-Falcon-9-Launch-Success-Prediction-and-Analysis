from __future__ import annotations

from spacex_falcon.evaluation import (
    bootstrap_interval,
    calibration_table,
    expanding_windows,
    feature_importance,
    logistic_coefficients,
)


def test_expanding_windows_match_documented_backtest():
    assert expanding_windows(90) == [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]


def test_experiment_suite_contains_models_and_ablations(experiment_results):
    models = set(experiment_results.comparison["Model"])
    assert models == {
        "Historical rate baseline",
        "Time only logistic",
        "Logistic regression",
        "Random forest",
        "Gradient boosting",
    }
    assert len(experiment_results.predictions) == 250
    assert len(experiment_results.ablations) == 5
    assert set(experiment_results.calibration["Model"]) == {
        "Logistic regression",
        "Calibrated logistic",
    }


def test_uncertainty_and_calibration_tables(experiment_results):
    logistic = experiment_results.predictions.loc[
        experiment_results.predictions["Model"] == "Logistic regression"
    ]
    lower, upper = bootstrap_interval(logistic, iterations=100)
    calibration = calibration_table(experiment_results.calibration_predictions)

    assert 0 <= lower <= upper <= 1
    assert set(calibration.columns) == {"Model", "Bin", "MeanPredicted", "ObservedRate"}


def test_interpretation_outputs_match_explicit_features(launch_data):
    coefficients = logistic_coefficients(launch_data)
    importance = feature_importance(launch_data)

    assert {"Feature", "Coefficient", "AbsoluteCoefficient"} == set(coefficients.columns)
    assert len(importance) == 8
    assert importance["Feature"].is_unique
