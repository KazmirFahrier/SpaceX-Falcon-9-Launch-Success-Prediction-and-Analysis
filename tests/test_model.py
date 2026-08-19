from __future__ import annotations

import pytest

from spacex_falcon.model import MODEL_FEATURES, prediction_record, train_temporal_model


def test_temporal_evaluation_and_final_model(launch_data):
    model, report = train_temporal_model(launch_data)

    assert report.train_samples == 32
    assert report.test_samples == 8
    assert report.train_max_flight < report.test_min_flight
    assert 0 <= report.baseline_accuracy <= 1
    assert 0 <= report.accuracy <= 1
    assert hasattr(model, "predict_proba")


def test_prediction_record_is_accepted_by_model(launch_data):
    model, _report = train_temporal_model(launch_data)
    row = prediction_record(
        launch_data,
        flight_number=41,
        payload_mass=2_000,
        orbit="LEO",
        launch_site="Site A",
        grid_fins=True,
        reused=True,
        legs=True,
    )

    probability = model.predict_proba(row)[0, 1]

    assert list(row.columns) == MODEL_FEATURES
    assert probability == pytest.approx(float(probability))
    assert 0 <= probability <= 1
