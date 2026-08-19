from __future__ import annotations

import pandas as pd
import pytest

from spacex_falcon.data import DataValidationError, load_launch_data, validate_launch_data


def test_validation_normalizes_and_orders_rows(launch_data):
    shuffled = launch_data.sample(frac=1, random_state=7)

    result = validate_launch_data(shuffled)

    assert result["FlightNumber"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert set(result["Class"].unique()) == {0, 1}


def test_validation_rejects_missing_columns(launch_data):
    with pytest.raises(DataValidationError, match="PayloadMass"):
        validate_launch_data(launch_data.drop(columns="PayloadMass"))


def test_loads_local_csv(tmp_path, launch_data):
    path = tmp_path / "launches.csv"
    launch_data.to_csv(path, index=False)

    result = load_launch_data(path)

    assert len(result) == len(launch_data)
