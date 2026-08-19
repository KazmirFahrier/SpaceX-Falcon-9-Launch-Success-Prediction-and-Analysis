from __future__ import annotations

from spacex_falcon.analytics import modeling_frame, site_success, yearly_success


def test_duckdb_analytics_build_ordered_modeling_frame(launch_data):
    frame = modeling_frame(launch_data)

    assert frame["FlightNumber"].is_monotonic_increasing
    assert {"LaunchYear", "HistoricalPeriod"}.issubset(frame.columns)


def test_duckdb_summaries_reconcile_to_source(launch_data):
    sites = site_success(launch_data)
    years = yearly_success(launch_data)

    assert sites["Launches"].sum() == len(launch_data)
    assert years["Launches"].sum() == len(launch_data)
