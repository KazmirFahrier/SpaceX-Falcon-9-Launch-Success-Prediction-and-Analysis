from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def launch_data() -> pd.DataFrame:
    rows = []
    for index in range(90):
        rows.append(
            {
                "FlightNumber": index + 1,
                "Date": f"2020-01-{(index % 28) + 1:02d}",
                "BoosterVersion": "Falcon 9",
                "PayloadMass": 500 + index * 125,
                "Orbit": "LEO" if index % 2 else "GTO",
                "LaunchSite": "Site A" if index % 3 else "Site B",
                "Outcome": "True ASDS" if index % 2 else "False ASDS",
                "Flights": 1 + index % 5,
                "GridFins": bool(index % 2),
                "Reused": bool(index % 3),
                "Legs": bool(index % 2),
                "LandingPad": "Pad A",
                "Block": 1 + index % 5,
                "ReusedCount": index % 4,
                "Serial": f"B{index:04d}",
                "Longitude": -80.5,
                "Latitude": 28.5,
                "Class": index % 2,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def experiment_results(launch_data):
    from spacex_falcon.evaluation import run_experiments

    return run_experiments(launch_data)


@pytest.fixture(scope="session")
def dashboard_app(launch_data, experiment_results):
    from spacex_falcon.dashboard import create_app

    return create_app(launch_data, experiment_results)
