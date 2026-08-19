"""Dataset retrieval and validation."""

from __future__ import annotations

import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
)
DATA_SHA256 = "176e74024eb330504fc392a2784dc0b25a605f5748d2f7df6d490f073c08edeb"

REQUIRED_COLUMNS = {
    "FlightNumber",
    "Date",
    "BoosterVersion",
    "PayloadMass",
    "Orbit",
    "LaunchSite",
    "Flights",
    "GridFins",
    "Reused",
    "Legs",
    "Block",
    "ReusedCount",
    "Class",
}


class DataValidationError(ValueError):
    """Raised when a launch dataset does not match the expected schema."""


def validate_launch_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a launch dataset without mutating the input."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise DataValidationError("Dataset contains no launch records")

    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data["Class"] = pd.to_numeric(data["Class"], errors="raise").astype(int)
    if not set(data["Class"].unique()).issubset({0, 1}):
        raise DataValidationError("Class must contain only 0 and 1")

    numeric_columns = ["FlightNumber", "PayloadMass", "Flights", "Block", "ReusedCount"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["FlightNumber", "PayloadMass"]].isna().any().any():
        raise DataValidationError("FlightNumber and PayloadMass must be numeric and nonempty")

    return data.sort_values(["FlightNumber", "Date"]).reset_index(drop=True)


def _default_cache_path() -> Path:
    configured = os.getenv("SPACEX_DATA_CACHE")
    return Path(configured) if configured else Path.cwd() / "data" / "dataset_part_2.csv"


def load_launch_data(source: str | Path | None = None) -> pd.DataFrame:
    """Load launch records from a path, URL, cache, or the documented source."""
    if source is not None:
        source_text = str(source)
        if source_text.startswith(("http://", "https://")):
            return validate_launch_data(pd.read_csv(source_text))
        return validate_launch_data(pd.read_csv(Path(source)))

    cache_path = _default_cache_path()
    if cache_path.exists():
        return validate_launch_data(pd.read_csv(cache_path))

    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        digest = sha256(response.content).hexdigest()
        if digest != DATA_SHA256:
            raise RuntimeError(
                "The remote dataset has changed. Review it before updating DATA_SHA256."
            )
        data = validate_launch_data(pd.read_csv(BytesIO(response.content)))
    except (requests.RequestException, OSError, ValueError) as exc:
        raise RuntimeError(
            "Could not retrieve the launch dataset. Set SPACEX_DATA_CACHE to a valid CSV path."
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return data
