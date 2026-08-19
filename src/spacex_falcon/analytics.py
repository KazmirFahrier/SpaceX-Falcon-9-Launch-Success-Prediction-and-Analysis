"""DuckDB backed analytical queries."""

from __future__ import annotations

import duckdb
import pandas as pd


def query_launches(data: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Run a read only SQL query against validated launch records."""
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("launch_records", data)
        return connection.execute(sql).fetchdf()
    finally:
        connection.close()


def modeling_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Build the ordered analytical frame used by experiments."""
    return query_launches(
        data,
        """
        SELECT
            *,
            EXTRACT(YEAR FROM TRY_CAST(Date AS DATE))::INTEGER AS LaunchYear,
            CASE
                WHEN FlightNumber <= 30 THEN 'Early'
                WHEN FlightNumber <= 60 THEN 'Middle'
                ELSE 'Recent'
            END AS HistoricalPeriod
        FROM launch_records
        ORDER BY FlightNumber
        """,
    )


def yearly_success(data: pd.DataFrame) -> pd.DataFrame:
    """Return annual launch volume and landing success rate."""
    return query_launches(
        data,
        """
        SELECT
            EXTRACT(YEAR FROM TRY_CAST(Date AS DATE))::INTEGER AS LaunchYear,
            COUNT(*)::INTEGER AS Launches,
            AVG(Class) AS SuccessRate
        FROM launch_records
        GROUP BY LaunchYear
        ORDER BY LaunchYear
        """,
    )


def site_success(data: pd.DataFrame) -> pd.DataFrame:
    """Return launch site volume and success rate."""
    return query_launches(
        data,
        """
        SELECT
            LaunchSite,
            COUNT(*)::INTEGER AS Launches,
            AVG(Class) AS SuccessRate
        FROM launch_records
        GROUP BY LaunchSite
        ORDER BY LaunchSite
        """,
    )
