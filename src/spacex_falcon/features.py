"""Feature definitions for experiments and interactive inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Typed columns used by one model experiment."""

    name: str
    numeric: tuple[str, ...] = ()
    boolean: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.boolean, *self.categorical]


TIME_ONLY = FeatureSpec(name="Time only", numeric=("FlightNumber",))

MISSION_ONLY = FeatureSpec(
    name="Mission characteristics",
    numeric=("PayloadMass",),
    categorical=("Orbit", "LaunchSite"),
)

MISSION_AND_BOOSTER = FeatureSpec(
    name="Mission and booster",
    numeric=("PayloadMass", "Flights"),
    boolean=("GridFins", "Reused", "Legs"),
    categorical=("Orbit", "LaunchSite"),
)

DEPLOYMENT_FEATURES = FeatureSpec(
    name="Explicit inference features",
    numeric=("FlightNumber", "PayloadMass", "Flights"),
    boolean=("GridFins", "Reused", "Legs"),
    categorical=("Orbit", "LaunchSite"),
)

ALL_FEATURES = FeatureSpec(
    name="All historical features",
    numeric=("FlightNumber", "PayloadMass", "Flights", "Block", "ReusedCount"),
    boolean=("GridFins", "Reused", "Legs"),
    categorical=("Orbit", "LaunchSite", "BoosterVersion"),
)

ABLATION_FEATURES = {
    "Time only": TIME_ONLY,
    "Mission only": MISSION_ONLY,
    "Mission and booster": MISSION_AND_BOOSTER,
    "All explicit inputs": DEPLOYMENT_FEATURES,
    "All historical features": ALL_FEATURES,
}
