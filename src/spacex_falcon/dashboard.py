"""Interactive analysis and inference dashboard."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html

from spacex_falcon.data import load_launch_data
from spacex_falcon.model import prediction_record, train_temporal_model


def create_app(data: pd.DataFrame | None = None) -> Dash:
    """Create a fully initialized Dash application."""
    launch_data = load_launch_data() if data is None else data.copy()
    model, report = train_temporal_model(launch_data)

    sites = sorted(launch_data["LaunchSite"].dropna().unique())
    orbits = sorted(launch_data["Orbit"].dropna().unique())
    min_payload = int(launch_data["PayloadMass"].min())
    max_payload = int(launch_data["PayloadMass"].max())
    next_flight = int(launch_data["FlightNumber"].max()) + 1
    roc_auc_text = f"{report.roc_auc:.1%}" if report.roc_auc is not None else "not available"

    app = Dash(__name__)
    app.title = "Falcon 9 Landing Analysis"
    app.layout = html.Main(
        [
            html.H1("Falcon 9 First Stage Landing Analysis"),
            html.P(
                "Explore historical outcomes and estimate a landing probability using a "
                "logistic regression pipeline. Predictions are educational, not operational."
            ),
            html.Section(
                [
                    html.H2("Evaluation on later launches"),
                    html.P(
                        f"Accuracy {report.accuracy:.1%} | Balanced accuracy "
                        f"{report.balanced_accuracy:.1%} | Baseline accuracy "
                        f"{report.baseline_accuracy:.1%} | ROC AUC {roc_auc_text} | "
                        f"F1 {report.f1:.1%} | "
                        f"Holdout records {report.test_samples}"
                    ),
                ],
                id="model-summary",
            ),
            html.Section(
                [
                    html.H2("Historical outcomes"),
                    dcc.Dropdown(
                        id="site-dropdown",
                        options=[{"label": "All sites", "value": "ALL"}]
                        + [{"label": site, "value": site} for site in sites],
                        value="ALL",
                        clearable=False,
                    ),
                    dcc.RangeSlider(
                        id="payload-slider",
                        min=min_payload,
                        max=max_payload,
                        step=250,
                        value=[min_payload, max_payload],
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dcc.Graph(id="success-pie-chart"),
                    dcc.Graph(id="success-payload-scatter-chart"),
                ]
            ),
            html.Section(
                [
                    html.H2("Landing probability"),
                    html.Label("Flight number", htmlFor="prediction-flight"),
                    dcc.Input(
                        id="prediction-flight",
                        type="number",
                        min=1,
                        value=next_flight,
                    ),
                    html.Label("Payload mass in kilograms", htmlFor="prediction-payload"),
                    dcc.Input(
                        id="prediction-payload",
                        type="number",
                        min=0,
                        value=int(launch_data["PayloadMass"].median()),
                    ),
                    html.Label("Launch site", htmlFor="prediction-site"),
                    dcc.Dropdown(
                        id="prediction-site",
                        options=[{"label": site, "value": site} for site in sites],
                        value=sites[0],
                        clearable=False,
                    ),
                    html.Label("Orbit", htmlFor="prediction-orbit"),
                    dcc.Dropdown(
                        id="prediction-orbit",
                        options=[{"label": orbit, "value": orbit} for orbit in orbits],
                        value=orbits[0],
                        clearable=False,
                    ),
                    dcc.Checklist(
                        id="prediction-options",
                        options=[
                            {"label": "Grid fins", "value": "grid_fins"},
                            {"label": "Reused booster", "value": "reused"},
                            {"label": "Landing legs", "value": "legs"},
                        ],
                        value=["grid_fins", "reused", "legs"],
                    ),
                    html.Button("Estimate probability", id="prediction-button", n_clicks=0),
                    html.Output(id="prediction-result"),
                ]
            ),
        ],
        style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px"},
    )

    @app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
    def update_pie(selected_site: str):
        if selected_site == "ALL":
            summary = launch_data.groupby("LaunchSite", as_index=False)["Class"].mean()
            return px.bar(
                summary,
                x="LaunchSite",
                y="Class",
                range_y=[0, 1],
                labels={"Class": "Landing success rate", "LaunchSite": "Launch site"},
                title="Landing success rate by launch site",
            )
        site_data = launch_data.loc[launch_data["LaunchSite"] == selected_site]
        counts = site_data["Class"].map({0: "Failure", 1: "Success"}).value_counts()
        return px.pie(
            values=counts.values,
            names=counts.index,
            title=f"Launch outcomes for {selected_site}",
        )

    @app.callback(
        Output("success-payload-scatter-chart", "figure"),
        Input("site-dropdown", "value"),
        Input("payload-slider", "value"),
    )
    def update_scatter(selected_site: str, payload_range: list[int]):
        low, high = payload_range
        filtered = launch_data.loc[launch_data["PayloadMass"].between(low, high)]
        if selected_site != "ALL":
            filtered = filtered.loc[filtered["LaunchSite"] == selected_site]
        return px.scatter(
            filtered,
            x="PayloadMass",
            y="Class",
            color="Orbit",
            hover_data=["FlightNumber", "LaunchSite", "Date"],
            labels={"PayloadMass": "Payload mass in kilograms", "Class": "Landing outcome"},
            title="Payload and recorded landing outcome",
        )

    @app.callback(
        Output("prediction-result", "children"),
        Input("prediction-button", "n_clicks"),
        State("prediction-flight", "value"),
        State("prediction-payload", "value"),
        State("prediction-site", "value"),
        State("prediction-orbit", "value"),
        State("prediction-options", "value"),
        prevent_initial_call=True,
    )
    def predict_landing(
        _clicks: int,
        flight_number: float,
        payload_mass: float,
        launch_site: str,
        orbit: str,
        options: list[str] | None,
    ):
        if flight_number is None or payload_mass is None:
            return "Enter a flight number and payload mass."
        selected = set(options or [])
        row = prediction_record(
            launch_data,
            flight_number=flight_number,
            payload_mass=payload_mass,
            launch_site=launch_site,
            orbit=orbit,
            grid_fins="grid_fins" in selected,
            reused="reused" in selected,
            legs="legs" in selected,
        )
        probability = float(model.predict_proba(row)[0, 1])
        return f"Estimated landing probability: {probability:.1%}"

    return app


def main() -> None:
    """Run the development server."""
    app = create_app()
    port = int(os.getenv("PORT", "8051"))
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
