"""Interactive historical analysis, inference, and model diagnostics."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from spacex_falcon.analytics import modeling_frame, site_success, yearly_success
from spacex_falcon.data import load_launch_data, validate_launch_data
from spacex_falcon.evaluation import (
    ExperimentResults,
    calibration_table,
    error_analysis,
    feature_importance,
    run_experiments,
)
from spacex_falcon.model import prediction_record, train_temporal_model


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _comparison_table(comparison: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in comparison.iterrows():
        rows.append(
            {
                "Model": row["Model"],
                "Balanced accuracy": _percent(row["BalancedAccuracy"]),
                "95% interval": (
                    f"{_percent(row['BalancedAccuracyCILow'])} to "
                    f"{_percent(row['BalancedAccuracyCIHigh'])}"
                ),
                "ROC AUC": _percent(row["ROCAUC"]),
                "Brier": f"{row['Brier']:.3f}",
                "F1": _percent(row["F1"]),
            }
        )
    return rows


def _html_table(records: list[dict], table_id: str, limit: int | None = None):
    """Render a dependency free responsive HTML table."""
    visible = records[:limit] if limit else records
    columns = list(visible[0]) if visible else []
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(column) for column in columns])),
                html.Tbody(
                    [
                        html.Tr([html.Td(record.get(column, "")) for column in columns])
                        for record in visible
                    ]
                ),
            ],
            id=table_id,
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        style={"overflowX": "auto"},
    )


def _diagnostic_figures(
    data: pd.DataFrame, experiments: ExperimentResults
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, pd.DataFrame]:
    comparison = experiments.comparison.copy()
    comparison["ErrorPlus"] = comparison["BalancedAccuracyCIHigh"] - comparison["BalancedAccuracy"]
    comparison["ErrorMinus"] = comparison["BalancedAccuracy"] - comparison["BalancedAccuracyCILow"]
    model_figure = go.Figure(
        go.Bar(
            x=comparison["Model"],
            y=comparison["BalancedAccuracy"],
            error_y={
                "type": "data",
                "array": comparison["ErrorPlus"],
                "arrayminus": comparison["ErrorMinus"],
            },
        )
    )
    model_figure.update_layout(
        title="Out of period balanced accuracy with bootstrap intervals",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        xaxis_title=None,
        yaxis_title="Balanced accuracy",
    )

    fold_figure = px.line(
        experiments.folds,
        x="TestEnd",
        y="BalancedAccuracy",
        color="Model",
        markers=True,
        title="Performance through expanding temporal windows",
        labels={"TestEnd": "Final flight in test window", "BalancedAccuracy": "Balanced accuracy"},
    )
    fold_figure.update_yaxes(range=[0, 1], tickformat=".0%")

    calibration = calibration_table(experiments.calibration_predictions)
    calibration_figure = px.line(
        calibration,
        x="MeanPredicted",
        y="ObservedRate",
        color="Model",
        markers=True,
        title="Probability calibration across temporal predictions",
        labels={"MeanPredicted": "Mean predicted probability", "ObservedRate": "Observed rate"},
    )
    calibration_figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line_dash="dash")
    )
    calibration_figure.update_xaxes(range=[0, 1], tickformat=".0%")
    calibration_figure.update_yaxes(range=[0, 1], tickformat=".0%")

    ablation_figure = px.bar(
        experiments.ablations,
        x="FeatureSet",
        y="BalancedAccuracy",
        title="Feature ablation under temporal validation",
        labels={"FeatureSet": "Feature set", "BalancedAccuracy": "Balanced accuracy"},
    )
    ablation_figure.update_yaxes(range=[0, 1], tickformat=".0%")

    importance = feature_importance(data)
    importance_figure = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="ImportanceStd",
        title="Permutation importance on flights 71 to 80",
    )
    errors = error_analysis(data, experiments.predictions, "Logistic regression")
    errors = errors.copy()
    errors["Date"] = errors["Date"].astype(str)
    errors["Probability"] = errors["Probability"].round(3)
    return model_figure, fold_figure, calibration_figure, ablation_figure, importance_figure, errors


def create_app(
    data: pd.DataFrame | None = None,
    experiment_results: ExperimentResults | None = None,
) -> Dash:
    """Create a fully initialized analytical Dash application."""
    validated = load_launch_data() if data is None else validate_launch_data(data)
    launch_data = modeling_frame(validated)
    experiments = experiment_results or run_experiments(launch_data)
    model, report = train_temporal_model(launch_data)

    sites = sorted(launch_data["LaunchSite"].dropna().unique())
    orbits = sorted(launch_data["Orbit"].dropna().unique())
    min_payload = int(launch_data["PayloadMass"].min())
    max_payload = int(launch_data["PayloadMass"].max())
    next_flight = int(launch_data["FlightNumber"].max()) + 1
    comparison_rows = _comparison_table(experiments.comparison)
    best = experiments.comparison.iloc[0]
    historical_rate = experiments.comparison.loc[
        experiments.comparison["Model"] == "Historical rate baseline"
    ].iloc[0]
    raw_calibration = experiments.calibration.loc[
        experiments.calibration["Model"] == "Logistic regression"
    ].iloc[0]
    sigmoid_calibration = experiments.calibration.loc[
        experiments.calibration["Model"] == "Calibrated logistic"
    ].iloc[0]
    key_finding_text = (
        f"{best['Model']} achieved {_percent(best['BalancedAccuracy'])} balanced accuracy "
        f"across five expanding windows, compared with "
        f"{_percent(historical_rate['BalancedAccuracy'])} for the historical rate baseline. "
        "The wide bootstrap interval shows that the 90 launch sample cannot support precise claims."
    )
    report_text = (
        f"Temporal ROC AUC {_percent(report.roc_auc)}, Brier score {report.brier:.3f}, and "
        f"balanced accuracy {_percent(report.balanced_accuracy)} with a 95% bootstrap interval "
        f"from {_percent(report.balanced_accuracy_ci_low)} to "
        f"{_percent(report.balanced_accuracy_ci_high)}."
    )
    (
        model_figure,
        fold_figure,
        calibration_figure,
        ablation_figure,
        importance_figure,
        errors,
    ) = _diagnostic_figures(launch_data, experiments)

    annual = yearly_success(validated)
    annual_figure = px.line(
        annual,
        x="LaunchYear",
        y="SuccessRate",
        markers=True,
        title="Landing success increased as the program matured",
        labels={"LaunchYear": "Launch year", "SuccessRate": "Landing success rate"},
    )
    annual_figure.update_yaxes(range=[0, 1], tickformat=".0%")

    site_summary = site_success(validated)
    site_figure = px.bar(
        site_summary,
        x="LaunchSite",
        y="SuccessRate",
        hover_data=["Launches"],
        title="Site outcomes reflect both location and historical era",
        labels={"LaunchSite": "Launch site", "SuccessRate": "Landing success rate"},
    )
    site_figure.update_yaxes(range=[0, 1], tickformat=".0%")

    app = Dash(__name__)
    app.title = "Falcon 9 Temporal Generalization"
    app.layout = html.Main(
        [
            html.H1("Falcon 9 Landing Success: Temporal Generalization"),
            html.P(
                "How well can landing success be predicted from information available before "
                "launch, and how does performance change as the program matures?"
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Overview",
                        children=[
                            html.Section(
                                [
                                    html.H2("Key finding"),
                                    html.P(key_finding_text),
                                ]
                            ),
                            _html_table(comparison_rows, "model-results-table"),
                            html.Section(
                                [
                                    html.H2("Explicit landing probability"),
                                    html.P(report_text),
                                    html.Label("Flight number", htmlFor="prediction-flight"),
                                    dcc.Input(
                                        id="prediction-flight",
                                        type="number",
                                        min=1,
                                        value=next_flight,
                                    ),
                                    html.Label(
                                        "Payload mass in kilograms", htmlFor="prediction-payload"
                                    ),
                                    dcc.Input(
                                        id="prediction-payload",
                                        type="number",
                                        min=0,
                                        value=int(launch_data["PayloadMass"].median()),
                                    ),
                                    html.Label(
                                        "Booster prior flights", htmlFor="prediction-prior-flights"
                                    ),
                                    dcc.Input(
                                        id="prediction-prior-flights",
                                        type="number",
                                        min=1,
                                        value=1,
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
                                        options=[
                                            {"label": orbit, "value": orbit} for orbit in orbits
                                        ],
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
                                    html.Button(
                                        "Estimate probability",
                                        id="prediction-button",
                                        n_clicks=0,
                                    ),
                                    html.Output(id="prediction-result"),
                                    html.P(
                                        "Every model input is shown above. The probability is an "
                                        "educational estimate and is not operational guidance."
                                    ),
                                ]
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Historical Analysis",
                        children=[
                            dcc.Graph(figure=annual_figure),
                            dcc.Graph(figure=site_figure),
                            html.Label("Filter launch site", htmlFor="site-dropdown"),
                            dcc.Dropdown(
                                id="site-dropdown",
                                options=[{"label": "All sites", "value": "ALL"}]
                                + [{"label": site, "value": site} for site in sites],
                                value="ALL",
                                clearable=False,
                            ),
                            html.Label("Filter payload range", htmlFor="payload-slider"),
                            dcc.RangeSlider(
                                id="payload-slider",
                                min=min_payload,
                                max=max_payload,
                                step=250,
                                value=[min_payload, max_payload],
                            ),
                            dcc.Graph(id="success-payload-scatter-chart"),
                        ],
                    ),
                    dcc.Tab(
                        label="Model Diagnostics",
                        children=[
                            dcc.Graph(figure=model_figure),
                            dcc.Graph(figure=fold_figure),
                            dcc.Graph(figure=calibration_figure),
                            html.P(
                                f"Sigmoid calibration changed Brier score from "
                                f"{raw_calibration['Brier']:.3f} to "
                                f"{sigmoid_calibration['Brier']:.3f}. The uncalibrated logistic "
                                "model is retained because calibration did not improve squared "
                                "probability error on the temporal predictions."
                            ),
                            dcc.Graph(figure=ablation_figure),
                            dcc.Graph(figure=importance_figure),
                            html.H2("Logistic regression errors"),
                            _html_table(
                                errors.to_dict("records"), "prediction-errors-table", limit=15
                            ),
                        ],
                    ),
                ]
            ),
        ],
        style={"maxWidth": "1150px", "margin": "0 auto", "padding": "24px"},
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
            symbol="Reused",
            hover_data=["FlightNumber", "LaunchSite", "Date", "Flights"],
            labels={"PayloadMass": "Payload mass in kilograms", "Class": "Landing outcome"},
            title="Payload and recorded landing outcome",
        )

    @app.callback(
        Output("prediction-result", "children"),
        Input("prediction-button", "n_clicks"),
        State("prediction-flight", "value"),
        State("prediction-payload", "value"),
        State("prediction-prior-flights", "value"),
        State("prediction-site", "value"),
        State("prediction-orbit", "value"),
        State("prediction-options", "value"),
        prevent_initial_call=True,
    )
    def predict_landing(
        _clicks: int,
        flight_number: float,
        payload_mass: float,
        booster_prior_flights: float,
        launch_site: str,
        orbit: str,
        options: list[str] | None,
    ):
        if any(value is None for value in (flight_number, payload_mass, booster_prior_flights)):
            return "Enter a flight number, payload mass, and booster flight count."
        selected = set(options or [])
        row = prediction_record(
            flight_number=flight_number,
            payload_mass=payload_mass,
            booster_prior_flights=booster_prior_flights,
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
