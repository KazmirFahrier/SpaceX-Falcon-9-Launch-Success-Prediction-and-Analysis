"""Generate reproducible experiment tables, diagnostics, and the model card."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from spacex_falcon.analytics import modeling_frame
from spacex_falcon.data import load_launch_data
from spacex_falcon.evaluation import (
    calibration_table,
    error_analysis,
    feature_importance,
    logistic_coefficients,
    run_experiments,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional dependencies."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in data[columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def rounded_results(data: pd.DataFrame) -> pd.DataFrame:
    """Format the main metrics for human review."""
    selected = data[
        [
            "Model",
            "BalancedAccuracy",
            "BalancedAccuracyCILow",
            "BalancedAccuracyCIHigh",
            "ROCAUC",
            "Brier",
            "F1",
        ]
    ].copy()
    numeric = selected.columns.difference(["Model"])
    selected[numeric] = selected[numeric].round(3)
    return selected


def model_card(comparison: pd.DataFrame, calibration: pd.DataFrame) -> str:
    """Create an evidence based model card from generated metrics."""
    best = comparison.iloc[0]
    baseline = comparison.loc[comparison["Model"] == "Historical rate baseline"].iloc[0]
    raw = calibration.loc[calibration["Model"] == "Logistic regression"].iloc[0]
    calibrated = calibration.loc[calibration["Model"] == "Calibrated logistic"].iloc[0]
    table = markdown_table(
        rounded_results(comparison),
        [
            "Model",
            "BalancedAccuracy",
            "BalancedAccuracyCILow",
            "BalancedAccuracyCIHigh",
            "ROCAUC",
            "Brier",
            "F1",
        ],
    )
    return f"""# Model card

## Intended use

This study asks how well Falcon 9 first stage landing success can be predicted from historical
information available before launch. It is an educational retrospective analysis, not an
operational SpaceX system.

## Evaluation design

All results use five expanding windows. The first model trains on flights 1 through 40 and tests
on 41 through 50. The training history then expands by ten flights until the final test window,
flights 81 through 90. Preprocessing is fitted independently inside every training window.

## Results

{table}

The strongest aggregate result is **{best["Model"]}** with balanced accuracy
**{best["BalancedAccuracy"]:.3f}** versus **{baseline["BalancedAccuracy"]:.3f}** for the historical
rate baseline. Its 95% bootstrap interval is **[{best["BalancedAccuracyCILow"]:.3f},
{best["BalancedAccuracyCIHigh"]:.3f}]**, so the small sample does not support a precise ranking.

## Calibration

Raw logistic Brier score is **{raw["Brier"]:.3f}**. Sigmoid calibration produced
**{calibrated["Brier"]:.3f}** and therefore did not improve squared probability error in this
backtest. The interactive application retains the raw logistic probability and displays this
limitation.

## Important limitations

1. The dataset contains only 90 historical launches.
2. Landing success increases sharply over time, creating distribution shift.
3. The final ten launches contain no failures, so fold specific discrimination metrics are not
   identifiable in that window.
4. Flight number represents historical era and should not be interpreted causally.
5. Launch site, booster configuration, and historical period are confounded.
6. Bootstrap intervals quantify sampling variation but not uncertainty from dataset construction.

## Reproduction

Run `python scripts/generate_report.py` after installing the project. Generated CSV files and
interactive figures are written beneath `reports/`.
"""


def main() -> None:
    """Run every experiment and write reviewable artifacts."""
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    data = modeling_frame(load_launch_data())
    results = run_experiments(data)
    importance = feature_importance(data)
    coefficients = logistic_coefficients(data)
    errors = error_analysis(data, results.predictions, "Logistic regression")
    calibration_points = calibration_table(results.calibration_predictions)

    csv_options = {"index": False, "float_format": "%.12g"}
    results.comparison.to_csv(REPORTS / "model_comparison.csv", **csv_options)
    results.folds.to_csv(REPORTS / "fold_metrics.csv", **csv_options)
    results.ablations.to_csv(REPORTS / "ablation_results.csv", **csv_options)
    results.calibration.to_csv(REPORTS / "calibration_results.csv", **csv_options)
    coefficients.to_csv(REPORTS / "logistic_coefficients.csv", **csv_options)
    importance.to_csv(REPORTS / "permutation_importance.csv", **csv_options)
    errors.to_csv(REPORTS / "prediction_errors.csv", **csv_options)
    (REPORTS / "model_card.md").write_text(
        model_card(results.comparison, results.calibration), encoding="utf-8"
    )

    comparison_figure = px.bar(
        results.comparison,
        x="Model",
        y="BalancedAccuracy",
        title="Expanding window model comparison",
    )
    comparison_figure.write_html(
        FIGURES / "model_comparison.html",
        include_plotlyjs="cdn",
        div_id="model-comparison-figure",
    )
    calibration_figure = px.line(
        calibration_points,
        x="MeanPredicted",
        y="ObservedRate",
        color="Model",
        markers=True,
        title="Temporal calibration",
    )
    calibration_figure.write_html(
        FIGURES / "calibration.html",
        include_plotlyjs="cdn",
        div_id="calibration-figure",
    )
    ablation_figure = px.bar(
        results.ablations,
        x="FeatureSet",
        y="BalancedAccuracy",
        title="Feature ablation",
    )
    ablation_figure.write_html(
        FIGURES / "ablation.html",
        include_plotlyjs="cdn",
        div_id="ablation-figure",
    )


if __name__ == "__main__":
    main()
