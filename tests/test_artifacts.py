from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_authored_notebooks_are_clean_case_studies():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "01_exploratory_analysis.ipynb",
        "02_temporal_modeling.ipynb",
        "03_model_interpretation.ipynb",
    ]

    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "".join(
            "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else ""
            for cell in notebook["cells"]
        ).lower()
        assert "skills network" not in source
        assert "estimated time needed" not in source
        assert "task 1" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                code = "".join(cell["source"])
                compile(code, str(path), "exec")


def test_generated_report_contains_expected_experiments():
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    assert len(comparison) == 5
    assert comparison.iloc[0]["Model"] == "Random forest"
    random_forest = comparison.iloc[0]
    baseline = comparison.loc[comparison["Model"] == "Historical rate baseline"].iloc[0]
    assert random_forest["BalancedAccuracy"] == pytest.approx(0.689, abs=0.001)
    assert random_forest["BalancedAccuracyCILow"] == pytest.approx(0.537, abs=0.001)
    assert random_forest["BalancedAccuracyCIHigh"] == pytest.approx(0.852, abs=0.001)
    assert baseline["BalancedAccuracy"] == pytest.approx(0.5)

    folds = pd.read_csv(ROOT / "reports" / "fold_metrics.csv")
    assert len(folds) == 25
    assert folds.groupby("Model")["Fold"].nunique().eq(5).all()

    assert (ROOT / "reports" / "model_card.md").exists()
    assert (ROOT / "reports" / "prediction_errors.csv").exists()
    for figure in ("model_comparison.html", "calibration.html", "ablation.html"):
        path = ROOT / "reports" / "figures" / figure
        assert path.exists()
        assert path.stat().st_size > 5_000


def test_course_notebooks_are_archived_outside_root():
    archived = list((ROOT / "archive" / "ibm_coursework").glob("*.ipynb"))
    assert len(archived) == 8
    assert not list(ROOT.glob("*.ipynb"))
