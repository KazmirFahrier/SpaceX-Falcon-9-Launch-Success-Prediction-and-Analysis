from __future__ import annotations


def _component_ids(component):
    found = set()
    component_id = getattr(component, "id", None)
    if component_id:
        found.add(component_id)
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or getattr(child, "id", None):
            found.update(_component_ids(child))
    return found


def test_dashboard_builds_with_analysis_and_prediction_controls(dashboard_app):
    ids = _component_ids(dashboard_app.layout)

    assert dashboard_app.server is not None
    assert "success-payload-scatter-chart" in ids
    assert "model-results-table" in ids
    assert "prediction-errors-table" in ids
    assert "prediction-prior-flights" in ids
    assert "prediction-button" in ids
    assert "prediction-result" in ids


def test_prediction_callback_returns_probability(dashboard_app):
    client = dashboard_app.server.test_client()
    response = client.post(
        "/_dash-update-component",
        json={
            "output": "prediction-result.children",
            "outputs": {"id": "prediction-result", "property": "children"},
            "inputs": [{"id": "prediction-button", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": "prediction-flight", "property": "value", "value": 41},
                {"id": "prediction-payload", "property": "value", "value": 2_000},
                {"id": "prediction-prior-flights", "property": "value", "value": 2},
                {"id": "prediction-site", "property": "value", "value": "Site A"},
                {"id": "prediction-orbit", "property": "value", "value": "LEO"},
                {
                    "id": "prediction-options",
                    "property": "value",
                    "value": ["grid_fins", "reused", "legs"],
                },
            ],
            "changedPropIds": ["prediction-button.n_clicks"],
        },
    )

    assert response.status_code == 200
    assert "Estimated landing probability" in response.get_data(as_text=True)
