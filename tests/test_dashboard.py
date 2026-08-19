from __future__ import annotations

from spacex_falcon.dashboard import create_app


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


def test_dashboard_builds_with_analysis_and_prediction_controls(launch_data):
    app = create_app(launch_data)
    ids = _component_ids(app.layout)

    assert app.server is not None
    assert "success-pie-chart" in ids
    assert "success-payload-scatter-chart" in ids
    assert "prediction-button" in ids
    assert "prediction-result" in ids


def test_prediction_callback_returns_probability(launch_data):
    app = create_app(launch_data)
    client = app.server.test_client()
    response = client.post(
        "/_dash-update-component",
        json={
            "output": "prediction-result.children",
            "outputs": {"id": "prediction-result", "property": "children"},
            "inputs": [{"id": "prediction-button", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": "prediction-flight", "property": "value", "value": 41},
                {"id": "prediction-payload", "property": "value", "value": 2_000},
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
