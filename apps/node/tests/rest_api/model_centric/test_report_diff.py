from unittest.mock import patch

from src.app.main.core.exceptions import PyGridError
from src.app.main.routes.model_centric import routes


def test_report_diff_bad_request_json(client):
    """assert that the endpoint returns a 400 for malformed JSON."""
    result = client.post(
        "/model-centric/report", data="{bad", content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json().get("error") == (
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    )


@patch("src.app.main.routes.model_centric.routes.report")
def test_report_diff_bad_request_pygrid(mock_report, client):
    """assert that the endpoint returns a 400 for a PyGridError."""
    mock_report.side_effect = PyGridError("test")

    result = client.post("/model-centric/report", json={"test": "data"})

    assert result.status_code == 400
    assert result.get_json().get("error") == "test"


@patch("src.app.main.routes.model_centric.routes.report")
def test_report_diff_internal_server_error(mock_report, client):
    """assert that the endpoint returns a 500 for an unknown error
    condition."""
    mock_report.side_effect = RuntimeError("test")

    result = client.post("/model-centric/report", json={"test": "data"})

    assert result.status_code == 500
    assert result.get_json().get("error") == "test"
