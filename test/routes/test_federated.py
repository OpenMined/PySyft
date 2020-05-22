from unittest.mock import patch

from grid.app.main.routes import federated
from grid.app.main.exceptions import PyGridError


def test_worker_cycle_request_bad_request_json(client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    result = client.post(
        "/federated/cycle-request", data="{bad", content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json().get("error") == (
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    )


@patch("grid.app.main.routes.federated.cycle_request")
def test_worker_cycle_request_bad_request_pygrid(mock_cycle_request, client):
    """
    assert that the endpoint returns a 400 for a PyGridError
    """
    mock_cycle_request.side_effect = PyGridError("test")

    result = client.post("/federated/cycle-request", json={"test": "data"})

    assert result.status_code == 400
    assert result.get_json().get("error") == "test"


@patch("grid.app.main.routes.federated.cycle_request")
def test_worker_cycle_request_internal_server_error(mock_cycle_request, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_cycle_request.side_effect = RuntimeError("test")

    result = client.post("/federated/cycle-request", json={"test": "data"})

    assert result.status_code == 500
    assert result.get_json().get("error") == "test"


def test_report_diff_bad_request_json(client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    result = client.post(
        "/federated/report", data="{bad", content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json().get("error") == (
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    )


@patch("grid.app.main.routes.federated.report")
def test_report_diff_bad_request_pygrid(mock_report, client):
    """
    assert that the endpoint returns a 400 for a PyGridError
    """
    mock_report.side_effect = PyGridError("test")

    result = client.post("/federated/report", json={"test": "data"})

    assert result.status_code == 400
    assert result.get_json().get("error") == "test"


@patch("grid.app.main.routes.federated.report")
def test_report_diff_internal_server_error(mock_report, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_report.side_effect = RuntimeError("test")

    result = client.post("/federated/report", json={"test": "data"})

    assert result.status_code == 500
    assert result.get_json().get("error") == "test"
