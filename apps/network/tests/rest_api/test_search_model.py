from unittest.mock import patch


@patch("src.app.routes.network._get_model_hosting_nodes")
def test_search_model_bad_request_value_error(mock_get_model_hosting_nodes, client):
    """assert that the endpoint returns a 400 for malformed JSON."""
    mock_get_model_hosting_nodes.side_effect = ValueError()

    result = client.post("/search-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


def test_search_model_bad_request_key_error(client):
    """assert that the endpoint returns a 400 for missing keys in sent JSON."""
    result = client.post("/search-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("src.app.routes.network._get_model_hosting_nodes")
def test_search_model_internal_server_error(mock_get_model_hosting_nodes, client):
    """assert that the endpoint returns a 500 for an unknown error
    condition."""
    mock_get_model_hosting_nodes.side_effect = RuntimeError("test")

    result = client.post("/search-model", json={"model_id": "data"})

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"
