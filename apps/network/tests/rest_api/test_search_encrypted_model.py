from unittest.mock import patch


@patch("src.app.routes.network.network_manager")
def test_search_encrypted_model_bad_request_value_error(mock_network_manager, client):
    """assert that the endpoint returns a 400 for malformed JSON."""
    mock_network_manager.connected_nodes.side_effect = ValueError()

    result = client.post("/search-encrypted-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("src.app.routes.network.network_manager")
def test_search_encrypted_model_bad_request_key_error(mock_network_manager, client):
    """assert that the endpoint returns a 400 for missing keys in sent JSON."""
    mock_network_manager.connected_nodes.side_effect = KeyError()

    result = client.post("/search-encrypted-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("src.app.routes.network.network_manager")
def test_search_encrypted_model_internal_server_error(mock_network_manager, client):
    """assert that the endpoint returns a 500 for an unknown error
    condition."""
    mock_network_manager.connected_nodes.side_effect = RuntimeError("test")

    result = client.post(
        "/search-encrypted-model", json={"node-id": "data", "node-address": "data"}
    )

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"
