from unittest.mock import patch


@patch("src.app.routes.network.network_manager")
def test_choose_encrypted_model_host_value_error(mock_network_manager, client):
    """assert that the endpoint returns a 400 for value error."""
    mock_network_manager.connected_nodes.return_value = {}

    result = client.get("/choose-encrypted-model-host")

    assert result.status_code == 400
    assert len(result.get_json()) == 0


@patch("src.app.routes.network.network_manager")
def test_choose_encrypted_model_host_internal_server_error(
    mock_network_manager, client
):
    """assert that the endpoint returns a 500 for an unknown error
    condition."""
    mock_network_manager.connected_nodes.side_effect = RuntimeError("test")

    result = client.get("/choose-encrypted-model-host")

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"
