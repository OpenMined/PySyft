from unittest.mock import patch

from grid.app.main.routes import network
from grid.app.main.exceptions import PyGridError


def test_join_grid_node_bad_request_json(client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    result = client.post("/join", data="{bad", content_type="application/json")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


def test_join_grid_node_bad_request_key_error(client):
    """
    assert that the endpoint returns a 400 for missing keys in sent JSON
    """
    result = client.post("/join", json={"test": "data"})
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network.network_manager")
def test_join_grid_node_already_registered(mock_network_manager, client):
    """
    assert that the endpoint returns a 400 for an already registered node
    """
    mock_network_manager.register_new_node.return_value = False

    result = client.post("/join", json={"node-id": "data", "node-address": "data"})

    assert result.status_code == 409
    assert result.get_json().get("message") == "This ID has already been registered"


@patch("grid.app.main.routes.network.network_manager")
def test_join_grid_node_internal_server_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_network_manager.register_new_node.side_effect = RuntimeError("test")

    result = client.post("/join", json={"node-id": "data", "node-address": "data"})

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"


def test_delete_grid_node_bad_request_json(client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    result = client.delete("/delete-node", data="{bad", content_type="application/json")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


def test_delete_grid_node_bad_request_key_error(client):
    """
    assert that the endpoint returns a 400 for missing keys in sent JSON
    """
    result = client.delete("/delete-node", json={"test": "data"})
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network.network_manager")
def test_join_grid_node_already_registered(mock_network_manager, client):
    """
    assert that the endpoint returns a 400 for a node not yet registered
    """
    mock_network_manager.delete_node.return_value = False

    result = client.delete(
        "/delete-node", json={"node-id": "data", "node-address": "data"}
    )

    assert result.status_code == 409
    assert (
        result.get_json().get("message") == "This ID was not found in connected nodes"
    )


@patch("grid.app.main.routes.network.network_manager")
def test_delete_grid_node_internal_server_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_network_manager.delete_node.side_effect = RuntimeError("test")

    result = client.delete(
        "/delete-node", json={"node-id": "data", "node-address": "data"}
    )

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"


@patch("grid.app.main.routes.network.network_manager")
def test_search_encrypted_model_bad_request_value_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    mock_network_manager.connected_nodes.side_effect = ValueError()

    result = client.post("/search-encrypted-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network.network_manager")
def test_search_encrypted_model_bad_request_key_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 400 for missing keys in sent JSON
    """
    mock_network_manager.connected_nodes.side_effect = KeyError()

    result = client.post("/search-encrypted-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network.network_manager")
def test_search_encrypted_model_internal_server_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_network_manager.connected_nodes.side_effect = RuntimeError("test")

    result = client.post(
        "/search-encrypted-model", json={"node-id": "data", "node-address": "data"}
    )

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"


@patch("grid.app.main.routes.network._get_model_hosting_nodes")
def test_search_model_bad_request_value_error(mock_get_model_hosting_nodes, client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    mock_get_model_hosting_nodes.side_effect = ValueError()

    result = client.post("/search-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


def test_search_model_bad_request_key_error(client):
    """
    assert that the endpoint returns a 400 for missing keys in sent JSON
    """
    result = client.post("/search-model")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network._get_model_hosting_nodes")
def test_search_model_internal_server_error(mock_get_model_hosting_nodes, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_get_model_hosting_nodes.side_effect = RuntimeError("test")

    result = client.post("/search-model", json={"model_id": "data"})

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"


@patch("grid.app.main.routes.network.network_manager")
def test_search_bad_request_value_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 400 for malformed JSON
    """
    mock_network_manager.connected_nodes.side_effect = ValueError()

    result = client.post("/search", data="{bad", content_type="application/json")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network.network_manager")
def test_search_bad_request_key_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 400 for missing keys in sent JSON
    """
    mock_network_manager.connected_nodes.side_effect = KeyError()

    result = client.post("/search", json={})
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("grid.app.main.routes.network.network_manager")
def test_search_internal_server_error(mock_network_manager, client):
    """
    assert that the endpoint returns a 500 for an unknown error condition
    """
    mock_network_manager.connected_nodes.side_effect = RuntimeError("test")

    result = client.post("/search", json={})

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"
