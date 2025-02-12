import json
import os

from fastapi.testclient import TestClient
from syft_core import Client

from syft_proxy.models import RPCSendRequest
from syft_proxy.server import app

syft_client = Client.load()

client = TestClient(app)


# Workflow Tests
def test_index_endpoint():
    """Test the index endpoint to ensure it returns a 200 status code and contains the expected text."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SyftBox HTTP Proxy" in response.text


def test_rpc_send_non_blocking():
    """Test sending a non-blocking RPC request and verify the response status and status message."""
    rpc_req = RPCSendRequest(
        url="syft://user@openmined.org",
        headers={},
        body={},
        expiry="30s",
        app_name="test_app",
    )
    response = client.post(
        "/rpc", json=rpc_req.model_dump(), params={"blocking": False}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "RPC_PENDING"


def test_rpc_send_blocking():
    """Test sending a blocking RPC request and verify the response status and ID presence."""
    rpc_req = RPCSendRequest(
        url="syft://user@openmined.org",
        headers={},
        body={},
        expiry="1s",
        app_name="test_app",
    )
    response = client.post("/rpc", json=rpc_req.model_dump(), params={"blocking": True})
    assert response.status_code in [200, 419]
    assert isinstance(response.json(), dict)
    assert response.json().get("id", None) is not None


def test_rpc_schema():
    """Test the RPC schema endpoint to ensure it returns the correct schema for the specified app."""
    app_path = syft_client.api_data("test_app")
    app_schema = app_path / "rpc" / "rpc.schema.json"

    os.makedirs(app_path / "rpc", exist_ok=True)
    schema = {
        "sender": "user@openmined.org",
        "method": "GET",
        "url": "syft://user1@openmined.org",
    }
    if not os.path.isfile(app_schema):
        with open(app_schema, "w") as f:
            f.write(json.dumps(schema))

    response = client.get("/rpc/schema/test_app")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert response.json() == schema


def test_rpc_status_found():
    """Test the RPC status endpoint to ensure it returns a 200 status code for a valid request ID."""
    rpc_req = RPCSendRequest(
        url="syft://test@openmined.org/public/rpc",
        headers={"Content-Type": "application/json", "User-Agent": "MyApp/1.0"},
        body={},
        expiry="30s",
        app_name="test_app",
    )
    response = client.post(
        "/rpc", json=rpc_req.model_dump(), params={"blocking": False}
    )

    rpc_request_id = response.json()["id"]
    response = client.get(f"/rpc/status/{rpc_request_id}")
    assert response.status_code == 200


def test_rpc_status_not_found():
    """Test the RPC status endpoint to ensure it returns a 404 status code for a non-existent request ID."""
    response = client.get("/rpc/status/non_existent_id")
    assert response.status_code == 404


# Edge Case Tests
def test_rpc_send_invalid_request():
    """Test sending an invalid RPC request to ensure it returns a 422 status code due to missing required fields."""
    response = client.post("/rpc", json={})  # Missing required fields
    assert response.status_code == 422


def test_rpc_schema_non_existent_app():
    """Test the RPC schema endpoint to ensure it returns a 500 status code for a non-existent app."""
    response = client.get("/rpc/schema/non_existent_app")
    assert response.status_code == 500


def test_rpc_status_non_existent_id():
    """Test the RPC status endpoint to ensure it returns a 404 status code for a non-existent request ID."""
    response = client.get("/rpc/status/non_existent_id")
    assert response.status_code == 404
