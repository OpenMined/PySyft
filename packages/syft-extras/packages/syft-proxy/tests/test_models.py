import pytest
from syft_rpc import SyftRequest

from syft_proxy.models import (
    RPCBroadcastRequest,
    RPCSendRequest,
    RPCStatus,
    RPCStatusCode,
)


def test_rpc_send_request_valid():
    """Test the creation of a valid RPCSendRequest."""
    request = RPCSendRequest(
        app_name="test_app",
        url="syft://user@openmined.org/datasite/public",
        body={"data": "test"},
    )
    assert request.app_name == "test_app"
    assert request.url == "syft://user@openmined.org/datasite/public"
    assert request.body == {"data": "test"}


def test_rpc_send_request_missing_app_name():
    """Test that RPCSendRequest raises a ValueError when the app_name is missing."""
    with pytest.raises(ValueError):
        RPCSendRequest(
            app_name="",
            url="syft://user@openmined.org/datasite/public",
            body={"data": "test"},
        )


def test_rpc_send_request_invalid_headers():
    """Test that RPCSendRequest raises a ValueError when headers are invalid.

    This test checks the behavior of the RPCSendRequest class when
    the headers parameter is provided with invalid data types.
    Specifically, it ensures that a ValueError is raised when
    the headers dictionary contains non-string keys.
    """
    with pytest.raises(ValueError):
        RPCSendRequest(
            app_name="test_app",
            url="syft://user@openmined.org/datasite/public",
            body={"data": "test"},
            headers={123: "value"},
        )


def test_rpc_broadcast_request_valid():
    """Test the creation of a valid RPCBroadcastRequest.

    This test verifies that an RPCBroadcastRequest can be created with
    a list of valid URLs and a body. It asserts that the length of the
    URLs in the broadcast request matches the expected number of URLs.
    """
    broadcast_request = RPCBroadcastRequest(
        urls=[
            "syft://user1@openmined.org/datasite/public",
            "syft://user2@openmined.org/datasite/public",
        ],
        body={"data": "test"},
    )
    assert len(broadcast_request.urls) == 2


def test_rpc_broadcast_request_empty_urls():
    """Test that RPCBroadcastRequest raises a ValueError when the URLs list is empty.

    This test checks the behavior of the RPCBroadcastRequest class when
    an empty list is provided for the URLs parameter. It ensures that
    a ValueError is raised, indicating that at least one URL must be
    provided for the broadcast request to be valid.
    """
    with pytest.raises(ValueError):
        RPCBroadcastRequest(urls=[], body={"data": "test"})


def test_rpc_status_valid():
    """Test the creation of a valid RPCStatus.

    This test verifies that an RPCStatus can be created with a valid
    ID, status code, and request. It asserts that the ID and status
    of the created RPCStatus match the expected values.
    """
    status = RPCStatus(
        id="1",
        status=RPCStatusCode.PENDING,
        request=SyftRequest(
            sender="user@openmined.org",
            url="syft://user@openmined.org/datasite/public",
        ),
        response=None,
    )
    assert status.id == "1"
    assert status.status == RPCStatusCode.PENDING


def test_rpc_status_invalid_code():
    """Test that RPCStatus raises a ValueError when an invalid status code is provided.

    This test checks the behavior of the RPCStatus class when an
    invalid status code is passed. It ensures that a ValueError is
    raised, indicating that the status code must be valid.
    """
    with pytest.raises(ValueError):
        RPCStatus(id="1", status="INVALID_CODE", request=None, response=None)


def test_rpc_status_missing_fields():
    """Test that RPCStatus raises a ValueError when required fields are missing.

    This test verifies that the RPCStatus class raises a ValueError
    when the ID is empty and a valid status code is provided. It
    ensures that all required fields must be present for the RPCStatus
    to be valid.
    """
    with pytest.raises(ValueError):
        RPCStatus(id="", status=RPCStatusCode.NOT_FOUND, request=None, response=None)
