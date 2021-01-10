# stdlib
import json
from unittest.mock import patch
from unittest.mock import Mock

# third party
from aiortc import RTCPeerConnection
from aiortc import RTCSessionDescription
from aiortc.contrib.signaling import object_from_string
import asyncio
from nacl.signing import SigningKey
import nest_asyncio
import pytest

# syft absolute
from syft.core.node.common.service.repr_service import ReprMessage
from syft.core.node.domain.domain import Domain
from syft.grid.connections.webrtc import WebRTCConnection


def get_signing_key() -> SigningKey:
    # return a the signing key used to sign the get_signed_message_bytes fixture
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


@pytest.mark.asyncio
async def test_init() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    assert webrtc is not None
    assert webrtc.node == domain
    assert webrtc.loop is not None
    assert isinstance(webrtc.producer_pool, asyncio.Queue)
    assert isinstance(webrtc.consumer_pool, asyncio.Queue)
    assert isinstance(webrtc.peer_connection, RTCPeerConnection)
    assert not webrtc._client_address


@pytest.mark.asyncio
@patch("syft.grid.connections.webrtc.logger")
@patch("syft.grid.connections.webrtc.get_running_loop", side_effect=RuntimeError())
@patch("asyncio.new_event_loop", return_value="mock_loop")
async def test_init_patch_runtime_error(
    mock_new_event_loop: Mock,
    mock_get_running_loop: Mock,
    mock_logger: Mock,
) -> None:
    nest_asyncio.apply()

    expected_log = "♫♫♫ > ...error getting a running event Loop... "

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)

    assert mock_logger.error.call_args[0][0] == expected_log
    assert webrtc.loop == "mock_loop"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_signaling_process() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)

    offer_payload = await webrtc._set_offer()
    offer_dict = json.loads(offer_payload)
    aiortc_session = object_from_string(offer_payload)

    assert "sdp" in offer_dict
    assert "type" in offer_dict
    assert offer_dict["type"] == "offer"
    assert isinstance(aiortc_session, RTCSessionDescription)

    answer_webrtc = WebRTCConnection(node=domain)
    answer_payload = await answer_webrtc._set_answer(payload=offer_payload)
    answer_dict = json.loads(answer_payload)
    aiortc_session = object_from_string(answer_payload)

    assert "sdp" in answer_dict
    assert "type" in answer_dict
    assert answer_dict["type"] == "answer"
    assert isinstance(aiortc_session, RTCSessionDescription)

    response = await webrtc._process_answer(payload=answer_payload)
    assert response is None


@pytest.mark.asyncio
async def test_consumer_request() -> None:
    nest_asyncio.apply()

    test_domain = Domain(name="test")

    webrtc_node = WebRTCConnection(node=test_domain)

    msg = ReprMessage(address=test_domain.address)
    signing_key = SigningKey.generate()
    test_domain.root_verify_key = signing_key.verify_key
    signed_msg = msg.sign(signing_key=signing_key)

    msg_bin = signed_msg.to_bytes()

    await webrtc_node.consumer(msg=msg_bin)
