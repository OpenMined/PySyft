# stdlib
import asyncio
import json
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

# third party
from aiortc import RTCDataChannel
from aiortc import RTCPeerConnection
from aiortc import RTCSessionDescription
from aiortc.contrib.signaling import object_from_string
from nacl.signing import SigningKey
import nest_asyncio
import pytest

# syft absolute
from syft.core.node.common.service.repr_service import ReprMessage
from syft.core.node.domain.domain import Domain
from syft.grid.connections.webrtc import DC_CHUNK_START_SIGN
from syft.grid.connections.webrtc import DC_MAX_CHUNK_SIZE
from syft.grid.connections.webrtc import OrderedChunk
from syft.grid.connections.webrtc import WebRTCConnection

# from pytest import MonkeyPatch


class AsyncMock(Mock):
    async def __call__(self, *args: Any, **kwargs: Any) -> None:
        return super(AsyncMock, self).__call__(*args, **kwargs)


def get_signing_key() -> SigningKey:
    # return a the signing key used to sign the get_signed_message_bytes fixture
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


# --------------------- INIT ---------------------


@pytest.mark.asyncio
def test_init() -> None:
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


# TODO: Fix with new traceback_and_raise method
# @pytest.mark.asyncio
# async def test_init_raise_exception(monkeypatch: MonkeyPatch) -> None:
#     nest_asyncio.apply()

#     with patch(
#         "syft.grid.connections.webrtc.logger", side_effect=RuntimeError()
#     ) as mock_logger:
#         with patch(
#             "syft.grid.connections.webrtc.RTCPeerConnection", side_effect=Exception()
#         ):
#             with pytest.raises(Exception):
#                 domain = Domain(name="test")
#                 WebRTCConnection(node=domain)

#             expected_log = "Got an exception in WebRTCConnection __init__. "
#             assert mock_logger.error.call_args[0][0] == expected_log


# --------------------- METHODS ---------------------


@pytest.mark.asyncio
async def test_set_offer_raise_exception() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)

    with patch(
        "syft.grid.connections.webrtc.RTCPeerConnection.createDataChannel",
        side_effect=Exception(),
    ):
        with pytest.raises(Exception):
            await webrtc._set_offer()


@pytest.mark.asyncio
async def test_set_offer_sets_channel() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    await webrtc._set_offer()
    assert isinstance(webrtc.channel, RTCDataChannel)
    assert webrtc.channel.bufferedAmountLowThreshold == 4 * DC_MAX_CHUNK_SIZE


@pytest.mark.asyncio
async def test_set_offer_on_open() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    await webrtc._set_offer()

    channel_methods = list(webrtc.channel._events.values())
    on_open = list(channel_methods[1].values())[0]

    coro_mock = AsyncMock()
    with patch(
        "syft.grid.connections.webrtc.WebRTCConnection.producer",
        return_value=coro_mock(),
    ) as producer_mock:
        await on_open()
        assert producer_mock.call_count == 1


@pytest.mark.asyncio
async def test_set_offer_on_message() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    await webrtc._set_offer()

    channel_methods = list(webrtc.channel._events.values())
    on_message = list(channel_methods[2].values())[0]

    coro_mock = AsyncMock()
    with patch(
        "syft.grid.connections.webrtc.WebRTCConnection.consumer",
        return_value=coro_mock(),
    ) as consumer_mock:
        await on_message(OrderedChunk(1, DC_CHUNK_START_SIGN).save())
        assert consumer_mock.call_count == 0

        await on_message(OrderedChunk(0, b"a").save())
        assert consumer_mock.call_count == 1


# TODO: Fix with new traceback_and_raise method
# @pytest.mark.asyncio
# async def test_set_answer_raise_exception() -> None:
#     nest_asyncio.apply()

#     domain = Domain(name="test")
#     webrtc = WebRTCConnection(node=domain)
#     offer_payload = await webrtc._set_offer()

#     # FIXME: Nahua is not happy with this test because it "indirectly" triggered exception
#     with patch("syft.grid.connections.webrtc.logger") as mock_logger:
#         with pytest.raises(Exception):
#             # This would fail because 'have-local-offer' is applied
#             await webrtc._set_answer(payload=offer_payload)
#         assert mock_logger.error

#         expected_log = "Got an exception in WebRTCConnection _set_answer."
#         assert expected_log in mock_logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_set_answer_on_datachannel() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    offer_payload = await webrtc._set_offer()

    answer_webrtc = WebRTCConnection(node=domain)
    await answer_webrtc._set_answer(payload=offer_payload)

    channel_methods = list(answer_webrtc.peer_connection._events.values())
    on_datachannel = list(channel_methods[1].values())[0]

    coro_mock = AsyncMock()
    with patch(
        "syft.grid.connections.webrtc.WebRTCConnection.producer",
        return_value=coro_mock(),
    ) as producer_mock:
        channel = answer_webrtc.peer_connection.createDataChannel("datachannel")
        on_datachannel(channel)
        assert producer_mock.call_count == 1


@pytest.mark.asyncio
async def test_set_answer_on_message() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    offer_payload = await webrtc._set_offer()

    answer_webrtc = WebRTCConnection(node=domain)
    await answer_webrtc._set_answer(payload=offer_payload)

    channel_methods = list(answer_webrtc.peer_connection._events.values())
    on_channel = list(channel_methods[1].values())[0]

    coro_mock = AsyncMock()
    with patch(
        "syft.grid.connections.webrtc.WebRTCConnection.consumer",
        return_value=coro_mock(),
    ) as consumer_mock:
        channel = answer_webrtc.peer_connection.createDataChannel("datachannel")
        on_channel(channel)

        channel_methods = list(answer_webrtc.channel._events.values())
        on_message = list(channel_methods[1].values())[0]

        await on_message(OrderedChunk(1, DC_CHUNK_START_SIGN).save())
        assert consumer_mock.call_count == 0

        await on_message(OrderedChunk(0, b"a").save())
        assert consumer_mock.call_count == 1


# TODO: Fix with new traceback_and_raise method
# @pytest.mark.asyncio
# async def test_finish_coroutines_raise_exception() -> None:
#     nest_asyncio.apply()

#     domain = Domain(name="test")
#     webrtc = WebRTCConnection(node=domain)

#     with patch("syft.grid.connections.webrtc.logger") as mock_logger:
#         with patch(
#             "syft.grid.connections.webrtc.RTCDataChannel.close", side_effect=Exception()
#         ):
#             with pytest.raises(Exception):
#                 webrtc._finish_coroutines()

#             expected_log = "Got an exception in WebRTCConnection _finish_coroutines."
#             assert expected_log in mock_logger.error.call_args[0][0]


# TODO: Fix with new traceback_and_raise method
# @pytest.mark.asyncio
# async def test_close_raise_exception() -> None:
#     nest_asyncio.apply()

#     domain = Domain(name="test")
#     webrtc = WebRTCConnection(node=domain)

#     with patch("syft.grid.connections.webrtc.logger") as mock_logger:
#         with patch(
#             "syft.grid.connections.webrtc.RTCDataChannel.close", side_effect=Exception()
#         ):
#             with pytest.raises(Exception):
#                 webrtc.close()

#             expected_log = "Got an exception in WebRTCConnection close."
#             assert expected_log in mock_logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_close() -> None:
    nest_asyncio.apply()

    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    await webrtc._set_offer()

    with patch("syft.grid.connections.webrtc.RTCDataChannel.send") as send_mock:
        with patch(
            "syft.grid.connections.webrtc.WebRTCConnection._finish_coroutines"
        ) as finish_mock:
            webrtc.close()
            assert send_mock.call_count == 1
            assert finish_mock.call_count == 1


# --------------------- INTEGRATION ---------------------


@pytest.mark.asyncio
def test_init_without_event_loop() -> None:
    domain = Domain(name="test")
    webrtc = WebRTCConnection(node=domain)
    assert webrtc is not None


@pytest.mark.slow
@pytest.mark.asyncio
async def test_signaling_process() -> None:

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

    test_domain = Domain(name="test")

    webrtc_node = WebRTCConnection(node=test_domain)

    msg = ReprMessage(address=test_domain.address)
    signing_key = SigningKey.generate()
    test_domain.root_verify_key = signing_key.verify_key
    signed_msg = msg.sign(signing_key=signing_key)

    msg_bin = signed_msg.to_bytes()

    await webrtc_node.consumer(msg=msg_bin)
