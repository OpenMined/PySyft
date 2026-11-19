"""The signaling server, run in this process with WebSocket clients."""

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode

from syft.sync.connections.webrtc.signaling.server import (
    REPLACED_CLOSE_CODE,
    SignalingServer,
    SignalingServerConfig,
)

ACCESS_KEY = "test-key"
ICE_SERVERS = [{"urls": "stun:stun.example.com:3478"}]
TIMEOUT_SECONDS = 5
A = "a@example.com"
B = "b@example.com"
# Emails with capitals, to check that the server reports each email as its
# owner spells it.
SPELLED_A = "A@Example.com"
SPELLED_B = "B@Example.com"


async def fake_ice_servers() -> list[dict]:
    return ICE_SERVERS


def run_with_server(scenario: Callable[[str], Awaitable[None]]) -> None:
    """Run ``scenario`` with the URL of a server in this process."""

    async def main() -> None:
        config = SignalingServerConfig(host="127.0.0.1", port=0, access_key=ACCESS_KEY)
        async with SignalingServer(config, fake_ice_servers).running() as port:
            await asyncio.wait_for(scenario(f"ws://127.0.0.1:{port}"), TIMEOUT_SECONDS)

    asyncio.run(main())


def hello(email: str, access_key: str = ACCESS_KEY) -> str:
    return json.dumps({"type": "hello", "email": email, "access_key": access_key})


async def receive(client: ClientConnection) -> dict:
    return json.loads(await client.recv())


async def sign_in(url: str, email: str) -> ClientConnection:
    client = await connect(url)
    await client.send(hello(email))
    assert (await receive(client))["type"] == "welcome"
    return client


async def close_code(client: ClientConnection) -> int:
    """The close code, once the server closes ``client``."""
    with pytest.raises(ConnectionClosed) as closed:
        await client.recv()
    received = closed.value.rcvd
    assert received is not None, "the server closed without a close frame"
    return received.code


def test_hello_with_right_key_gets_ice_servers():
    async def scenario(url: str) -> None:
        async with connect(url) as client:
            await client.send(hello(A))
            assert await receive(client) == {
                "type": "welcome",
                "ice_servers": ICE_SERVERS,
            }

    run_with_server(scenario)


def test_hello_with_wrong_key_is_refused():
    async def scenario(url: str) -> None:
        async with connect(url) as client:
            await client.send(hello(A, access_key="wrong"))
            assert await receive(client) == {
                "type": "error",
                "reason": "The access key is wrong",
            }
            assert await close_code(client) == CloseCode.POLICY_VIOLATION

    run_with_server(scenario)


def test_non_json_message_closes_connection():
    async def scenario(url: str) -> None:
        async with connect(url) as client:
            await client.send("not json")
            assert (await receive(client))["type"] == "error"
            assert await close_code(client) == CloseCode.POLICY_VIOLATION

    run_with_server(scenario)


def test_offer_and_answer_reach_peers():
    async def scenario(url: str) -> None:
        a, b = await sign_in(url, A), await sign_in(url, B)

        await a.send(json.dumps({"type": "offer", "to": B, "id": "1", "sdp": "o"}))
        assert await receive(b) == {"type": "offer", "from": A, "id": "1", "sdp": "o"}
        await b.send(json.dumps({"type": "answer", "to": A, "id": "1", "sdp": "x"}))
        assert await receive(a) == {"type": "answer", "from": B, "id": "1", "sdp": "x"}

    run_with_server(scenario)


def test_offer_to_offline_peer_gets_error():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)

        await a.send(json.dumps({"type": "offer", "to": B, "id": "1", "sdp": "o"}))
        assert await receive(a) == {
            "type": "error",
            "id": "1",
            "reason": f"{B} is offline",
        }

    run_with_server(scenario)


def test_watch_reports_peer_already_online():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)
        await sign_in(url, B)

        await a.send(json.dumps({"type": "watch", "peers": [B]}))
        assert await receive(a) == {"type": "online", "email": B}

    run_with_server(scenario)


def test_watch_reports_peer_coming_online_and_going_offline():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)
        # A is online, so its own entry comes back at once: the server has the watch.
        await a.send(json.dumps({"type": "watch", "peers": [B, A]}))
        assert await receive(a) == {"type": "online", "email": A}

        b = await sign_in(url, B)
        assert await receive(a) == {"type": "online", "email": B}
        await b.close()
        assert await receive(a) == {"type": "offline", "email": B}

    run_with_server(scenario)


def test_second_sign_in_replaces_first():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)
        first = await sign_in(url, B)
        second = await sign_in(url, B)
        assert await close_code(first) == REPLACED_CLOSE_CODE

        await a.send(json.dumps({"type": "offer", "to": B, "id": "1", "sdp": "o"}))
        assert (await receive(second))["type"] == "offer"

    run_with_server(scenario)


def test_emails_compare_in_lower_case():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)
        b = await sign_in(url, "B@Example.com")

        await a.send(
            json.dumps({"type": "offer", "to": "b@EXAMPLE.com", "id": "1", "sdp": "o"})
        )
        assert (await receive(b))["type"] == "offer"

    run_with_server(scenario)


async def send_peering_request(
    url: str, email: str, request: dict, access_key: str = ACCESS_KEY
) -> dict:
    async with connect(url) as client:
        credentials = {"email": email, "access_key": access_key}
        await client.send(json.dumps({**request, **credentials}))
        return await receive(client)


def test_peer_request_reaches_peer():
    async def scenario(url: str) -> None:
        request = {"type": "request_peer", "peer": B}
        assert await send_peering_request(url, A, request) == {"type": "ok"}

        listed = await send_peering_request(url, B, {"type": "list_peer_requests"})
        assert listed == {"type": "peer_requests", "emails": [A]}

    run_with_server(scenario)


def test_document_needs_grant_before_peer_reads_it():
    async def scenario(url: str) -> None:
        put = {"type": "put_document", "name": "info", "data": "x"}
        grant = {"type": "grant_document", "name": "info", "reader": B}
        get = {"type": "get_document", "owner": A, "name": "info"}

        await send_peering_request(url, A, put)
        assert await send_peering_request(url, B, get) == {
            "type": "document",
            "data": None,
        }
        await send_peering_request(url, A, grant)
        assert await send_peering_request(url, B, get) == {
            "type": "document",
            "data": "x",
        }

    run_with_server(scenario)


def test_peering_request_with_wrong_key_is_refused():
    async def scenario(url: str) -> None:
        request = {"type": "list_peer_requests"}
        reply = await send_peering_request(url, A, request, access_key="wrong")
        assert reply == {"type": "error", "reason": "The access key is wrong"}

    run_with_server(scenario)


def test_peer_request_keeps_sender_spelling():
    async def scenario(url: str) -> None:
        await send_peering_request(url, SPELLED_A, {"type": "request_peer", "peer": B})

        listed = await send_peering_request(url, B, {"type": "list_peer_requests"})
        assert listed == {"type": "peer_requests", "emails": [SPELLED_A]}

    run_with_server(scenario)


def test_session_reports_peer_spelling():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)
        # A is online, so its own entry comes back at once: the server has the watch.
        await a.send(json.dumps({"type": "watch", "peers": [B, A]}))
        assert await receive(a) == {"type": "online", "email": A}

        b = await sign_in(url, SPELLED_B)
        assert await receive(a) == {"type": "online", "email": SPELLED_B}
        await b.send(json.dumps({"type": "offer", "to": A, "id": "1", "sdp": "o"}))
        offer = {"type": "offer", "from": SPELLED_B, "id": "1", "sdp": "o"}
        assert await receive(a) == offer

    run_with_server(scenario)


def test_watch_replaces_earlier_watch_list():
    async def scenario(url: str) -> None:
        a = await sign_in(url, A)
        await a.send(json.dumps({"type": "watch", "peers": [B]}))
        # A is online, so its own entry comes back at once: the server has the watch.
        await a.send(json.dumps({"type": "watch", "peers": [A]}))
        assert await receive(a) == {"type": "online", "email": A}

        await sign_in(url, B)
        await a.send(json.dumps({"type": "watch", "peers": [A]}))
        assert await receive(a) == {"type": "online", "email": A}

    run_with_server(scenario)
