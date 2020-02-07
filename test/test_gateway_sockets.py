import json

import pytest
import websockets
import aiounittest
from uuid import UUID

from test import GATEWAY_WS_URL


# Useful functions


async def send_ws_message(data):
    try:
        async with websockets.connect(GATEWAY_WS_URL) as websocket:
            await websocket.send(json.dumps(data))
            message = await websocket.recv()
            if message:
                return json.loads(message)
            else:
                return None
    except websockets.exceptions.ConnectionClosed:
        pytest.fail("The connection to the grid websocket served was closed.")


def get_user(message):
    return json.loads(message.get("data")).get("user")


async def get_protocol():
    return await send_ws_message(
        {"type": "get-protocol", "data": {"protocolId": "test-protocol"}}
    )


# Gateway WebSockets API Test


class GatewaySocketsTest(aiounittest.AsyncTestCase):
    async def test_socket_ping_alive(self):
        response = await send_ws_message({"type": "socket-ping", "data": {}})
        self.assertEqual(response, {"alive": "True"})

    async def test_type_get_protocol(self):
        response = await get_protocol()
        message_type = response.get("type")
        self.assertEqual(message_type, "get-protocol")

    async def test_get_protocol_creator_role(self):
        response = await get_protocol()

        user = get_user(response)
        role = user.get("role")

        self.assertEqual(role, "creator")

    async def test_get_protocol_join_scope(self):
        response = await get_protocol()

        creator = get_user(response)
        generated_scope_id = creator.get("scopeId")

        response = await send_ws_message(
            {
                "type": "get-protocol",
                "data": {"protocolId": "test-protocol", "scopeId": generated_scope_id},
            }
        )

        participant = get_user(response)
        role = participant.get("role")

        self.assertEqual(role, "participant")

    async def test_protocol_id(self):
        protocol_id = "not-fake-protocol-id"
        response = await send_ws_message(
            {"type": "get-protocol", "data": {"protocolId": protocol_id}}
        )

        user = get_user(response)
        self.assertEqual(protocol_id, user.get("protocolId"))

        creator = get_user(response)
        generated_scope_id = creator.get("scopeId")

        response = await send_ws_message(
            {
                "type": "get-protocol",
                "data": {"protocolId": "test-protocol", "scopeId": generated_scope_id},
            }
        )

        participant = get_user(response)
        role = participant.get("role")

        self.assertEqual(role, "participant")

    async def test_type_webrtc_join_room(self):
        response = await get_protocol()

        user = get_user(response)
        worker_id = user.get("workerId")
        scope_id = user.get("scopeId")
        try:
            response = await send_ws_message(
                {
                    "type": "webrtc: join-room",
                    "data": {"workerId": worker_id, "scopeId": scope_id},
                }
            )
        except:
            pytest.fail("There was an error trying webrtc: join-room")
        else:
            self.assertEqual(response, None)

    async def test_webrtc_left_room(self):
        response = await get_protocol()

        user = get_user(response)
        worker_id = user.get("workerId")
        scope_id = user.get("scopeId")

        await send_ws_message(
            {
                "type": "webrtc: join-room",
                "data": {"workerId": worker_id, "scopeId": scope_id},
            }
        )

        try:
            response = await send_ws_message(
                {
                    "type": "webrtc: peer-left",
                    "data": {"workerId": worker_id, "scopeId": scope_id},
                }
            )
        except:
            pytest.fail("There was an error trying webrtc: peer-left")
        else:
            self.assertEqual(response, None)

    async def test_webrtc_internal_message(self):
        response = await get_protocol()

        user = get_user(response)
        scope_id = user.get("scopeId")

        creator_id = user.get("workerId")

        response = await send_ws_message(
            {
                "type": "get-protocol",
                "data": {"protocol-id": "test-protocol", "scopeId": scope_id},
            }
        )

        participant_id = get_user(response).get("workerId")

        try:
            response = await send_ws_message(
                {
                    "type": "webrtc: internal-message",
                    "data": {
                        "workerId": creator_id,
                        "scopeId": scope_id,
                        "to": participant_id,
                        "type": "offer",
                        "data": "some message here",
                    },
                }
            )
        except:
            pytest.fail("There was an error trying webrtc: peer-left")
        else:
            self.assertEqual(response, None)

    async def test_invalid_message_type(self):
        response = await send_ws_message(
            {"type": "not-a-type", "data": {"protocol-id": "test-protocol"}}
        )
        self.assertEqual(response, {"error": "Invalid JSON format/field!"})
