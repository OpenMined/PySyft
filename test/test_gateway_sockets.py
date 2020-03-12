import json
import pytest
import binascii
import websockets
import aiounittest

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import syft as sy

from syft.serde.serde import serialize, deserialize
from syft.serde.msgpack import serde

from uuid import UUID

from test import GATEWAY_WS_URL

hook = sy.TorchHook(th)

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

    @pytest.mark.skip
    async def test_fl_process(self):
        """ 1 - Host Federated Training """
        # Plan Functions
        @sy.func2plan(args_shape=[(1,), (1,), (1,)])
        def foo_1(x, y, z):
            a = x + x
            b = x + z
            c = y + z
            return c, b, a

        @sy.func2plan(args_shape=[(1,), (1,), (1,)])
        def foo_2(x, y, z):
            a = x + x
            b = x + z
            return b, a

        @sy.func2plan(args_shape=[(1,), (1,)])
        def avg_plan(x, y):
            result = x + y / 2
            return result

        # Plan Model
        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(2, 3)
                self.fc2 = nn.Linear(3, 2)
                self.fc3 = nn.Linear(2, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                return F.log_softmax(x, dim=0)

        model = Net()
        model.build(th.tensor([1.0, 2]))

        # Serialize plans / protocols and model
        serialized_plan_method_1 = binascii.hexlify(serialize(foo_1)).decode()
        serialized_plan_method_2 = binascii.hexlify(serialize(foo_2)).decode()
        serialized_avg_plan = binascii.hexlify(serialize(avg_plan)).decode()
        serialized_plan_model = binascii.hexlify(serialize(model)).decode()

        # As mentioned at federated learning roadmap.
        # We're supposed to set up client / server configs
        client_config = {
            "name": "my-federated-model",
            "version": "0.1.0",
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "SGD",
        }

        server_config = {
            "max_workers": 100,
            "pool_selection": "random",  # or "iterate"
            "num_cycles": 5,
            "do_not_reuse_workers_until_cycle": 4,
            "cycle_length": 8 * 60 * 60,  # 8 hours
            "minimum_upload_speed": 2,  # 2 mbps
            "minimum_download_speed": 4,  # 4 mbps
        }

        # "federated/host-training" request body
        host_training_message = {
            "type": "federated/host-training",
            "data": {
                "model": serialized_plan_model,
                "plans": {
                    "foo_1": serialized_plan_method_1,
                    "foo_2": serialized_plan_method_2,
                },
                "protocols": {"protocol_1": "serialized_protocol_mockup"},
                "averaging_plan": serialized_avg_plan,
                "client_config": client_config,
                "server_config": server_config,
            },
        }

        # Send host_training message
        response = await send_ws_message(host_training_message)
        self.assertEqual(response, {"status": "success"})

        """ 2 - Authentication Request """

        # "federated/authenticate" request body
        auth_msg = {"type": "federated/authenticate"}

        # Send worker authentication message
        response = await send_ws_message(auth_msg)
        self.assertEqual(response["status"], "success")
        worker_id = response.get("worker_id", None)

        assert worker_id != None

        """ 3 - Cycle Request """
        # "federated/cycle-request" request body
        req_cycle_msg = {
            "worker_id": worker_id,
            "model": "my-federated-model",
            "version": "0.1.0",
            "ping": 8,
            "download": 46.3,
            "upload": 23.7,
        }

        message = {"type": "federated/cycle-request", "data": req_cycle_msg}

        # Send worker authentication message
        response = await send_ws_message(message)
        self.assertEqual(response["status"], "accepted")

        response_fields = [
            "request_key",
            "model",
            "plans",
            "protocols",
            "client_config",
        ]

        # Check if response fields are empty
        for field in response_fields:
            assert response.get(field, None) != None

    @pytest.mark.skip
    async def test_type_get_protocol(self):
        response = await get_protocol()
        message_type = response.get("type")
        self.assertEqual(message_type, "get-protocol")

    @pytest.mark.skip
    async def test_get_protocol_creator_role(self):
        response = await get_protocol()

        user = get_user(response)
        role = user.get("role")

        self.assertEqual(role, "creator")

    @pytest.mark.skip
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

    @pytest.mark.skip
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

    @pytest.mark.skip
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

    @pytest.mark.skip
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

    @pytest.mark.skip
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

    @pytest.mark.skip
    async def test_invalid_message_type(self):
        response = await send_ws_message(
            {"type": "not-a-type", "data": {"protocol-id": "test-protocol"}}
        )
        self.assertEqual(response, {"error": "Invalid JSON format/field!"})
