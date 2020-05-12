import json
import pytest
import binascii
import websockets
import aiounittest
from uuid import UUID

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import syft as sy

from grid.app.main.syft_assets.plan_manager import PlanManager
from grid.app.main.models.model_manager import ModelManager

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
        serialized_plan_method_1 = binascii.hexlify(
            PlanManager.serialize_plan(foo_1)
        ).decode()
        serialized_plan_method_2 = binascii.hexlify(
            PlanManager.serialize_plan(foo_2)
        ).decode()
        serialized_avg_plan = binascii.hexlify(
            PlanManager.serialize_plan(avg_plan)
        ).decode()
        serialized_plan_model = binascii.hexlify(
            ModelManager.serialize_model_params(model.parameters())
        ).decode()
        serialized_protocol_mockup = binascii.hexlify(
            "serialized_protocol_mockup".encode("utf-8")
        ).decode()

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
                "protocols": {"protocol_1": serialized_protocol_mockup},
                "averaging_plan": serialized_avg_plan,
                "client_config": client_config,
                "server_config": server_config,
            },
        }

        # Send host_training message
        response = await send_ws_message(host_training_message)
        self.assertEqual(response["data"], {"status": "success"})

        """ 2 - Authentication Request """

        # "federated/authenticate" request body
        auth_msg = {"type": "federated/authenticate"}

        # Send worker authentication message
        response = await send_ws_message(auth_msg)
        self.assertEqual(response["data"]["status"], "success")
        worker_id = response["data"].get("worker_id", None)

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
        self.assertEqual(response["data"]["status"], "accepted")

        response_fields = [
            "request_key",
            "model",
            "version",
            "plans",
            "protocols",
            "client_config",
        ]

        # Check if response fields are empty
        for field in response_fields:
            assert response["data"].get(field, None) != None
