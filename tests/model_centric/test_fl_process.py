import binascii
import json
from uuid import uuid4

import aiounittest
import pytest
import syft as sy
import torch as th
import websockets
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

from apps.node.src.app.main.model_centric.models.model_manager import ModelManager
from apps.node.src.app.main.model_centric.syft_assets.plan_manager import PlanManager
from tests import NETWORK_WS_URL

hook = sy.TorchHook(th)

# Useful functions


async def send_ws_message(data):
    try:
        async with websockets.connect(NETWORK_WS_URL) as websocket:
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
serialized_plan_method_1 = binascii.hexlify(PlanManager.serialize_plan(foo_1)).decode()
serialized_plan_method_2 = binascii.hexlify(PlanManager.serialize_plan(foo_2)).decode()
serialized_avg_plan = binascii.hexlify(PlanManager.serialize_plan(avg_plan)).decode()
serialized_plan_model = binascii.hexlify(
    ModelManager.serialize_model_params(model.parameters())
).decode()
serialized_protocol_mockup = binascii.hexlify(
    "serialized_protocol_mockup".encode("utf-8")
).decode()


class ModelCentricAPISocketsTest(aiounittest.AsyncTestCase):
    async def test_socket_ping_alive(self):
        response = await send_ws_message({"type": "socket-ping", "data": {}})
        self.assertEqual(response, {"alive": "True"})

    async def test_fl_process(self):
        """ 1 - Host Federated Training """

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
            "authentication": {
                "secret": "abc",
                "pub_key": """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0+rhzQe72Sef+wJuxoTO
Rx/nijb9PpPyb+Rgk0sNN4nB1wkNSKMlaHQkORWY/y5c8qlBF3/WlQUIQIAt1zP1
wM29GaaDuO3htRL9pjxwWdbX86Sl2CrjR1w0N2jaN+Bz9EZHYasd/0GJWbPTF7j5
JXrKRgvu+xB5wRRgZV/9gr/AzJHynPnDk95vcbEjPoTZ5dcv/UuMKngceZBex0Ea
ac+gPRWjh6FkXTiqedbKxrVcHD/72RdmBiTgTpu9a5DbA+vAIWIhj3zfvKQpUY1p
riWYMKALI61uc+NH0jr+B5/XTV/KlNqmbuEWfZdgRcXodNmIXt+LGHOQ1C+X+7OY
0wIDAQAB
-----END PUBLIC KEY-----
                    """.strip(),
            },
        }

        # "model-centric/host-training" request body
        host_training_message = {
            "type": "model-centric/host-training",
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

        # "model-centric/authenticate" request body
        request_id = str(uuid4())
        auth_msg = {
            "request_id": request_id,
            "type": "model-centric/authenticate",
            "data": {"model_name": "my-federated-model", "model_version": "0.1.0"},
        }

        # Send worker authentication message (no token!)
        response = await send_ws_message(auth_msg)
        self.assertEqual(
            response["data"]["error"],
            "Authentication is required, please pass an 'auth_token'.",
        )
        self.assertEqual(response["request_id"], request_id)
        worker_id = response["data"].get("worker_id", None)
        self.assertIsNone(worker_id)

        # Send worker authentication message (invalid token)
        auth_msg["data"]["auth_token"] = "just kidding!"
        response = await send_ws_message(auth_msg)
        self.assertEqual(
            response["data"]["error"], "The 'auth_token' you sent is invalid."
        )
        self.assertEqual(response["request_id"], request_id)
        worker_id = response["data"].get("worker_id", None)
        self.assertIsNone(worker_id)

        # Send worker authentication message (valid for secret)
        auth_msg["data"][
            "auth_token"
        ] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.yYhP2xosmpuyV5aoT8mz7GFESzq3hKSy-CRWC-vYOIU"
        response = await send_ws_message(auth_msg)
        self.assertEqual(response["data"]["status"], "success")
        self.assertEqual(response["request_id"], request_id)
        worker_id = response["data"].get("worker_id", None)
        self.assertIsNotNone(worker_id)

        # Send worker authentication message (valid for pub_key)
        auth_msg["data"][
            "auth_token"
        ] = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.e30.jOleZNk89aGMWhWVpV8UYul94y7rxBJAg4HnhY72y-DrLfxfhnR8b31FOMUcngxcw-N4MaSz5fulYFSTBt9NwIWWDUeAo0MqNMK-M6RRoxYd35k8SHNTIRAk0KnybKHMnTC4Qay3plXcu3FfMpOkX8Relpb8SUO3T1_B6RFqgNPO_l4KlmtXnxXgeFC86qF8b7fFCo8U1UKVUEbqw4JUCW5OmDnSmGxmb9felzASzuM5sO5MOkksuQ0DGVoi6AadhXQ5zB7k2Mj4fjJH7XyauHeuB2xjNM0jhoeR_DAoztvVEW5qx9fu2JfOiM6ZsBguCL7uKg1h1bQq278btHROpA"
        response = await send_ws_message(auth_msg)
        self.assertEqual(response["data"]["status"], "success")
        self.assertEqual(response["request_id"], request_id)
        worker_id = response["data"].get("worker_id", None)
        self.assertIsNotNone(worker_id)

        """ 3 - Cycle Request """
        # "model-centric/cycle-request" request body
        req_cycle_msg = {
            "worker_id": worker_id,
            "model": "my-federated-model",
            "version": "0.1.0",
            "ping": 8,
            "download": 46.3,
            "upload": 23.7,
        }

        request_id = str(uuid4())
        message = {
            "type": "model-centric/cycle-request",
            "request_id": request_id,
            "data": req_cycle_msg,
        }

        # Send worker authentication message
        response = await send_ws_message(message)
        self.assertEqual(response["data"]["status"], "accepted")
        self.assertEqual(response["request_id"], request_id)

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
            self.assertIsNotNone(response["data"].get(field, None))

    async def test_requires_speed_test_true(self):

        client_config = {
            "name": "my-federated-model-2",
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

        host_training_message = {
            "type": "model-centric/host-training",
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
        await send_ws_message(host_training_message)
        request_id = str(uuid4())
        auth_msg = {
            "type": "model-centric/authenticate",
            "request_id": request_id,
            "data": {"model_name": "my-federated-model-2", "model_version": "0.1.0"},
        }

        response = await send_ws_message(auth_msg)

        worker_id = response["data"].get("worker_id", None)
        requires_speed_test = response["data"].get("requires_speed_test")

        self.assertIsNotNone(worker_id)
        self.assertEqual(response["request_id"], request_id)
        self.assertTrue(requires_speed_test)

        # Speed must be required in cycle-request
        request_id = str(uuid4())
        cycle_req = {
            "type": "model-centric/cycle-request",
            "request_id": request_id,
            "data": {
                "worker_id": worker_id,
                "model": "my-federated-model-2",
                "version": "0.1.0",
            },
        }
        response = await send_ws_message(cycle_req)
        self.assertIsNotNone(response["data"].get("error"))
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(response["data"].get("status"), "rejected")

        # Should accept into cycle if all speed fields are sent
        request_id = str(uuid4())
        cycle_req = {
            "type": "model-centric/cycle-request",
            "request_id": request_id,
            "data": {
                "worker_id": worker_id,
                "model": "my-federated-model-2",
                "version": "0.1.0",
                "ping": 1,
                "download": 5,
                "upload": 5,
            },
        }
        response = await send_ws_message(cycle_req)
        self.assertIsNone(response["data"].get("error"))
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(response["data"].get("status"), "accepted")

    async def test_requires_speed_test_false(self):

        client_config = {
            "name": "my-federated-model-3",
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
        }

        host_training_message = {
            "type": "model-centric/host-training",
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
        await send_ws_message(host_training_message)
        request_id = str(uuid4())
        auth_msg = {
            "type": "model-centric/authenticate",
            "request_id": request_id,
            "data": {"model_name": "my-federated-model-3", "model_version": "0.1.0"},
        }

        response = await send_ws_message(auth_msg)
        worker_id = response["data"].get("worker_id", None)
        requires_speed_test = response["data"].get("requires_speed_test")

        self.assertIsNotNone(worker_id)
        self.assertFalse(requires_speed_test)
        self.assertEqual(response["request_id"], request_id)

        # Speed is not required in cycle-request
        request_id = str(uuid4())
        cycle_req = {
            "type": "model-centric/cycle-request",
            "request_id": request_id,
            "data": {
                "worker_id": worker_id,
                "model": "my-federated-model-3",
                "version": "0.1.0",
            },
        }
        response = await send_ws_message(cycle_req)
        self.assertIsNone(response["data"].get("error"))
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(response["data"].get("status"), "accepted")
