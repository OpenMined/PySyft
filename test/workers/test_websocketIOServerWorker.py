import time

import torch

import syft as sy
from grid.workers import WebsocketIOServerWorker


def _payload(location):
    x = torch.tensor([10, 20, 30, 40, 50.0])
    x.send(location)


hook = sy.TorchHook(torch)
server_worker = WebsocketIOServerWorker(hook, "localhost", 5000, log_msgs=True, payload=_payload)


def test_client_id():
    android = server_worker.socketio.test_client(server_worker.app)
    android.emit("client_id", "android")
    assert len(server_worker.clients) == 1
    android.disconnect()
    server_worker.terminate()


def test_payload_execution():
    android = server_worker.socketio.test_client(server_worker.app)
    android.emit("client_id", "android")
    time.sleep(0.1)
    android.emit("client_ack", "Android")
    time.sleep(0.3)
    android.emit("client_ack", "Android")
    time.sleep(0.3)
    assert server_worker.response_from_client == "ACK"
    assert not server_worker.wait_for_client_event

    android.disconnect()
    server_worker.terminate()
