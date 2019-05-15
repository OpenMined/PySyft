import time

import socketio
import torch

from grid.workers import WebsocketIOServerWorker, WebsocketIOClientWorker


def create_dummy_client():
    sio = socketio.Client()
    sio.connect("http://localhost:5000")

    @sio.on("message")
    def on_message(args):
        sio.emit("client_ack", "android")

    return sio


def test_send_tensor(hook, start_proc):
    kwargs = {"host": "localhost", "port": 5000, "hook": hook, "id": "sever_worker_3"}
    server_worker = start_proc(WebsocketIOServerWorker, kwargs)
    time.sleep(0.1)

    android = create_dummy_client()

    juan = WebsocketIOClientWorker(hook, host="localhost", port=5000, id="juan", log_msgs=True)
    juan.connect()

    x = torch.tensor([1.0])
    x.send(juan)
    time.sleep(0.2)

    assert not juan.wait_for_client_event
    assert "ACK" == juan.response_from_client

    juan.disconnect()
    time.sleep(0.1)
    android.disconnect()
    time.sleep(0.1)
    server_worker.terminate()
