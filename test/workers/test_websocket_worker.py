import time

import torch

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def test_websocket_worker(hook, start_proc):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker"""

    kwargs = {"id": "fed1", "host": "localhost", "port": 8766, "hook": hook}
    server = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)
    x = torch.ones(5)

    socket_pipe = WebsocketClientWorker(**kwargs)

    x = x.send(socket_pipe)
    y = x + x
    y = y.get()

    assert (y == torch.ones(5) * 2).all()

    del x

    socket_pipe.ws.shutdown()
    time.sleep(0.1)
    server.terminate()


def test_websocket_workers_search(hook, start_proc):
    """Evaluates that a client can search and find tensors that belong
    to another party"""
    # Sample tensor to store on the server
    sample_data = torch.tensor([1, 2, 3, 4]).tag("#sample_data", "#another_tag")
    # Args for initializing the websocket server and client
    base_kwargs = {"id": "fed2", "host": "localhost", "port": 8767, "hook": hook}
    server_kwargs = base_kwargs
    server_kwargs["data"] = [sample_data]
    server = start_proc(WebsocketServerWorker, server_kwargs)

    time.sleep(0.1)

    client_worker = WebsocketClientWorker(**base_kwargs)

    # Search for the tensor located on the server by using its tag
    results = client_worker.search("#sample_data", "#another_tag")

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    # Search multiple times should still work
    results = client_worker.search("#sample_data", "#another_tag")

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    client_worker.ws.shutdown()
    client_worker.ws.close()
    time.sleep(0.1)
    server.terminate()
