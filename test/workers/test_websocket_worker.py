import time

import torch

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def test_websocket_worker(hook, start_proc):
    """Evaluates that you can do basic tensor operations using
    WebsocketServerWorker"""

    kwargs = {"id": "fed1", "host": "localhost", "port": 8766, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)
    x = torch.ones(5)

    local_worker = WebsocketClientWorker(**kwargs)

    x = x.send(local_worker)
    y = x + x
    y = y.get()

    assert (y == torch.ones(5) * 2).all()

    del x

    local_worker.remove_worker_from_local_worker_registry()
    local_worker.ws.shutdown()
    time.sleep(0.1)
    process_remote_worker.terminate()


def test_websocket_workers_search(hook, start_proc):
    """Evaluates that a client can search and find tensors that belong
    to another party"""
    # Sample tensor to store on the server
    sample_data = torch.tensor([1, 2, 3, 4]).tag("#sample_data", "#another_tag")
    # Args for initializing the websocket server and client
    base_kwargs = {"id": "fed2", "host": "localhost", "port": 8767, "hook": hook}
    server_kwargs = base_kwargs
    server_kwargs["data"] = [sample_data]
    process_remote_worker = start_proc(WebsocketServerWorker, server_kwargs)

    time.sleep(0.1)

    local_worker = WebsocketClientWorker(**base_kwargs)

    # Search for the tensor located on the server by using its tag
    results = local_worker.search("#sample_data", "#another_tag")

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    # Search multiple times should still work
    results = local_worker.search("#sample_data", "#another_tag")

    assert results
    assert results[0].owner.id == "me"
    assert results[0].location.id == "fed2"

    local_worker.remove_worker_from_local_worker_registry()
    local_worker.ws.shutdown()
    time.sleep(0.1)
    process_remote_worker.terminate()


def test_list_objects_remote(hook, start_proc):

    kwargs = {"id": "fed", "host": "localhost", "port": 8765, "hook": hook}
    process_remote_fed1 = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)

    kwargs = {"id": "fed", "host": "localhost", "port": 8765, "hook": hook}
    local_worker = WebsocketClientWorker(**kwargs)

    x = torch.tensor([1, 2, 3]).send(local_worker)

    res = local_worker.list_objects_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 1

    y = torch.tensor([4, 5, 6]).send(local_worker)
    res = local_worker.list_objects_remote()
    res_dict = eval(res.replace("tensor", "torch.tensor"))
    assert len(res_dict) == 2

    # delete x before terminating the websocket connection
    del x
    del y
    time.sleep(0.1)
    local_worker.ws.shutdown()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_fed1.terminate()


def test_objects_count_remote(hook, start_proc):

    kwargs = {"id": "fed", "host": "localhost", "port": 8764, "hook": hook}
    process_remote_worker = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)

    kwargs = {"id": "fed", "host": "localhost", "port": 8764, "hook": hook}
    local_worker = WebsocketClientWorker(**kwargs)

    x = torch.tensor([1, 2, 3]).send(local_worker)

    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 1

    y = torch.tensor([4, 5, 6]).send(local_worker)
    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 2

    x.get()
    nr_objects = local_worker.objects_count_remote()
    assert nr_objects == 1

    # delete remote object before terminating the websocket connection
    del y
    time.sleep(0.1)
    local_worker.ws.shutdown()
    time.sleep(0.1)
    local_worker.remove_worker_from_local_worker_registry()
    process_remote_worker.terminate()
