"""All the tests relative to garbage collection of all kinds of remote or local tensors"""
import time

import torch

from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.workers import WebsocketServerWorker
from syft.workers import WebsocketClientWorker

# TESTING POINTERS


def test_explicit_garbage_collect_pointer(workers):
    """Tests whether deleting a PointerTensor garbage collects the remote object too"""
    bob = workers["bob"]

    alice, bob = workers["alice"], workers["bob"]

    # create tensor
    x = torch.Tensor([1, 2])

    # send tensor to bob
    x_ptr = x.send(bob)

    # ensure bob has tensor
    assert x.id in bob._objects

    # delete pointer to tensor, which should
    # automatically garbage collect the remote
    # object on Bob's machine
    del x_ptr

    # ensure bob's object was garbage collected
    assert x.id not in bob._objects


def test_explicit_garbage_collect_double_pointer(workers):
    """Tests whether deleting a pointer to a pointer garbage collects
    the remote object too"""

    alice, bob = workers["alice"], workers["bob"]

    # create tensor
    x = torch.Tensor([1, 2])

    # send tensor to bob and then pointer to alice
    x_ptr = x.send(bob)
    x_ptr_ptr = x_ptr.send(alice)

    # ensure bob has tensor
    assert x.id in bob._objects

    # delete pointer to pointer to tensor, which should automatically
    # garbage collect the remote object on Bob's machine
    del x_ptr_ptr

    # ensure bob's object was garbage collected
    assert x.id not in bob._objects
    # TODO: shouldn't we check that alice's object was
    # garbage collected as well?
    # assert x.id not in workers["alice"]._objects

    # Chained version
    x = torch.Tensor([1, 2])
    x_id = x.id
    # send tensor to bob and then pointer to alice
    x = x.send(bob).send(alice)
    # ensure bob has tensor
    assert x_id in bob._objects
    # delete pointer to pointer to tensor
    del x
    # ensure bob's object was garbage collected
    assert x_id not in bob._objects
    # TODO: shouldn't we check that alice's object was
    # garbage collected as well?
    # assert x.id not in workers["alice"]._objects


def test_implicit_garbage_collection_pointer(workers):
    """Tests whether GCing a PointerTensor GCs the remote object too."""
    bob = workers["bob"]

    alice, bob = workers["alice"], workers["bob"]

    # create tensor
    x = torch.Tensor([1, 2])

    # send tensor to bob
    x_ptr = x.send(bob)

    # ensure bob has tensor
    assert x.id in bob._objects

    # delete pointer to tensor, which should
    # automatically garbage collect the remote
    # object on Bob's machine
    x_ptr = "asdf"

    # ensure bob's object was garbage collected
    assert x.id not in bob._objects


def test_implicit_garbage_collect_double_pointer(workers):
    """Tests whether GCing a pointer to a pointer garbage collects
    the remote object too"""

    alice, bob = workers["alice"], workers["bob"]

    # create tensor
    x = torch.Tensor([1, 2])

    # send tensor to bob and then pointer to alice
    x_ptr = x.send(bob)
    x_ptr_ptr = x_ptr.send(alice)

    # ensure bob has tensor
    assert x.id in bob._objects

    # delete pointer to pointer to tensor, which should automatically
    # garbage collect the remote object on Bob's machine
    x_ptr_ptr = "asdf"

    # ensure bob's object was garbage collected
    assert x.id not in bob._objects
    # TODO: shouldn't we check that alice's object was
    # garbage collected as well?
    # assert x.id not in alice._objects

    # Chained version
    x = torch.Tensor([1, 2])
    x_id = x.id
    # send tensor to bob and then pointer to alice
    x = x.send(bob).send(alice)
    # ensure bob has tensor
    assert x_id in bob._objects
    # delete pointer to pointer to tensor
    x = "asdf"
    # ensure bob's object was garbage collected
    assert x_id not in bob._objects
    # TODO: shouldn't we check that alice's object was
    # garbage collected as well?
    # assert x.id not in alice._objects


# TESTING IN PLACE METHODS


def test_inplace_method_on_pointer(workers):
    bob = workers["bob"]

    tensor = torch.tensor([[1.0, 2], [4.0, 2]])
    pointer = tensor.send(bob)
    pointer.add_(pointer)
    tensor_back = pointer.get()
    assert (tensor * 2 == tensor_back).all()


# TESTING LOGGING TENSORS


def test_explicit_garbage_collect_logging_on_pointer(workers):
    """
    Tests whether deleting a LoggingTensor on a PointerTensor
    garbage collects the remote object too
    """
    bob = workers["bob"]

    x = torch.Tensor([1, 2])
    x_id = x.id

    x = x.send(bob)
    x = LoggingTensor().on(x)
    assert x_id in bob._objects

    del x

    assert x_id not in bob._objects


def test_implicit_garbage_collect_logging_on_pointer(workers):
    """
    Tests whether GCing a LoggingTensor on a PointerTensor
    garbage collects the remote object too
    """
    bob = workers["bob"]

    x = torch.Tensor([1, 2])
    x_id = x.id

    x = x.send(bob)
    x = LoggingTensor().on(x)
    assert x_id in bob._objects

    x = "open-source"
    assert x_id not in bob._objects


def test_websocket_garbage_collection(hook, start_proc):
    # Args for initializing the websocket server and client
    base_kwargs = {"id": "ws_gc", "host": "localhost", "port": 8555, "hook": hook}
    server = start_proc(WebsocketServerWorker, **base_kwargs)

    time.sleep(0.1)
    client_worker = WebsocketClientWorker(**base_kwargs)

    sample_data = torch.tensor([1, 2, 3, 4])
    sample_ptr = sample_data.send(client_worker)

    _ = sample_ptr.get()
    assert sample_data not in client_worker._objects

    client_worker.ws.shutdown()
    client_worker.ws.close()
    time.sleep(0.1)
    server.terminate()
