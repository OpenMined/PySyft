import pytest
import time

import syft as sy
import torch as th
from unittest import mock
from types import MethodType

from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker


def test_create_already_existing_worker(hook):
    # Shares tensor with bob
    bob = sy.VirtualWorker(hook, "bob")
    x = th.tensor([1, 2, 3]).send(bob)

    # Recreates bob and shares a new tensor
    bob = sy.VirtualWorker(hook, "bob")
    y = th.tensor([2, 2, 2]).send(bob)

    # Recreates bob and shares a new tensor
    bob = sy.VirtualWorker(hook, "bob")
    z = th.tensor([2, 2, 10]).send(bob)

    # Both workers should be the same, so the following operation should be valid
    _ = x + y * z


def test_clear_object_for_worker_created_with_pre_existing_id(hook):

    worker = sy.VirtualWorker(hook, id="worker")
    worker.clear_objects()

    ptr = th.tensor([1, 2, 3]).send(worker)

    assert len(worker._known_workers[worker.id]._objects) == len(worker._objects)
    assert len(worker._objects) == 1

    # create worker with pre-existing id
    worker = sy.VirtualWorker(hook, id="worker")
    worker.clear_objects()

    assert len(worker._known_workers[worker.id]._objects) == len(worker._objects)
    assert len(worker._objects) == 0

    ptr = th.tensor([1, 2, 3]).send(worker)

    assert len(worker._known_workers[worker.id]._objects) == len(worker._objects)
    assert len(worker._objects) == 1


def test_create_already_existing_worker_with_different_type(hook, start_proc):
    # Shares tensor with bob
    bob = sy.VirtualWorker(hook, "bob")
    _ = th.tensor([1, 2, 3]).send(bob)

    kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}
    server = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)

    # Recreates bob as a different type of worker
    kwargs = {"id": "bob", "host": "localhost", "port": 8765, "hook": hook}
    with pytest.raises(RuntimeError):
        bob = WebsocketClientWorker(**kwargs)

    server.terminate()


def test_execute_worker_function(hook):
    sy.VirtualWorker.mocked_function = MethodType(
        mock.Mock(return_value="bob_mocked_function"), sy.VirtualWorker
    )

    bob = sy.VirtualWorker(hook, "bob")
    x = th.tensor([1, 2, 3]).send(bob)

    message = bob.create_execute_worker_function_message(
        command_name="mocked_function", command_owner="self"
    )

    serialized_message = sy.serde.serialize(message)

    response = bob._recv_msg(serialized_message)
    response = sy.serde.deserialize(response)

    assert response == "bob_mocked_function"

    bob.mocked_function.assert_called()


def test_enable_registration_with_ctx(hook):
    assert hook.local_worker.is_client_worker == True
    with hook.local_worker.registration_enabled():
        hook.local_worker.is_client_worker == False
    assert hook.local_worker.is_client_worker == True


def test_send_command_whitelist(hook, workers):
    bob = workers["bob"]
    whitelisted_methods = {
        "torch": {"tensor": [1, 2, 3], "rand": (2, 3), "randn": (2, 3), "zeros": (2, 3)}
    }

    for framework, methods in whitelisted_methods.items():
        attr = getattr(bob.remote, framework)

        for method, inp in methods.items():
            x = getattr(attr, method)(inp)

            if "rand" not in method:
                assert (x.get() == getattr(th, method)(inp)).all()


def test_send_command_not_whitelisted(hook, workers):
    bob = workers["bob"]

    method_not_exist = "openmind"

    for framework in bob.remote.frameworks:
        if framework in dir(bob.remote):
            attr = getattr(bob.remote, framework)

            with pytest.raises(AttributeError):
                getattr(attr, method_not_exist)
