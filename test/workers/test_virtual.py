from time import time
from unittest.mock import patch

import syft as sy
from syft.exceptions import GetNotPermittedError
from syft.workers.virtual import VirtualWorker
from syft.codes import MSGTYPE
from syft import messaging
from syft import serde

from syft.frameworks.torch import pointers

import pytest
import torch
import torch as th


def test_send_msg():
    """Tests sending a message with a specific ID

    This is a simple test to ensure that the BaseWorker interface
    can properly send/receive a message containing a tensor.
    """

    # get pointer to local worker
    me = sy.torch.hook.local_worker

    # create a new worker (to send the object to)
    worker_id = sy.ID_PROVIDER.pop()
    bob = VirtualWorker(sy.torch.hook, id=f"bob{worker_id}")

    # initialize the object and save it's id
    obj = torch.Tensor([100, 100])
    obj_id = obj.id

    # Send data to bob
    me.send_msg(MSGTYPE.OBJ, obj, bob)

    # ensure that object is now on bob's machine
    assert obj_id in bob._objects


def test_send_msg_using_tensor_api():
    """Tests sending a message with a specific ID

    This is a simple test to ensure that the high level tensor .send()
    method correctly sends a message to another worker.
    """

    # create worker to send object to
    worker_id = sy.ID_PROVIDER.pop()
    bob = VirtualWorker(sy.torch.hook, id=f"bob{worker_id}")

    # create a tensor to send (default on local_worker)
    obj = torch.Tensor([100, 100])

    # save the object's id
    obj_id = obj.id

    # send the object to Bob (from local_worker)
    _ = obj.send(bob)

    # ensure tensor made it to Bob
    assert obj_id in bob._objects


def test_recv_msg():
    """Tests the recv_msg command with 2 tests

    The first test uses recv_msg to send an object to alice.

    The second test uses recv_msg to request the object
    previously sent to alice."""

    # TEST 1: send tensor to alice

    # create a worker to send data to
    worker_id = sy.ID_PROVIDER.pop()
    alice = VirtualWorker(sy.torch.hook, id=f"alice{worker_id}")

    # create object to send
    obj = torch.Tensor([100, 100])

    # create/serialize message
    message = messaging.Message(MSGTYPE.OBJ, obj)
    bin_msg = serde.serialize(message)

    # have alice receive message
    alice.recv_msg(bin_msg)

    # ensure that object is now in alice's registry
    assert obj.id in alice._objects

    # Test 2: get tensor back from alice

    # Create message: Get tensor from alice
    message = messaging.Message(MSGTYPE.OBJ_REQ, obj.id)

    # serialize message
    bin_msg = serde.serialize(message)

    # call receive message on alice
    resp = alice.recv_msg(bin_msg)

    obj_2 = serde.deserialize(resp)

    # assert that response is correct type
    assert type(resp) == bytes

    # ensure that the object we receive is correct
    assert obj_2.id == obj.id


def tests_worker_convenience_methods():
    """Tests send and get object methods on BaseWorker

    This test comes in two parts. The first uses the simple
    BaseWorker.send_obj and BaseWorker.request_obj to send a
    tensor to Alice and to get the worker back from Alice.

    The second part shows that the same methods work between
    bob and alice directly.
    """

    me = sy.torch.hook.local_worker
    worker_id = sy.ID_PROVIDER.pop()
    bob = VirtualWorker(sy.torch.hook, id=f"bob{worker_id}")
    worker_id = sy.ID_PROVIDER.pop()
    alice = VirtualWorker(sy.torch.hook, id=f"alice{worker_id}")
    obj = torch.Tensor([100, 100])

    # Send data to alice
    me.send_obj(obj, alice)

    # Get data from alice
    resp_alice = me.request_obj(obj.id, alice)

    assert (resp_alice == obj).all()

    obj2 = torch.Tensor([200, 200])

    # Set data on self
    bob.set_obj(obj2)

    # Get data from self
    resp_bob_self = bob.get_obj(obj2.id)

    assert (resp_bob_self == obj2).all()

    # Get data from bob as alice
    resp_bob_alice = alice.request_obj(obj2.id, bob)

    assert (resp_bob_alice == obj2).all()


def test_search():
    worker_id = sy.ID_PROVIDER.pop()
    bob = VirtualWorker(sy.torch.hook, id=f"bob{worker_id}")

    x = (
        torch.tensor([1, 2, 3, 4, 5])
        .tag("#fun", "#mnist")
        .describe("The images in the MNIST training dataset.")
        .send(bob)
    )

    y = (
        torch.tensor([1, 2, 3, 4, 5])
        .tag("#not_fun", "#cifar")
        .describe("The images in the MNIST training dataset.")
        .send(bob)
    )

    z = (
        torch.tensor([1, 2, 3, 4, 5])
        .tag("#fun", "#boston_housing")
        .describe("The images in the MNIST training dataset.")
        .send(bob)
    )

    a = (
        torch.tensor([1, 2, 3, 4, 5])
        .tag("#not_fun", "#boston_housing")
        .describe("The images in the MNIST training dataset.")
        .send(bob)
    )

    assert len(bob.search("#fun")) == 2
    assert len(bob.search("#mnist")) == 1
    assert len(bob.search("#cifar")) == 1
    assert len(bob.search("#not_fun")) == 2
    assert len(bob.search("#not_fun", "#boston_housing")) == 1


def test_obj_not_found(workers):
    """Test for useful error message when trying to call a method on
    a tensor which does not exist on a worker anymore."""

    bob = workers["bob"]

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    bob._objects = {}

    try:
        y = x + x
    except KeyError as e:
        assert "If you think this tensor does exist" in str(e)


def test_get_not_permitted(workers):
    bob = workers["bob"]
    with patch.object(th.Tensor, "allowed_to_get") as mock_allowed_to_get:
        mock_allowed_to_get.return_value = False
        x = th.tensor([1, 2, 3, 4, 5]).send(bob)
        with pytest.raises(GetNotPermittedError):
            x.get()
        mock_allowed_to_get.assert_called_once()


def test_spinup_time(hook):
    """Tests to ensure that virtual workers intialized with 10000 data points
    load in under 0.05 seconds. This is needed to ensure that virtual workers
    spun up inside web frameworks are created quickly enough to not cause timeout errors"""
    data = []
    for i in range(10000):
        data.append(th.Tensor(5, 5).random_(100))
    start_time = time()
    dummy = sy.VirtualWorker(hook, id="dummy", data=data)
    end_time = time()
    assert (end_time - start_time) < 0.05


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_send_jit_scriptmodule(hook, workers):  # pragma: no cover
    bob = workers["bob"]

    @torch.jit.script
    def foo(x):
        return x + 2

    foo_wrapper = pointers.ObjectWrapper(obj=foo, id=99)
    foo_ptr = hook.local_worker.send(foo_wrapper, bob)

    res = foo_ptr(torch.tensor(4))
    assert res == torch.tensor(6)
