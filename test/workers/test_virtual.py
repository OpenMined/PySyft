import syft as sy

from syft.workers.virtual import VirtualWorker
from syft.workers.base import MSGTYPE_OBJ
from syft.workers.base import MSGTYPE_OBJ_REQ
from syft import serde

import numpy
import torch


def test_send_msg():
    """Tests sending a message with a specific ID

    This is a simple test to ensure that the BaseWorker interface
    can properly send/receive a message containing a tensor.
    """

    # get pointer to local worker
    me = sy.torch.hook.local_worker

    # create a new worker (to send the object to)
    bob = VirtualWorker()

    # initialize the object and save it's id
    obj = torch.Tensor([100, 100])
    obj_id = obj.id

    # Send data to bob
    me.send_msg(MSGTYPE_OBJ, obj, bob)

    # ensure that object is now on bob's machine
    assert obj_id in bob._objects


def test_send_msg_using_tensor_api():
    """Tests sending a message with a specific ID

    This is a simple test to ensure that the high level tensor .send()
    method correctly sends a message to another worker.
    """

    # create worker to send object to
    bob = VirtualWorker()

    # create a tensor to send (default on local_worker)
    obj = torch.Tensor([100, 100])

    # save the object's id
    obj_id = obj.id

    # send the object to Bob (from local_worker)
    obj_ptr = obj.send(bob)

    # ensure tensor made it to Bob
    assert obj_id in bob._objects


def test_recv_msg():
    """Tests the recv_msg command with 2 tests

    The first test uses recv_msg to send an object to alice.

    The second test uses recv_msg to request the object
    previously sent to alice."""

    # TEST 1: send tensor to alice

    # create a worker to send data to
    alice = VirtualWorker()

    # create object to send
    obj = torch.Tensor([100, 100])

    # create/serialize message
    msg = (MSGTYPE_OBJ, obj)
    bin_msg = serde.serialize(msg)

    # have alice receive message
    alice.recv_msg(bin_msg)

    # ensure that object is now in alice's registry
    assert obj.id in alice._objects

    # Test 2: get tensor back from alice

    # Create message: Get tensor from alice
    msg = (MSGTYPE_OBJ_REQ, obj.id)

    # serialize message
    bin_msg = serde.serialize(msg)

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
    bob = VirtualWorker()
    alice = VirtualWorker()
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
