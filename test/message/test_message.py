import syft as sy
import torch as th

import syft as sy
from syft.messaging import message


def test_cmd_message(workers):

    bob = workers["bob"]

    bob.log_msgs = True

    x = th.tensor([1, 2, 3, 4]).send(bob)

    y = x + x  # this is the test
    assert isinstance(bob._get_msg(-1), message.TensorCommandMessage)

    y = y.get()

    assert (y == th.tensor([2, 4, 6, 8])).all()

    bob.log_msgs = False


def test_obj_message(workers):

    bob = workers["bob"]

    bob.log_msgs = True

    x = th.tensor([1, 2, 3, 4]).send(bob)  # this is the test

    assert isinstance(bob._get_msg(-1), message.ObjectMessage)

    y = x + x

    y = y.get()

    assert (y == th.tensor([2, 4, 6, 8])).all()

    bob.log_msgs = False


def test_obj_req_message(workers):

    bob = workers["bob"]

    bob.log_msgs = True

    x = th.tensor([1, 2, 3, 4]).send(bob)

    y = x + x

    y = y.get()  # this is the test
    assert isinstance(bob._get_msg(-1), message.ObjectRequestMessage)

    assert (y == th.tensor([2, 4, 6, 8])).all()

    bob.log_msgs = False


def test_get_shape_message(workers):

    bob = workers["bob"]

    bob.log_msgs = True

    x = th.tensor([1, 2, 3, 4]).send(bob)

    y = x + x

    z = y.shape  # this is the test
    assert isinstance(bob._get_msg(-1), message.GetShapeMessage)

    assert z == th.Size([4])

    bob.log_msgs = False


def test_force_object_delete_message(workers):

    bob = workers["bob"]

    bob.log_msgs = True

    x = th.tensor([1, 2, 3, 4]).send(bob)

    id_on_worker = x.id_at_location

    assert id_on_worker in bob.object_store._objects

    del x  # this is the test
    assert isinstance(bob._get_msg(-1), message.ForceObjectDeleteMessage)

    assert id_on_worker not in bob.object_store._objects

    bob.log_msgs = False


def test_is_none_message(workers):

    bob = workers["bob"]

    bob.log_msgs = True

    x = th.tensor([1, 2, 3, 4]).send(bob)

    y = th.tensor([1]).send(bob)
    y.child.id_at_location = x.id_at_location

    assert not bob.request_is_remote_tensor_none(x)
    assert isinstance(bob._get_msg(-1), message.IsNoneMessage)
    assert not x.child.is_none()
    assert isinstance(bob._get_msg(-1), message.IsNoneMessage)

    del x

    assert y.child.is_none()

    bob.log_msgs = False
