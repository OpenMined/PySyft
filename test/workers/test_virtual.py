from syft.workers.virtual import VirtualWorker
from syft.workers.base import MSGTYPE_OBJ
from syft.workers.base import MSGTYPE_OBJ_REQ
from syft import serde
from unittest import TestCase
import numpy


class TestVirtualWorker(TestCase):
    def test_send_msg(self):
        bob = VirtualWorker()
        alice = VirtualWorker()
        obj = numpy.random.random([100, 100])
        obj_data = (1, obj)

        # Send data to alice
        msg = (MSGTYPE_OBJ, obj_data)
        bin_msg = serde.serialize(msg)
        alice.recv_msg(bin_msg)

        # Get data from alice
        bob.send_msg(MSGTYPE_OBJ_REQ, 1, alice)

    def test_recv_msg(self):
        alice = VirtualWorker()
        obj = numpy.random.random([100, 100])
        obj_data = (1, obj)

        # Send data to alice
        msg = (MSGTYPE_OBJ, obj_data)
        bin_msg = serde.serialize(msg)
        alice.recv_msg(bin_msg)

        # Get data from alice
        msg = (MSGTYPE_OBJ_REQ, 1)
        bin_msg = serde.serialize(msg)
        resp = alice.recv_msg(bin_msg)

        assert type(resp) == bytes

    def test_convenience_methods(self):
        bob = VirtualWorker()
        alice = VirtualWorker()
        obj = numpy.random.random([100, 100])
        obj_data = (1, obj)

        # Send data to alice
        bob.send_obj(obj_data, alice)

        # Get data from alice
        resp_alice = bob.request_obj(1, alice)

        assert (resp_alice == obj).all()

        # Set data on self
        bob.set_obj(obj_data)

        # Get data from self
        resp_bob_self = bob.get_obj(1)

        assert (resp_bob_self == obj).all()

        # Get data from bob as alice
        resp_bob_alice = alice.request_obj(1, bob)

        assert (resp_bob_alice == obj).all()
