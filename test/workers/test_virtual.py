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
        obj = {"id": 1, "data": numpy.random.random([100, 100])}

        # Send data to alice
        msg = (MSGTYPE_OBJ, obj)
        bin_msg = serde.serialize(msg)
        alice.recv_msg(bin_msg)

        # Get data from alice
        bob.send_msg(MSGTYPE_OBJ_REQ, 1, alice)

    def test_recv_msg(self):
        alice = VirtualWorker()
        obj = {"id": 1, "data": numpy.random.random([100, 100])}

        # Send data to alice
        msg = (MSGTYPE_OBJ, obj)
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
        obj = {"id": 1, "data": numpy.random.random([100, 100])}

        # Send data to alice
        bob.send_obj(obj, alice)

        # Get data from alice
        bob.request_obj(1, alice)
