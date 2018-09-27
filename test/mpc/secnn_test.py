### SecureNN Tests
import syft as sy
from syft.spdz import spdz
from syft.mpc.securenn import decompose, select_shares, private_compare
from syft.core.frameworks.torch.tensor import _GeneralizedPointerTensor, _SPDZTensor
from .test_utils import _generate_mpc_number_pair

import unittest
import numpy as np
import torch
import importlib

class TestSecureNN(unittest.TestCase):
    def setUp(self):
        self.hook = sy.TorchHook(verbose=True)

        me = self.hook.local_worker
        me.is_client_worker = False

        self.bob = sy.VirtualWorker(id="bob", hook=self.hook, is_client_worker=False)
        self.alice = sy.VirtualWorker(id="alice", hook=self.hook, is_client_worker=False)

        me.add_workers([self.bob, self.alice])
        self.bob.add_workers([me, self.alice])
        self.alice.add_workers([me, self.bob])

    def generate_mpc_number_pair(self, n1, n2):
        return _generate_mpc_number_pair(self, n1, n2)

    def prep_decompose(self):
        x = np.array([2 ** 32 + 3,
                      2 ** 30 - 1,
                      2 ** 30,
                      -3])
        expected = np.array([
            list(reversed(np.binary_repr(3, width=31))),
            list(reversed(np.binary_repr(2 ** 30 - 1, width=31))),
            list(reversed(np.binary_repr(2 ** 30, width=31))),
            list(reversed(np.binary_repr(-3, width=31)))
        ]).astype('int')
        tensor = torch.LongTensor(x)
        expected = torch.LongTensor(expected)
        return expected, tensor

    def test_decompose(self):
        expected, tensor = self.prep_decompose()
        bin = decompose(tensor)
        assert (bin == expected).all()

    def test_select_shares(self):
        workers = (self.alice, self.bob)
        a, b = self.generate_mpc_number_pair(5, 10)
        abit, bbit = self.generate_mpc_number_pair(0, 1)
        za = select_shares(abit, a, b, workers)
        zb = select_shares(bbit, a, b, workers)
        print('za', za)
        print('zb', zb)
        assert za == a
        assert zb == b


if __name__ == '__main__':
    unittest.main()
