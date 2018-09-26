### SecureNN Tests
import syft as sy
from syft.mpc import spdz
from syft.mpc.securenn import decompose, select_shares, private_compare
from syft.core.frameworks.torch.tensor import _GeneralizedPointerTensor, _MPCTensor
from syft.mpc.test_utils import _generate_mpc_number_pair

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

    def prep_decompose(self, *shape):
        x = np.random.choice(spdz.field - 1, shape)
        f_bin = np.vectorize(np.binary_repr)
        true_bin = f_bin(x.astype(np.uint64), width=64)
        tensor = torch.LongTensor(x)
        return true_bin, tensor

    def test_decompose(self):
        true_bin, tensor = self.prep_decompose(3, 3, 3)
        bin = decompose(tensor)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bin_repr = ""
                    y = bin[i, j, k, :]
                    for n in range(len(y)):
                        bin_repr += str(y[n])
                    assert bin_repr[::-1] == true_bin[i, j, k][64 - spdz.Q_BITS:]

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
