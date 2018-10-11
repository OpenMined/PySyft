### SecureNN Tests
import syft as sy
from syft.spdz import spdz
from syft.mpc.securenn import decompose, private_compare
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
        self.alice = sy.VirtualWorker(
            id="alice", hook=self.hook, is_client_worker=False
        )
        self.workers = (self.alice, self.bob)

        me.add_workers([self.bob, self.alice])
        self.bob.add_workers([me, self.alice])
        self.alice.add_workers([me, self.bob])

    def generate_mpc_number_pair(self, n1, n2, workers):
        mpcs = []
        for i in [n1, n2]:
            mpcs.append(torch.LongTensor([i]).share(*workers))
        return mpcs

    def prep_decompose(self):
        x = np.array([2 ** 32 + 3, 2 ** 30 - 1, 2 ** 30, -3])
        expected = np.array(
            [
                list(reversed(np.binary_repr(3, width=31))),
                list(reversed(np.binary_repr(2 ** 30 - 1, width=31))),
                list(reversed(np.binary_repr(2 ** 30, width=31))),
                list(reversed(np.binary_repr(-3, width=31))),
            ]
        ).astype("int")
        tensor = torch.LongTensor(x)
        expected = torch.LongTensor(expected)
        return expected, tensor

    def test_decompose(self):
        expected, tensor = self.prep_decompose()
        bin = decompose(tensor)
        assert (bin == expected).all()

    # def test_select_shares(self):
    # a, b = self.generate_mpc_number_pair(5, 10, self.workers)
    # abit, bbit = self.generate_mpc_number_pair(0, 1, self.workers)

    # za = select_shares(abit, a, b, self.workers)
    # zb = select_shares(bbit, a, b, self.workers)
    #
    # assert za.get()[0] == 5
    # assert zb.get()[0] == 10

    def prepPC(self):
        x = torch.LongTensor([1, 2, 3, 4])
        x_bit = decompose(x)
        x_bit_sh = x_bit.share(*self.workers)

        r = torch.ones(x.get_shape()).long().send(*self.workers) + 2

        j0 = torch.zeros(x_bit_sh.get_shape()).long().send(self.workers[0]).child
        j1 = (torch.ones(x_bit_sh.get_shape())).long().send(self.workers[1]).child
        j = sy._GeneralizedPointerTensor(
            {self.workers[0]: j0, self.workers[1]: j1}, torch_type="syft.LongTensor"
        ).wrap(True)

        # 4)
        BETA = torch.zeros(x.get_shape()).long().send(*self.workers)

        return x_bit_sh, r, BETA, j

    def test_private_compare(self):
        x_bit_sh, r, BETA, j = self.prepPC()

        BETA_prime = private_compare(
            x_bit_sh, r, BETA=BETA, j=j, alice=self.workers[0], bob=self.workers[1]
        ).long()

        assert (BETA_prime == torch.LongTensor([1, 1, 1, 0])).all()


#
#
# if __name__ == '__main__':
#     unittest.main()
