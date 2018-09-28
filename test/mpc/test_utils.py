import torch
from syft.spdz import spdz
from syft.core.frameworks.torch import _GeneralizedPointerTensor, _SPDZTensor


def _generate_mpc_number_pair(self, n1, n2):
    mpcs = []
    for i in [n1, n2]:
        mpcs.append(torch.LongTensor([i]).share(self.bob, self.alice))

    return mpcs
