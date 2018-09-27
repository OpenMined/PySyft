import torch
from syft.spdz import spdz
from syft.core.frameworks.torch import _GeneralizedPointerTensor, _SPDZTensor


def _generate_mpc_number_pair(self, n1, n2):
    mpcs = []
    for i in [n1, n2]:
        x = torch.LongTensor([i])
        enc = spdz.encode(x)
        x_alice, x_bob = spdz.share(enc, 2)
        x_alice.send(self.alice)
        x_bob.send(self.bob)
        x_pointer_tensor_dict = {self.alice: x_alice.child, self.bob: x_bob.child}
        x_gp = _GeneralizedPointerTensor(x_pointer_tensor_dict).on(x)
        mpcs.append(_SPDZTensor(x_gp))
    return mpcs
