# stdlib
import sys

# third party
import numpy as np
import torch

# syft absolute
import syft as sy
from syft import logger
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
from syft.core.tensor.smpc.share_tensor import ShareTensor
from syft.core.tensor.tensor import Tensor

vms = [sy.VirtualMachine(name=name) for name in ["alice", "bob", "theo", "andrew"]]
clients = [vm.get_client() for vm in vms]


def test_remote_sharing():
    value = Tensor(np.array([[1, 2, 3, 4, -5]], dtype=np.int64))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(
        parties=clients, secret=remote_value, shape=value.shape, seed_shares=52
    )

    assert len(mpc_tensor.child) == len(clients)

    shares = [share.get_copy() for share in mpc_tensor.child]
    assert all([isinstance(share, ShareTensor) for share in shares])
    import pdb; pdb.set_trace()
    assert (mpc_tensor.reconstruct() == value).all()
