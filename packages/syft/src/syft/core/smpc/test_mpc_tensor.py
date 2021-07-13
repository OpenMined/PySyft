# stdlib
import sys

# third party
import numpy as np
import torch

# syft absolute
import syft as sy
from syft import logger
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
from syft.core.tensor.tensor import Tensor

logger.remove()

vms = [sy.VirtualMachine(name=name) for name in ["alice", "bob", "theo", "andrew"]]
clients = [vm.get_client() for vm in vms]

# One of the clients hold the secret
remote_value_1 = clients[0].syft.core.tensor.tensor.Tensor(
    np.array([[1, 2, 3, 4, -5]], dtype=np.int64)
)

mpc_tensor_1 = MPCTensor(
    parties=clients, secret=remote_value_1, shape=(1, 5), seed_shares=52
)
print(mpc_tensor_1)

# One of the clients hold the secret
remote_value_2 = clients[1].syft.core.tensor.tensor.Tensor(
    np.array([[5]], dtype=np.int64)
)

mpc_tensor_2 = MPCTensor(
    parties=clients,
    secret=remote_value_2,
    shape=(1, 1),
    seed_shares=42,
)

print(mpc_tensor_2)

res = mpc_tensor_1 + mpc_tensor_2
print("Reconstructed value", res.reconstruct())


public_value = Tensor(np.array([[3]], dtype=np.int64))

res = mpc_tensor_1 * public_value
print("Reconstructed value", res.reconstruct())
