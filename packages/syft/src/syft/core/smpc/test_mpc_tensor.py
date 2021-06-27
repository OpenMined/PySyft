# third party
import numpy as np
import torch

# syft absolute
import syft as sy
from syft.core.tensor.mpc_tensor import MPCTensor
from syft.core.tensor.tensor import Tensor

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

remote_value = alice_client.syft.core.tensor.tensor.Tensor(np.array([1, 2, 3, 4, -5]))

seeds_przs_generators = [42, 32]
mpc_tensor = MPCTensor(
    parties=[alice_client, bob_client],
    secret=remote_value,
    shape=(1, 5),
    seeds_przs_generators=seeds_przs_generators,
)
print(mpc_tensor)


print(mpc_tensor.reconstruct())
