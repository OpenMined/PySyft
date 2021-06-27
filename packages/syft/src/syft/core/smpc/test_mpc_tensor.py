# third party
import numpy as np
import syft as sy
import torch

# syft absolute
from syft.core.tensor.mpc_tensor import MPCTensor
from syft.core.tensor.tensor import Tensor

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

remote_value = alice_client.syft.core.tensor.tensor.Tensor([1, 2, 3, 4, -5])


mpc_tensor = MPCTensor(parties=[alice_client, bob_client], secret=remote_value, shape=(5, ))
print(mpc_tensor)
