# third party
import numpy as np
import torch

# syft absolute
import syft as sy
from syft.core.tensor.mpc_tensor import MPCTensor
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.share_tensor import ShareTensor
from syft.core.tensor.tensor import Tensor

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

# stdlib
import pdb

pdb.set_trace()
share = alice_client.syft.core.tensor.tensor.Tensor([[1, 2, 3], [4, 5, 6]])
print(share)
res = MPCTensor(secret=share, parties=[alice_client, bob_client], shape=(2, 3))
