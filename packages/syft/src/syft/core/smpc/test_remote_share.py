# third party
import numpy as np
import torch

# syft absolute
import syft as sy
from syft.core.tensor.share_tensor import ShareTensor
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.tensor import Tensor

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

import pdb

pdb.set_trace()
# value = Tensor(torch.Tensor([[1, 2, 3], [4, 5, 6]])).share(alice_client, bob_client)
