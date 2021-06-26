# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.tensor.share_tensor import ShareTensor

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")


alice_client = alice.get_client()
bob_client = bob.get_client()

orig_value = np.array([[1, 2, 3], [4, 5, 6]])

# stdlib
import pdb; pdb.set_trace()
shares = ShareTensor.generate_shares(secret=orig_value, nr_shares=2)
