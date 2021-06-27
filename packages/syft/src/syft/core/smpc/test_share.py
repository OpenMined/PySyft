# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor
from syft.core.tensor.share_tensor import ShareTensor

value = np.array([1, 2, 3, 4, -5])


share = FixedPrecisionTensor()
share.child = ShareTensor(rank=0, value=value)
# stdlib
import pdb

pdb.set_trace()

# Test send
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_client()
print(share.send(alice_client))

seeds = [0, 1, 2]

share_1 = ShareTensor(rank=0, value=None, seed_generators=seeds[:2])
share_2 = ShareTensor(rank=1, value=value, seed_generators=seeds[1:])

print(share_1)
print(share_2)

# share_1.generate_przs(shape=value.shape)
# share_2.generate_przs(shape=value.shape)
