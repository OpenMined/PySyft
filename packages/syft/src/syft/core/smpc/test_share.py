# third party
import numpy as np

# syft absolute
from syft.core.tensor.share_tensor import ShareTensor

value = np.array([1, 2, 3, 4, -5])


share = ShareTensor(rank=0, value=value)
print(share)

seeds = [0, 1, 2]

share_1 = ShareTensor(rank=0, value=None, seed_generators=seeds[:2])
share_2 = ShareTensor(rank=1, value=value, seed_generators=seeds[1:])

print(share_1)
print(share_2)

# share_1.generate_przs(shape=value.shape)
# share_2.generate_przs(shape=value.shape)

print(share_1)
print(share_2)
