# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor
from syft.core.tensor.share_tensor import ShareTensor

vm = sy.VirtualMachine(name="alice")
client = vm.get_client()


value1 = np.array([1, 2, 3, 4, -5])

share1 = ShareTensor(rank=0, value=value1)
share1_ptr = share1.send(client)

value2 = np.array([100])
share2 = ShareTensor(rank=0, value=value2)
share2_ptr = share2.send(client)

share3_ptr = share1_ptr + share2_ptr
