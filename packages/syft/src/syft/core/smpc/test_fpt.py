# third party
import numpy as np

# syft absolute
import syft as sy
from syft import Tensor
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor

orig_value = np.array([[1, 2, 3.23], [32, 10.232, 42.42]])

fpt_tensor = FixedPrecisionTensor(value=orig_value, base=10, precision=3)
print(fpt_tensor)


print("=====")
value = fpt_tensor.decode()
print(value)


# Test send
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_client()
print(fpt_tensor.send(alice_client))
