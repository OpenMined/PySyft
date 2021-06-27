# third party
import numpy as np

# syft absolute
from syft import Tensor
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor

orig_value = np.array([[1, 2, 3.23], [32, 10.232, 42.42]])

fpt_tensor = FixedPrecisionTensor(value=orig_value, base=10, precision=3)
print(fpt_tensor)


print("=====")
value = fpt_tensor.decode()
print(value)


val = Tensor(orig_value)
fpt_val = val.fix_precision()
print(fpt_val)

float_val = fpt_val.decode()
print(float_val)
import pdb

pdb.set_trace()

print(fpt_val + fpt_val)
