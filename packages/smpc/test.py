
from tensor import FixedPrecisionTensor
import numpy as np



orig_value = np.array([[1,2,3.23], [32, 10.232, 42.42]])

fpt_tensor = FixedPrecisionTensor(value=orig_value, base=10, precision=3)
print(fpt_tensor)


print("=====")
value = fpt_tensor.decode()
print(value)
