import syft as sy
import numpy as np
import jax

data = np.array([2., 1.])
phi_tensor = sy.Tensor(data).annotate_with_dp_metadata(lower_bound=0, upper_bound=2, data_subjects="ishan")
gamma_tensor = (phi_tensor[0] * 2 + phi_tensor[1]) * phi_tensor[0] 

from syft.core.tensor.autodp.gamma_functions import GAMMA_FUNC_MAPPER
from syft.core.tensor.autodp.gamma_tensor_ops import GAMMA_TENSOR_OP

f_for_jit = GAMMA_FUNC_MAPPER[GAMMA_TENSOR_OP(gamma_tensor.func_str)]
f_for_jit([[0., 0., 0.]])
