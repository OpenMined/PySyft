import numpy as np
from syft.proto.lib.numpy.tensor_pb2 import TensorProto
from syft.lib.numpy.tensor_util import numpy_array_to_tensor
from syft.lib.numpy.tensor_util import tensor_to_numpy_array
from forbiddenfruit import curse

from syft.core.common.serde.serializable import Serializable


class NumpyTensorWrapper(Serializable):
    def __init__(self, value, as_wrapper):

        self.numpy_array = value

    def _object2proto(self) -> TensorProto:
        return numpy_array_to_tensor(self.numpy_array)

    @staticmethod
    def _proto2object(proto) -> int:
        return tensor_to_numpy_array(proto)

    @staticmethod
    def get_protobuf_schema() -> type:
        return TensorProto

    @staticmethod
    def get_wrapped_type() -> type:
        return np.array


ndarray = type(np.array([1, 2, 3]))
curse(ndarray, "serializable_wrapper_type", NumpyTensorWrapper)
