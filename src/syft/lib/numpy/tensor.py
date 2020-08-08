import numpy as np
from syft.proto.lib.numpy.tensor_pb2 import TensorProto
from syft.lib.numpy.tensor_util import numpy_array_to_tensor
from syft.lib.numpy.tensor_util import tensor_to_numpy_array
from forbiddenfruit import curse
from google.protobuf.message import Message
from syft.core.store.storeable_object import StorableObject


class NumpyTensorWrapper(StorableObject):
    def __init__(self, value):
        super().__init__(
            data=value,
            key=value.id,
            tags=value.tags if hasattr(value, "tags") else [],
            description=value.description if hasattr(value, "description") else "",
        )
        print("Wrapped tensor with id:" + str(value.id))
        self.value = value

    def _data_object2proto(self) -> TensorProto:
        return numpy_array_to_tensor(self.value)

    @staticmethod
    def _data_proto2object(proto: Message) -> int:

        # proto -> original numpy type
        data = tensor_to_numpy_array(proto)

        # convert to numpy subclassed type (see ..generic.py)
        data = (np.array(data.shape) * 0) + data
        return data

    @staticmethod
    def get_data_protobuf_schema() -> type:
        return TensorProto

    @staticmethod
    def get_wrapped_type() -> type:
        return np.array

    @staticmethod
    def construct_new_object(id, data, tags, description):
        data.id = id
        data.tags = tags
        data.description = description
        return data


ndarray = type(np.array([1, 2, 3]))
curse(ndarray, "serializable_wrapper_type", NumpyTensorWrapper)
