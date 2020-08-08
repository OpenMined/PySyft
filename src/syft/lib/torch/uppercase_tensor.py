from ..generic import ObjectConstructor
from syft.proto.lib.numpy.tensor_pb2 import TensorProto
from syft.core.store.storeable_object import StorableObject
from syft.lib.numpy.tensor_util import numpy_array_to_tensor
from syft.lib.numpy.tensor_util import tensor_to_numpy_array
from ...util import aggressive_set_attr

import torch as th


class UppercaseTensorConstructor(ObjectConstructor):

    __name__ = "UppercaseTensorConstructor"

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Tensor"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = th


# Step 3: create constructor and install it in the library
UppercaseTensorConstructor().install_inside_library()

torch_tensor_type = type(th.tensor([1, 2, 3]))


class TorchTensorWrapper(StorableObject):
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
        return numpy_array_to_tensor(self.value.numpy())

    @staticmethod
    def _data_proto2object(proto: TensorProto) -> int:

        # proto -> original numpy type
        data = tensor_to_numpy_array(proto)

        return th.tensor(data)

    @staticmethod
    def get_data_protobuf_schema() -> type:
        return TensorProto

    @staticmethod
    def get_wrapped_type() -> type:
        return torch_tensor_type

    @staticmethod
    def construct_new_object(id, data, tags, description):
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=torch_tensor_type, name="serializable_wrapper_type", attr=TorchTensorWrapper
)
