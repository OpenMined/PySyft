# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch as th

# syft relative
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.torch.tensor_pb2 import TensorProto as Tensor_PB
from ...util import aggressive_set_attr
from ..generic import ObjectConstructor


class GeneratorConstructor(ObjectConstructor):

    __name__ = "GeneratorConstructor"

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Generator"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = th

    original_type = th.Generator


# Step 3: create constructor and install it in the library
GeneratorConstructor().install_inside_library()

# generator_type = type(th.Generator(device='cpu'))
# b = torch.

# aggressive_set_attr(
#     obj=torch_tensor_type, name="serializable_wrapper_type", attr=TorchTensorWrapper
# )
