# third party
import numpy as np
import torch

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.primitive_factory import PrimitiveFactory
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.numpy.flags_pb2 import FlagsProto


def object2proto(obj: np.ndarray) -> FlagsProto:
    tensor = torch.from_numpy(obj).clone()
    tensor_proto = protobuf_tensor_serializer(tensor)
    vars_dict = PrimitiveFactory.generate_primitive(value=vars(obj.flags))
    dict_proto = vars_dict._object2proto()

    return FlagsProto(tensor=tensor_proto, id=dict_proto.id, flags=dict_proto)


def proto2object(proto: FlagsProto) -> np.ndarray:
    tensor = protobuf_tensor_deserializer(proto.tensor)
    array = tensor.to("cpu").detach().numpy().copy()

    return array


GenerateWrapper(
    wrapped_type=np.flagsobj,
    import_path="numpy.flagsobj",
    protobuf_scheme=FlagsProto,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
