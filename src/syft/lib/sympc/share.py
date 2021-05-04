# stdlib

# third party
from sympc.tensor import ShareTensor

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.sympc.share_tensor_pb2 import ShareTensor as ShareTensor_PB
from .session_util import protobuf_session_deserializer
from .session_util import protobuf_session_serializer


def object2proto(obj: object) -> ShareTensor_PB:

    share: ShareTensor = obj

    session = protobuf_session_serializer(share.session)
    proto = ShareTensor_PB(session=session)

    tensor_data = getattr(share.tensor, "data", None)
    if tensor_data is not None:
        proto.tensor.tensor.CopyFrom(protobuf_tensor_serializer(tensor_data))

    return proto


def proto2object(proto: ShareTensor_PB) -> ShareTensor:
    session = protobuf_session_deserializer(proto=proto.session)

    data = protobuf_tensor_deserializer(proto.tensor.tensor)
    share = ShareTensor(data=None, session=session)

    # Manually put the tensor since we do not want to re-encode it
    share.tensor = data.type(session.tensor_type)

    return share


GenerateWrapper(
    wrapped_type=ShareTensor,
    import_path="sympc.tensor.ShareTensor",
    protobuf_scheme=ShareTensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
