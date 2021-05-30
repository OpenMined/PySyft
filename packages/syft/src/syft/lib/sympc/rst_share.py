# third party
from sympc.tensor import ReplicatedSharedTensor

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.sympc.replicatedshared_tensor_pb2 import (
    ReplicatedSharedTensor as ReplicatedSharedTensor_PB,
)
from .session_util import protobuf_session_deserializer
from .session_util import protobuf_session_serializer


def object2proto(obj: object) -> ReplicatedSharedTensor_PB:
    share: ReplicatedSharedTensor = obj

    session = protobuf_session_serializer(share.session)
    proto = ReplicatedSharedTensor_PB(session=session)

    tensor_data = getattr(share.shares, "data", None)
    if tensor_data is not None:
        proto.tensor.tensor.CopyFrom(protobuf_tensor_serializer([tensor_data]))

    return proto


def proto2object(proto: ReplicatedSharedTensor_PB) -> ReplicatedSharedTensor:
    session = protobuf_session_deserializer(proto=proto.session)

    data = protobuf_tensor_deserializer(proto.tensor.tensor)
    share = ReplicatedSharedTensor(shares=None, session=session)

    # Manually put the tensor since we do not want to re-encode it
    share.shares = data.type(session.tensor_type)

    return share


GenerateWrapper(
    wrapped_type=ReplicatedSharedTensor,
    import_path="sympc.tensor.ReplicatedSharedTensor",
    protobuf_scheme=ReplicatedSharedTensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
