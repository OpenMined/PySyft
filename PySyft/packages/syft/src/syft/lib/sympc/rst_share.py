# stdlib
import dataclasses
from uuid import UUID

# third party
import sympc
from sympc.config import Config
from sympc.tensor import ReplicatedSharedTensor

# syft absolute
import syft

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.sympc.replicatedshared_tensor_pb2 import (
    ReplicatedSharedTensor as ReplicatedSharedTensor_PB,
)
from ..python.primitive_factory import PrimitiveFactory


def object2proto(obj: object) -> ReplicatedSharedTensor_PB:
    share: ReplicatedSharedTensor = obj

    session_uuid = ""
    config = {}

    if share.session_uuid is not None:
        session_uuid = str(share.session_uuid)

    config = dataclasses.asdict(share.config)
    session_uuid_syft = session_uuid
    conf_syft = syft.serialize(
        PrimitiveFactory.generate_primitive(value=config), to_proto=True
    )
    proto = ReplicatedSharedTensor_PB(session_uuid=session_uuid_syft, config=conf_syft)

    for tensor in share.shares:
        proto.tensor.append(protobuf_tensor_serializer(tensor))

    return proto


def proto2object(proto: ReplicatedSharedTensor_PB) -> ReplicatedSharedTensor:
    if proto.session_uuid:
        session = sympc.session.get_session(proto.session_uuid)
        if session is None:
            raise ValueError(f"The session {proto.session_uuid} could not be found")

        config = dataclasses.asdict(session.config)
    else:
        config = syft.deserialize(proto.config, from_proto=True)

    output_shares = []

    for tensor in proto.tensor:
        output_shares.append(protobuf_tensor_deserializer(tensor))

    share = ReplicatedSharedTensor(shares=None, config=Config(**config))

    if proto.session_uuid:
        share.session_uuid = UUID(proto.session_uuid)

    share.shares = output_shares

    return share


GenerateWrapper(
    wrapped_type=ReplicatedSharedTensor,
    import_path="sympc.tensor.ReplicatedSharedTensor",
    protobuf_scheme=ReplicatedSharedTensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
