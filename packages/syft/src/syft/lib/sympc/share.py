# stdlib

# stdlib
import dataclasses
from uuid import UUID

# third party
import sympc
from sympc.config import Config
from sympc.tensor import ShareTensor

# syft absolute
import syft

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.sympc.share_tensor_pb2 import ShareTensor as ShareTensor_PB
from ..python.primitive_factory import PrimitiveFactory


def object2proto(obj: object) -> ShareTensor_PB:
    share: ShareTensor = obj

    session_uuid = ""
    config = {}

    if share.session_uuid is not None:
        session_uuid = str(share.session_uuid)

    config = dataclasses.asdict(share.config)
    session_uuid_syft = session_uuid
    conf_syft = syft.serialize(
        PrimitiveFactory.generate_primitive(value=config), to_proto=True
    )
    proto = ShareTensor_PB(session_uuid=session_uuid_syft, config=conf_syft)

    tensor_data = getattr(share.tensor, "data", None)
    if tensor_data is not None:
        proto.tensor.CopyFrom(syft.serialize(share.tensor, to_proto=True))

    return proto


def proto2object(proto: ShareTensor_PB) -> ShareTensor:
    if proto.session_uuid:
        session = sympc.session.get_session(proto.session_uuid)
        if session is None:
            raise ValueError(f"The session {proto.session_uuid} could not be found")

        config = dataclasses.asdict(session.config)
    else:
        config = syft.deserialize(proto.config, from_proto=True)

    tensor = syft.deserialize(proto.tensor, from_proto=True)
    share = ShareTensor(data=None, config=Config(**config))

    if proto.session_uuid:
        share.session_uuid = UUID(proto.session_uuid)

    # Manually put the tensor since we do not want to re-encode it
    share.tensor = tensor

    return share


GenerateWrapper(
    wrapped_type=ShareTensor,
    import_path="sympc.tensor.ShareTensor",
    protobuf_scheme=ShareTensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
