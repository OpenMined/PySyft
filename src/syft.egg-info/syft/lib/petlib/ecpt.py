# third party
import petlib as pl

# syft relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serialize import _serialize as serialize
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.petlib.ecpt_pb2 import EcPt as EcPt_PB


def object2proto(obj: pl.ec.EcPt) -> EcPt_PB:
    return EcPt_PB(group=serialize(obj.group, to_proto=True), data=obj.export())


def proto2object(proto: EcPt_PB) -> pl.ec.EcPt:
    eg = deserialize(blob=proto.group, from_proto=True)
    return pl.ec.EcPt.from_binary(proto.data, eg)


GenerateWrapper(
    wrapped_type=pl.ec.EcPt,
    import_path="petlib.ec.EcPt",
    protobuf_scheme=EcPt_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
