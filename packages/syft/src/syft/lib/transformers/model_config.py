# third party
import transformers
from transformers import PretrainedConfig

# relative
from ... import deserialize
from ...core.common.serde.serializable import serializable
from ...proto.lib.transformers.hf_config_pb2 import HFConfig as HFConfig_PB
from ..python.primitive_factory import PrimitiveFactory
from ..python.util import upcast


def object2proto(obj: PretrainedConfig) -> HFConfig_PB:
    config_class = PrimitiveFactory.generate_primitive(value=type(obj).__name__)
    config_attrs = PrimitiveFactory.generate_primitive(value=obj.to_dict())

    config_proto = HFConfig_PB(
        id=config_class.id._object2proto(),
        config_class=config_class._object2proto(),
        config_attrs=config_attrs._object2proto(),
    )
    return config_proto


def proto2object(proto: HFConfig_PB) -> PretrainedConfig:
    config_class = getattr(transformers, upcast(deserialize(proto.config_class)))
    config = config_class.from_dict(upcast(deserialize(proto.config_attrs)))
    return config


serializable(generate_wrapper=True)(
    wrapped_type=PretrainedConfig,
    import_path="transformers.PretrainedConfig",
    protobuf_scheme=HFConfig_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
