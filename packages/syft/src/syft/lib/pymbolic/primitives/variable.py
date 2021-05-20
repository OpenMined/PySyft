# third party
from pymbolic.primitives import Variable

# syft relative
from ....generate_wrapper import GenerateWrapper
from ....proto.lib.pymbolic.variable_pb2 import Variable as Variable_PB
from ...util import full_name_with_name


def object2proto(obj: object) -> Variable_PB:
    proto = Variable_PB()
    proto.name = obj.name
    return proto


def proto2object(proto: Variable_PB) -> Variable:
    obj = Variable(name=proto.name)
    return obj


GenerateWrapper(
    wrapped_type=Variable,
    import_path=full_name_with_name(Variable),
    protobuf_scheme=Variable_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
