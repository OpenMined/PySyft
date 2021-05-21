# third party
from pymbolic.primitives import Variable

# syft relative
from ....generate_wrapper import GenerateWrapper
from ....proto.lib.pymbolic.expression_pb2 import (
    PymbolicExpression as PymbolicExpression_PB,
)
from ...util import full_name_with_name


def object2proto(obj: object) -> PymbolicExpression_PB:
    return PymbolicExpression_PB(
        obj_type=full_name_with_name(klass=type(obj)), name=obj.name
    )


def proto2object(proto: PymbolicExpression_PB) -> Variable:
    return Variable(name=proto.name)


GenerateWrapper(
    wrapped_type=Variable,
    import_path=full_name_with_name(Variable),
    protobuf_scheme=PymbolicExpression_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
