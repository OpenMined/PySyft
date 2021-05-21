# third party
from sympy.core.symbol import Symbol

# syft relative
from ....generate_wrapper import GenerateWrapper
from ....proto.lib.sympy.expression_pb2 import SympyExpression as SympyExpression_PB
from ...util import full_name_with_name


def object2proto(obj: object) -> SympyExpression_PB:
    a = SympyExpression_PB(obj_type=full_name_with_name(klass=type(obj)), name=obj.name)
    return a


def proto2object(proto: SympyExpression_PB) -> Symbol:
    return Symbol(name=proto.name)


GenerateWrapper(
    wrapped_type=Symbol,
    import_path=full_name_with_name(Symbol),
    protobuf_scheme=SympyExpression_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
