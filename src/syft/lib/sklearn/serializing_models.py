# third party
import sklearn  # noqa: 401

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.dict import Dict
from ...lib.python.primitive_factory import PrimitiveFactory
from ...proto.lib.sklearn.logistic_model_pb2 import Logistic as Logistic_PB


def object2proto(obj: sklearn.linear_model.LogisticRegression) -> Logistic_PB:
    vars_dict = PrimitiveFactory.generate_primitive(value=vars(obj))
    dict_proto = vars_dict._object2proto()

    return Logistic_PB(
        id=dict_proto.id,
        model=dict_proto,
    )


def proto2object(proto: Logistic_PB) -> sklearn.linear_model.LogisticRegression:
    vars_dict = Dict._proto2object(proto=proto.model)
    ret_model = sklearn.linear_model.LogisticRegression()

    for attribute, value in vars_dict.items():
        if str(type(value)).find("_SyNone") != -1:
            value = None
        setattr(ret_model, attribute, value)

    return ret_model


GenerateWrapper(
    wrapped_type=sklearn.linear_model.LogisticRegression,
    import_path="sklearn.linear_model.LogisticRegression",
    protobuf_scheme=Logistic_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
