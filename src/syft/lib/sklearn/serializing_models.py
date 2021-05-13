"""Serde method for LogisticRegression (currently)!"""

# third party
import sklearn  # noqa: 401

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.dict import Dict
from ...lib.python.primitive_factory import PrimitiveFactory
from ...proto.lib.sklearn.logistic_model_pb2 import Logistic as Logistic_PB


def object2proto(obj: sklearn.linear_model.LogisticRegression) -> Logistic_PB:
    """Object to proto conversion using Logistic_PB.

    Args:
        obj: Model which we have to send to client

    Returns:
        Serialized proto.

    """
    vars_dict = PrimitiveFactory.generate_primitive(value=vars(obj))
    dict_proto = vars_dict._object2proto()

    return Logistic_PB(
        id=dict_proto.id,
        model=dict_proto,
    )


def proto2object(proto: Logistic_PB) -> sklearn.linear_model.LogisticRegression:
    """Proto to object conversion using to return desired model.

    Args:
        proto: Serialized version of model, which will be used to re-construct model.

    Returns:
        Re-constrcuted model.
    """
    vars_dict = Dict._proto2object(proto=proto.model)
    ret_model = sklearn.linear_model.LogisticRegression()

    for attribute, value in vars_dict.items():
        if hasattr(value, "upcast"):
            # This would convert all Sy objects to their original types
            value = value.upcast()
        setattr(ret_model, attribute, value)

    return ret_model


GenerateWrapper(
    wrapped_type=sklearn.linear_model.LogisticRegression,
    import_path="sklearn.linear_model.LogisticRegression",
    protobuf_scheme=Logistic_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
