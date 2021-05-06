# third party
import pandas as pd

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.list import List
from ...lib.python.primitive_factory import PrimitiveFactory
from ...proto.lib.pandas.categorical_pb2 import (
    PandasCategorical as PandasCategorical_PB,
)


def object2proto(obj: pd.Categorical) -> PandasCategorical_PB:
    pd_codes_list = PrimitiveFactory.generate_primitive(value=obj.codes.tolist())
    codes_proto = pd_codes_list._object2proto()
    pd_cat_list = PrimitiveFactory.generate_primitive(value=obj.categories.tolist())
    cat_proto = pd_cat_list._object2proto()

    return PandasCategorical_PB(
        id=codes_proto.id,
        codes=codes_proto,
        categories=cat_proto,
        ordered=obj.ordered,
    )


def proto2object(proto: PandasCategorical_PB) -> pd.Categorical:
    categories = List._proto2object(proto.categories).upcast()
    codes = List._proto2object(proto.codes).upcast()
    ordered = proto.ordered
    return pd.Categorical.from_codes(codes, categories=categories, ordered=ordered)


GenerateWrapper(
    wrapped_type=pd.Categorical,
    import_path="pandas.Categorical",
    protobuf_scheme=PandasCategorical_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
