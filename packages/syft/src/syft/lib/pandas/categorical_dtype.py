# third party
import pandas as pd

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.pandas.categorical_pb2 import (
    PandasCategoricalDtype as PandasCategoricalDtype_PB,
)
from ..python.list import List
from ..python.primitive_factory import PrimitiveFactory


def object2proto(obj: pd.CategoricalDtype) -> PandasCategoricalDtype_PB:
    # since pd.Index type is not integrated converted obj.categories to List
    pd_cat_list = PrimitiveFactory.generate_primitive(value=obj.categories.tolist())
    cat_list_proto = pd_cat_list._object2proto()

    return PandasCategoricalDtype_PB(
        id=cat_list_proto.id, categories=cat_list_proto, ordered=obj.ordered
    )


def proto2object(proto: PandasCategoricalDtype_PB) -> pd.CategoricalDtype:
    categories = List._proto2object(proto.categories).upcast()
    ordered = proto.ordered
    return pd.CategoricalDtype(categories=categories, ordered=ordered)


serializable(generate_wrapper=True)(
    wrapped_type=pd.CategoricalDtype,
    import_path="pandas.CategoricalDtype",
    protobuf_scheme=PandasCategoricalDtype_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
