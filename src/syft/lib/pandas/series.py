# third party
import pandas as pd

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.dict import Dict
from ...lib.python.primitive_factory import PrimitiveFactory
from ...proto.lib.pandas.series_pb2 import PandasSeries as PandasSeries_PB


def object2proto(obj: pd.Series) -> PandasSeries_PB:
    series_dict = PrimitiveFactory.generate_primitive(value=obj.to_dict())
    dict_proto = series_dict._object2proto()

    return PandasSeries_PB(
        id=dict_proto.id,
        series=dict_proto,
    )


def proto2object(proto: PandasSeries_PB) -> pd.Series:
    series_dict = Dict._proto2object(proto=proto.series)
    return pd.Series(series_dict.upcast())


GenerateWrapper(
    wrapped_type=pd.Series,
    import_path="pandas.Series",
    protobuf_scheme=PandasSeries_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
