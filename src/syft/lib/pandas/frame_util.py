# third party
import pandas as pd

# syft relative
from ...proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB
from ..python import Dict
from ..python.primitive_factory import PrimitiveFactory


def protobuf_dataframe_serializer(dataframe: pd.DataFrame) -> PandasDataFrame_PB:
    dataframe = PrimitiveFactory.generate_primitive(value=dataframe.to_dict())
    dataframe_proto = dataframe._object2proto()

    return PandasDataFrame_PB(
        dataframe=dataframe_proto,
    )


def protobuf_dataframe_deserializer(proto: PandasDataFrame_PB) -> pd.DataFrame:
    dataframe_dict = Dict._proto2object(proto=proto.dataframe)
    try:
        dataframe_dict = {
            key.data: dict(value) for key, value in dataframe_dict.items()
        }
    except AttributeError:
        dataframe_dict = {key: dict(value) for key, value in dataframe_dict.items()}

    df = pd.DataFrame.from_dict(dataframe_dict)
    return df
