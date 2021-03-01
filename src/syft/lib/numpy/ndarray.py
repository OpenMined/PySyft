# third party
import numpy as np
import pandas as pd

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.dict import Dict
from ...lib.python.primitive_factory import PrimitiveFactory
from ...proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB


def object2proto(obj: np.ndarray) -> PandasDataFrame_PB:
    value = pd.DataFrame(obj)
    pd_dict = PrimitiveFactory.generate_primitive(value=value.to_dict())
    dict_proto = pd_dict._object2proto()

    return PandasDataFrame_PB(
        id=dict_proto.id,
        dataframe=dict_proto,
    )


def proto2object(proto: PandasDataFrame_PB) -> np.ndarray:
    dataframe_dict = Dict._proto2object(proto=proto.dataframe)
    return pd.DataFrame.from_dict(dataframe_dict.upcast()).to_numpy()


GenerateWrapper(
    wrapped_type=np.ndarray,
    import_path="np.ndarray",
    protobuf_scheme=PandasDataFrame_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
