# third party
import PIL
import numpy as np

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.python.dict import Dict
from ...lib.python.primitive_factory import PrimitiveFactory
from ...proto.lib.numpy import tensor_pb2 as Tensor_PB
#from ...proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB


def object2proto(obj: PIL.Image.Image) -> Tensor_PB:
    image_array = PrimitiveFactory.generate_primitive(value=np.array(obj))
    array_proto = image_array._object2proto()

    return Tensor_PB(
        id=dict_proto.id,
        dataframe=dict_proto,
    )


def proto2object(proto: PandasDataFrame_PB) -> pd.DataFrame:
    dataframe_dict = Dict._proto2object(proto=proto.dataframe)
    return pd.DataFrame.from_dict(dataframe_dict.upcast())


GenerateWrapper(
    wrapped_type=pd.DataFrame,
    import_path="pandas.DataFrame",
    protobuf_scheme=PandasDataFrame_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
