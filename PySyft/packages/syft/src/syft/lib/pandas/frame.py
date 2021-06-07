"""Serde method for pd.DataFrame."""

# third party
import pandas as pd
import pyarrow as pa

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB


def object2proto(obj: pd.DataFrame) -> PandasDataFrame_PB:
    """Convert pd.DataFrame to PandasDataFrame_PB with pyarrow.

    Args:
        obj: target Dataframe

    Returns:
        Serialized version of Dataframe, which will be used to reconstruction.

    """
    schema = pa.Schema.from_pandas(obj)
    table = pa.Table.from_pandas(obj)
    sink = pa.BufferOutputStream()

    writer = pa.ipc.new_file(sink, schema)
    writer.write(table)
    writer.close()

    buf = sink.getvalue()

    siz = len(buf)
    df_bytes = pa.compress(buf, asbytes=True)

    return PandasDataFrame_PB(dataframe=df_bytes, decompressed_size=siz)


def proto2object(proto: PandasDataFrame_PB) -> pd.DataFrame:
    """Proto to object conversion using to return desired model.

    Args:
        proto: Serialized version of Dataframe, which will be used to reconstruction.

    Returns:
        Re-constructed dataframe.
    """
    buf = pa.decompress(proto.dataframe, decompressed_size=proto.decompressed_size)
    return pa.ipc.open_file(buf).read_pandas()


GenerateWrapper(
    wrapped_type=pd.DataFrame,
    import_path="pandas.DataFrame",
    protobuf_scheme=PandasDataFrame_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
