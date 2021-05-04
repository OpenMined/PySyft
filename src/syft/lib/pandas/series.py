"""Serde method for pd.Series."""

# third party
import pandas as pd
import pyarrow as pa

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.pandas.series_pb2 import PandasSeries as PandasSeries_PB


def object2proto(obj: pd.Series) -> PandasSeries_PB:
    """Convert pd.Series to PandasDataFrame_PB with pyarrow.

    Args:
        obj: target Series

    Returns:
        Serialized version of Series, which will be used to reconstruction.

    """
    # https://arrow.apache.org/docs/python/pandas.html
    # series must either be converted to a dataframe or use pa.Array
    # however pa.Array mentions you must account for the null values yourself
    dataframe = obj.to_frame()
    schema = pa.Schema.from_pandas(dataframe)
    table = pa.Table.from_pandas(dataframe)
    sink = pa.BufferOutputStream()

    writer = pa.ipc.new_file(sink, schema)
    writer.write(table)
    writer.close()

    buf = sink.getvalue()

    siz = len(buf)
    df_bytes = pa.compress(buf, asbytes=True)

    return PandasSeries_PB(series=df_bytes, decompressed_size=siz)


def proto2object(proto: PandasSeries_PB) -> pd.Series:
    """Convert PandasSeries_PB to pd.Series with pyarrow.

    Args:
        proto: Serialized version of Series, which will be used to reconstruction.

    Returns:
        Re-constructed Series.

    """
    buf = pa.decompress(proto.series, decompressed_size=proto.decompressed_size)
    dataframe = pa.ipc.open_file(buf).read_pandas()
    # we know that this is a series being stored as a dataframe so just grab the first
    return dataframe[dataframe.columns[0]]


GenerateWrapper(
    wrapped_type=pd.Series,
    import_path="pandas.Series",
    protobuf_scheme=PandasSeries_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
