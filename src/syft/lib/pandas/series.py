"""Serde method for pd.Series."""

# third party
import pandas as pd
import pyarrow as pa

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.pandas.series_pb2 import PandasSeries as PandasSeries_PB


def object2proto(obj: pd.Series) -> PandasSeries_PB:
    """Convert pd.Series to PandasSeries_PB with pyarrow.

    # noqa: DAR101
    # noqa: DAR201

    """
    table = pa.Table.from_pandas(obj)
    return PandasSeries_PB(series=pa.serialize(table).to_buffer().to_pybytes())


def proto2object(proto: PandasSeries_PB) -> pd.Series:
    """Convert PandasSeries_PB to pd.Series with pyarrow.

    # noqa: DAR101
    # noqa: DAR201

    """
    reconstructed_buf = pa.py_buffer(proto.series)
    return pa.deserialize(reconstructed_buf).to_pandas()


GenerateWrapper(
    wrapped_type=pd.Series,
    import_path="pandas.Series",
    protobuf_scheme=PandasSeries_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
