# stdlib
from typing import Any

# third party
from pandas import DataFrame
from pandas import Series

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ...common.serde.serializable import serializable
from .action_object import ActionObject
from .action_types import action_types


@serializable(recursive_serde=True)
class PandasDataFrameObject(ActionObject):
    __canonical_name__ = "PandasDataframeObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = DataFrame
    syft_passthrough_attrs = []
    syft_dont_wrap_attrs = []

    def __dataframe__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__dataframe__(*args, **kwargs)


@serializable(recursive_serde=True)
class PandasSeriesObject(ActionObject):
    __canonical_name__ = "PandasSeriesObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = Series
    syft_passthrough_attrs = []
    syft_dont_wrap_attrs = []


action_types[DataFrame] = PandasDataFrameObject
action_types[Series] = PandasSeriesObject
