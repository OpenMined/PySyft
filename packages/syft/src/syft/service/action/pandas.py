# stdlib
from typing import Any
from typing import ClassVar

# third party
from pandas import DataFrame
from pandas import Series

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from .action_object import ActionObject
from .action_object import BASE_PASSTHROUGH_ATTRS
from .action_types import action_types


@serializable()
class PandasDataFrameObject(ActionObject):
    __canonical_name__ = "PandasDataframeObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[type] = DataFrame
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS
    # this is added for instance checks for dataframes
    # syft_dont_wrap_attrs = ["shape"]

    def __dataframe__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__dataframe__(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def syft_get_property(self, obj: Any, method: str) -> Any:
        return getattr(self.syft_action_data, method)

    def syft_is_property(self, obj: Any, method: str) -> bool:
        cols = self.syft_action_data.columns.values.tolist()
        if method in cols:
            return True
        return super().syft_is_property(obj, method)

    def __bool__(self) -> bool:
        if self.syft_action_data_cache is None:
            return False
        return bool(self.syft_action_data_cache.empty)


@serializable()
class PandasSeriesObject(ActionObject):
    __canonical_name__ = "PandasSeriesObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = Series
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS

    # name: Optional[str] = None
    # syft_dont_wrap_attrs = ["shape"]

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def syft_get_property(self, obj: Any, method: str) -> Any:
        return getattr(self.syft_action_data, method)

    def syft_is_property(self, obj: Any, method: str) -> bool:
        if method in ["str"]:
            return True
        # cols = self.syft_action_data.columns.values.tolist()
        # if method in cols:
        #     return True
        return super().syft_is_property(obj, method)


action_types[DataFrame] = PandasDataFrameObject
action_types[Series] = PandasSeriesObject
