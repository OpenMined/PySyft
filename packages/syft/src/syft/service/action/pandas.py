# stdlib
from typing import Any
from typing import ClassVar
from typing import Type

# third party
from pandas import DataFrame
from pandas import Series

# relative
from ...serde.serializable import serializable
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.transforms import drop
from ...types.transforms import make_set_default
from .action_object import ActionObject
from .action_object import ActionObjectV1
from .action_object import BASE_PASSTHROUGH_ATTRS
from .action_types import action_types


@serializable()
class PandasDataFrameObjectV1(ActionObjectV1):
    __canonical_name__ = "PandasDataframeObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[Type[Any]] = DataFrame
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS


@serializable()
class PandasDataFrameObject(ActionObject):
    __canonical_name__ = "PandasDataframeObject"
    __version__ = SYFT_OBJECT_VERSION_2

    syft_internal_type: ClassVar[Type[Any]] = DataFrame
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
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


@migrate(PandasDataFrameObject, PandasDataFrameObjectV1)
def downgrade_pandasdataframeobject_v2_to_v1():
    return [
        drop("syft_resolved"),
    ]


@migrate(PandasDataFrameObjectV1, PandasDataFrameObject)
def upgrade_pandasdataframeobject_v1_to_v2():
    return [
        make_set_default("syft_resolved", True),
    ]


@serializable()
class PandasSeriesObjectV1(ActionObjectV1):
    __canonical_name__ = "PandasSeriesObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = Series
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS


@serializable()
class PandasSeriesObject(ActionObject):
    __canonical_name__ = "PandasSeriesObject"
    __version__ = SYFT_OBJECT_VERSION_2

    syft_internal_type = Series
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS

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


@migrate(PandasSeriesObject, PandasSeriesObjectV1)
def downgrade_pandasseriesframeobject_v2_to_v1():
    return [
        drop("syft_resolved"),
    ]


@migrate(PandasSeriesObjectV1, PandasSeriesObject)
def upgrade_pandasseriesframeobject_v1_to_v2():
    return [
        make_set_default("syft_resolved", True),
    ]


action_types[DataFrame] = PandasDataFrameObject
action_types[Series] = PandasSeriesObject
