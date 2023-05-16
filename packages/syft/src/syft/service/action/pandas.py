# stdlib
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import Type

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

    syft_internal_type: ClassVar[Type[Any]] = DataFrame
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS + [
        "_typ",
        "shape",
        "ndim",
        "_get_axis",
        "axis",
        "axes",
        "_get_axis_number",
        "_get_block_manager_axis",
        "_mgr",
        "_constructor",
        "attrs",
    ]
    # this is added for instance checks for dataframes
    _typ = "dataframe"
    # syft_dont_wrap_attrs = ["shape"]

    def __dataframe__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__dataframe__(*args, **kwargs)

    def _get_axis(self, axis):
        if isinstance(axis, ActionObject):
            axis = axis.syft_action_data
        return self.syft_action_data._get_axis(axis)

    def _get_block_manager_axis(self, axis):
        if isinstance(axis, ActionObject):
            axis = axis.syft_action_data
        return self.syft_action_data._get_block_manager_axis(axis)

    def _get_axis_number(self, axis):
        if isinstance(axis, ActionObject):
            axis = axis.syft_action_data
        return self.syft_action_data._get_axis_number(axis)

    @property
    def attrs(self):
        return self.syft_action_data.attrs

    @property
    def _constructor(self):
        def wrapper(*args, **kwargs):
            obj = self.syft_action_data._constructor(*args, **kwargs)
            return ActionObject.from_obj(obj)

        return wrapper

    @property
    def _mgr(self):
        return self.syft_action_data._mgr

    @property
    def axis(self):
        return self.syft_action_data.axis

    @property
    def axes(self):
        return self.syft_action_data.axes

    @property
    def shape(self):
        return self.syft_action_data.shape

    @property
    def ndim(self):
        return self.syft_action_data.ndim

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def syft_get_property(self, obj: Any, method: str) -> Any:
        return getattr(self.syft_action_data, method)

    def syft_is_property(self, obj: Any, method: str) -> bool:
        cols = self.syft_action_data.columns.values.tolist()
        if method in cols:
            return True
        return super().syft_is_property(obj, method)


@serializable()
class PandasSeriesObject(ActionObject):
    __canonical_name__ = "PandasSeriesObject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type = Series
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS + [
        "_typ",
        "shape",
        "ndim",
        "_get_axis",
        "axis",
        "axes",
        "_get_axis_number",
        "_get_block_manager_axis",
        "_mgr",
        "_constructor",
        "attrs",
        "_constructor_expanddim",
    ]

    name: Optional[str] = None
    # syft_dont_wrap_attrs = ["shape"]
    _typ = "series"

    def __series__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__series__(*args, **kwargs)

    def __dataframe__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__dataframe__(*args, **kwargs)

    def _get_axis(self, axis):
        if isinstance(axis, ActionObject):
            axis = axis.syft_action_data
        return self.syft_action_data._get_axis(axis)

    def _get_block_manager_axis(self, axis):
        if isinstance(axis, ActionObject):
            axis = axis.syft_action_data
        return self.syft_action_data._get_block_manager_axis(axis)

    def _get_axis_number(self, axis):
        if isinstance(axis, ActionObject):
            axis = axis.syft_action_data
        return self.syft_action_data._get_axis_number(axis)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def syft_get_property(self, obj: Any, method: str) -> Any:
        return getattr(self.syft_action_data, method)

    @property
    def shape(self):
        return self.syft_action_data.shape

    @property
    def _constructor_expanddim(self):
        return self.syft_action_data._constructor_expanddim

    @property
    def attrs(self):
        return self.syft_action_data.attrs

    @property
    def _constructor(self):
        def wrapper(*args, **kwargs):
            obj = self.syft_action_data._constructor(*args, **kwargs)
            return ActionObject.from_obj(obj)

        return wrapper

    @property
    def _mgr(self):
        return self.syft_action_data._mgr

    @property
    def axis(self):
        return self.syft_action_data.axis

    @property
    def axes(self):
        return self.syft_action_data.axes

    @property
    def ndim(self):
        return self.syft_action_data.ndim

    def syft_is_property(self, obj: Any, method: str) -> bool:
        if method in ["str"]:
            return True
        # cols = self.syft_action_data.columns.values.tolist()
        # if method in cols:
        #     return True
        return super().syft_is_property(obj, method)


action_types[DataFrame] = PandasDataFrameObject
action_types[Series] = PandasSeriesObject
