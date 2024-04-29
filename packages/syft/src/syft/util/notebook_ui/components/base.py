# third party
import ipywidgets as widgets

# relative
from ....types.syft_object import SYFT_OBJECT_VERSION_1
from ....types.syft_object import SyftBaseObject


class HTMLComponentBase(SyftBaseObject):
    __canonical_name__ = "HTMLComponentBase"
    __version__ = SYFT_OBJECT_VERSION_1

    def to_html(self) -> str:
        raise NotImplementedError()

    def to_widget(self) -> widgets.Widget:
        return widgets.HTML(value=self.to_html())

    def _repr_html_(self) -> str:
        return self.to_html()
