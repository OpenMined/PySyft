from typing import Any
from typing import ClassVar
from typing import Type

from ...serde.serializable import serializable
from .action_object import ActionObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from .action_object import BASE_PASSTHROUGH_ATTRS

class ModelPointer:
    pass

@serializable
class ModelObject(ActionObject):
    __canonical_name__ = "Model"
    __version__ = SYFT_OBJECT_VERSION_1
    
    syft_internal_type: ClassVar[Type[Any]]
    syft_pointer_type = ModelPointer
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS
    # syft_dont_wrap_attrs = ["dtype"]