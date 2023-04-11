from typing import ClassVar
from typing import Type
from typing import Any 
from jaxlib.xla_extension import DeviceArray
import numpy as np
from jax import numpy as jnp

from .serializable import serializable
from .action_object import ActionObject
from .syft_object import SYFT_OBJECT_VERSION_1
from .action_types import action_types
from .numpy import NumpyArrayObject

class DeviceArrayObjectPointer:
    pass

@serializable()
class DeviceArrayObject(ActionObject, np.lib.mixins.NDArrayOperatorsMixin):
    __canonical_name__ = "DeviceArrayObject"
    __version__ = SYFT_OBJECT_VERSION_1
    
    syft_internal_type: ClassVar[Type[Any]] = DeviceArray
    syft_pointer_type = DeviceArrayObjectPointer
    syft_passthrough_attrs = []
    syft_dont_wrap_attrs = []
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(
            x.syft_action_data
            if isinstance(x, DeviceArrayObject)
            else x
            for x in inputs
        )

        result = getattr(ufunc, method)(*inputs, **kwargs)
        if type(result) is tuple:
            return tuple(
                DeviceArrayObject(syft_action_data=x)
                for x in result
            )
        else:
            return DeviceArrayObject(syft_action_data=result)
         
        

action_types[DeviceArray] = DeviceArrayObject
    