from typing import ClassVar
from typing import Type
from typing import Any 
from jaxlib.xla_extension import DeviceArray
from jaxlib.xla_extension import Device
import numpy as np
import jax.numpy as hidden_jnp
import jax as hidden_jax
import jaxlib as hidden_jaxlib

from .serializable import serializable
from .action_object import ActionObject
from .syft_object import SYFT_OBJECT_VERSION_1
from .action_types import action_types
from .numpy import NumpyArrayObject
import sys
from typing import Callable

class DeviceArrayObjectPointer:
    pass

# @serializable()
# class DeviceObject(ActionObject):
#     __canonical_name__ = "DeviceObject"
#     __version__ = SYFT_OBJECT_VERSION_1
    
#     syft_internal_type: ClassVar[Type[Any]] = Device
#     syft_passthrough_attrs = ["id"]
#     syft_dont_wrap_attrs = []

#     def id(self):
#         return self.syft_action_data.id


def wrap_result(result: Any):
    if isinstance(result, tuple):
        return tuple(wrap_result(x) for x in result)
    if isinstance(result, DeviceArray):
        return DeviceArrayObject(syft_action_data=result)
    if isinstance(result, np.ndarray):
        return DeviceArrayObject(syft_action_data=hidden_jnp.asarray(result))
    # if isinstance(result, hidden_jaxlib.)
    if isinstance(result, Callable):
        def wrapper(*args, **kwargs):
            return wrap_result(result(*args, **kwargs))
        return wrapper
    return result

class WrapperJaxNumpy:
    def __getattribute__(self, __name: str) -> Any:
        def wrapper(*args, **kwargs):
            return wrap_result(hidden_jnp.__getattribute__(__name)(*args, **kwargs))
        
        return wrapper

class WrapperJax:
    def __getattribute__(self, __name: str) -> Any:
        # if __name == "jit":
        #     return jit_wrapper
        def wrapper(*args, **kwargs):
            return wrap_result(hidden_jax.__getattribute__(__name)(*args, **kwargs))
        
        return wrapper

jax = WrapperJax()
jnp = WrapperJaxNumpy()

# class SyftCompiledFunctionPointer(ActionObject):

#     def __init__(self, compiled_function):
#         self.compiled_function = compiled_function

#     def __call__(self, *args, **kwargs):
#         result = self.compiled_function(*args, **kwargs)
#         return wrap_result(result)

@serializable()
class DeviceArrayObject(ActionObject, np.lib.mixins.NDArrayOperatorsMixin):
    __canonical_name__ = "DeviceArrayObject"
    __version__ = SYFT_OBJECT_VERSION_1
    
    syft_internal_type: ClassVar[Type[Any]] = DeviceArray
    syft_pointer_type = DeviceArrayObjectPointer
    syft_passthrough_attrs = ["__jax_array__"]
    syft_dont_wrap_attrs = ["__jax_array__", "device", "devices", "_device", "dtype", "shape", "__array__"]

    def __jax_array__(self) -> DeviceArray:
        # TODO: test this for gpus (probably we should check jax default config)
        self.syft_action_data._device = jax.devices()[0]
        return self.syft_action_data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(
            x.syft_action_data
            if isinstance(x, DeviceArrayObject)
            else x
            for x in inputs
        )

        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, np.ndarray):
            result = hidden_jnp.asarray(result)
        if type(result) is tuple:
            return tuple(DeviceArrayObject(syft_action_data=x) for x in result)
        else:
            return DeviceArrayObject(syft_action_data=result)

    def __iter__(self) -> Any:
        return self.syft_action_data.__iter__()

action_types[DeviceArray] = DeviceArrayObject
    