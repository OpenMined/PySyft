# stdlib
from collections import OrderedDict
from typing import Any
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional

# third party
import torch as th

# syft relative
from .....core.node.common.client import Client
from .....core.store import ObjectStore
from .....lib.python.collections import OrderedDict as SyOrderedDict
from .....lib.python.dict import Dict
from .....lib.python.list import List
from .....lib.python.primitive_interface import PyPrimitive
from .....logger import traceback_and_raise
from .....util import obj2pointer_type
from ....plan.plan_builder import PLAN_BUILDER_VM
from ....plan.plan_builder import ROOT_CLIENT
from ....pointer.pointer import Pointer
from ...plan import Plan
from .plan import PlanTorchscript

__LIST_TYPE = (list, List)
__DICT_TYPE = (dict, Dict, OrderedDict, SyOrderedDict)


def is_list(*args: TypeList[Any]) -> bool:
    return all([isinstance(arg, __LIST_TYPE) for arg in args])


def is_dict(*args: TypeList[Any]) -> bool:
    return all([isinstance(arg, __DICT_TYPE) for arg in args])


def get_pointer_to_data_in_store(
    store: ObjectStore, value: Any, client: Client
) -> Optional[Any]:
    for obj in store.values():
        store_value: Any = obj.data
        if is_list(value, store_value):
            # Assume lists match if their contents are equal
            found = all(map(lambda a, b: a is b, value, store_value))
        elif is_dict(value, store_value):
            # Assume dicts match if their contents are equal
            for k1, k2 in sorted(value.keys()), sorted(store_value.keys()):
                if k1 != k2:
                    found = False
                    break
                if value[k1] is not store_value[k2]:
                    found = False
                    break
            else:
                found = True
        else:
            found = store_value is value

        if found:
            ptr_type = obj2pointer_type(obj.data)
            ptr = ptr_type(
                client=client,
                id_at_location=obj.id,
                object_type=obj.object_type,
                tags=obj.tags,
                description=obj.description,
            )
            return ptr

    return None


def translate(plan: Plan) -> PlanTorchscript:
    """Translates Syft Plan to torchscript"""

    class PlanWrapper(th.nn.Module):
        """
        Plan needs to be executed with kwargs while torchscript needs to be executed with args,
        so we need to have kwarg names inside torchscript to pass them into Plan.
        We use nn.Module to store input kwarg names inside torchscript.
        """

        def __init__(self, kwarg_names: TypeList[str]):
            _kwarg_names: TypeList[str]
            super(PlanWrapper, self).__init__()
            self._kwarg_names = kwarg_names

        def forward(self, *args: Any) -> Any:
            kwarg_ptrs: TypeDict[str, Pointer] = {}

            # Since Syft Plan needs pointers as args,
            # reverse-map actual arg values to pointers
            # by looking up these args in VM store
            for name, arg in zip(self._kwarg_names, args):
                ptr = get_pointer_to_data_in_store(
                    PLAN_BUILDER_VM.store, arg, ROOT_CLIENT
                )
                if not ptr:
                    traceback_and_raise(f"Could not map '{name}' arg value to Pointer")
                kwarg_ptrs[name] = ptr

            # Execute Plan in the same VM where it was built!
            res = plan(PLAN_BUILDER_VM, PLAN_BUILDER_VM.verify_key, **kwarg_ptrs)  # type: ignore
            return res

    # Builder VM holds inputs in the store, retrieve actual arg values from there
    kwarg_names = list(plan.inputs.keys())

    args = tuple(
        PLAN_BUILDER_VM.store[plan.inputs[name].id_at_location].data
        for name in kwarg_names
    )

    args = tuple(arg.upcast() if isinstance(arg, PyPrimitive) else arg for arg in args)

    # Dummy module that holds kwarg names
    wrapper: PlanWrapper = PlanWrapper(kwarg_names)

    # Run trace, passing real arg values
    ts = th.jit.trace(wrapper, args, check_trace=False)

    # Wrap in serializable class
    return PlanTorchscript(torchscript=ts)
