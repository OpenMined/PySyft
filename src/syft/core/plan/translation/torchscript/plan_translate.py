# stdlib
from collections import OrderedDict
from typing import Any
from typing import List

# third party
import torch as th

# syft absolute
from syft.core.plan import Plan
from syft.core.plan.plan_builder import PLAN_BUILDER_VM
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.lib.python.collections import OrderedDict as SyOrderedDict
from syft.lib.python.dict import Dict
from syft.lib.python.list import List as SyList
from syft.lib.python.primitive_interface import PyPrimitive
from syft.logger import traceback_and_raise

# syft relative
from ....pointer.pointer import Pointer
from .plan import PlanTorchscript

__LIST_TYPE = (SyList, list)
__DICT_TYPE = (dict, Dict, OrderedDict, SyOrderedDict)


def is_list(arg: Any, store_value: Any) -> bool:
    return isinstance(arg, __LIST_TYPE) and isinstance(store_value, __LIST_TYPE)


def is_dict(arg: Any, store_value: Any) -> bool:
    return isinstance(arg, __DICT_TYPE) and isinstance(store_value, __DICT_TYPE)


def translate(plan: Plan) -> PlanTorchscript:
    """Translates Syft Plan to torchscript"""

    class PlanWrapper(th.nn.Module):
        """
        Plan needs to be executed with kwargs while torchscript needs to be executed with args,
        so we need to have kwarg names inside torchscript to pass them into Plan.
        We use nn.Module to store input kwarg names inside torchscript.
        """

        def __init__(self, kwarg_names: List[str]):
            _kwarg_names: List[str]
            super(PlanWrapper, self).__init__()
            self._kwarg_names = kwarg_names

        def forward(self, *args: Any) -> Any:
            kwarg_ptrs: Dict[str, Pointer] = {}

            # Since Syft Plan needs pointers as args,
            # reverse-map actual arg values to pointers
            for name, arg in zip(self._kwarg_names, args):
                found = False
                for ptr in ROOT_CLIENT.store:
                    store_value: Any = PLAN_BUILDER_VM.store[ptr.id_at_location].data
                    if is_list(arg, store_value):
                        # Assume lists match if their contents are equal
                        found = all(map(lambda a, b: a is b, store_value, arg))
                    elif is_dict(arg, store_value):
                        # Assume dicts match if their contents are equal
                        for k1, k2 in sorted(arg.keys()), sorted(store_value.keys()):
                            if k1 != k2:
                                found = False
                                break

                            if args[k1] is not store_value[k2]:
                                found = False
                                break
                        else:
                            found = True
                    else:
                        found = store_value is arg

                    if found:
                        kwarg_ptrs[name] = ptr
                        break

                if not found:
                    traceback_and_raise(f"Could not map '{name}' arg value to Pointer")

            # Execute Plan in the same VM where it was built!
            res = plan(PLAN_BUILDER_VM, PLAN_BUILDER_VM.verify_key, **kwarg_ptrs)
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
