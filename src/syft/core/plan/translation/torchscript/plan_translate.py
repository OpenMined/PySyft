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
from syft.logger import traceback_and_raise

# syft relative
from ....pointer.pointer import Pointer
from .plan import PlanTorchscript


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
                    store_value = PLAN_BUILDER_VM.store[ptr.id_at_location].data
                    if isinstance(arg, (SyList, list)) and isinstance(
                        store_value, (SyList, list)
                    ):
                        # Assume lists match if their contents are equal
                        found = all(map(lambda a, b: a is b, store_value, arg))
                    elif isinstance(
                        arg, (dict, Dict, OrderedDict, SyOrderedDict)
                    ) and isinstance(
                        store_value, (dict, Dict, OrderedDict, SyOrderedDict)
                    ):
                        # Assume dicts match if their contents are equal
                        arg_keys, store_value_keys = sorted(arg.keys()), sorted(
                            store_value.keys()
                        )
                        arg_values = tuple(arg[k] for k in arg_keys)
                        store_value_values = tuple(arg[k] for k in arg_keys)
                        keys_match = all(
                            map(lambda a, b: a == b, arg_keys, store_value_keys)
                        )
                        values_match = all(
                            map(lambda a, b: a is b, arg_values, store_value_values)
                        )
                        found = keys_match and values_match
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

    # Remove wrappers, th.jit.trace doesn't like them
    args = tuple(list(arg) if isinstance(arg, SyList) else arg for arg in args)
    args = tuple(
        dict((str(k), v) for k, v in arg.items())
        if isinstance(arg, SyOrderedDict)
        else arg
        for arg in args
    )

    # Dummy module that holds kwarg names
    wrapper: PlanWrapper = PlanWrapper(kwarg_names)

    # Run trace, passing real arg values
    ts = th.jit.trace(wrapper, args, check_trace=False)

    # Wrap in serializable class
    return PlanTorchscript(torchscript=ts)
