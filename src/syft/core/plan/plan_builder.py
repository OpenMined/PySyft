# stdlib
import inspect
from typing import Callable
from typing import Dict

# syft absolute
from syft import Plan
from syft.core.node.vm.plan_vm import PlanVirtualMachine
from syft.core.pointer.pointer import Pointer

PLAN_BUILDER_VM = PlanVirtualMachine(name="alice")
ROOT_CLIENT = PLAN_BUILDER_VM.get_root_client()


def build_plan_inputs(forward_func: Callable) -> Dict[str, Pointer]:
    signature = inspect.signature(forward_func)
    res = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            res[k] = v.default.send(ROOT_CLIENT)
        else:
            raise ValueError(
                f"arg {k} has no placeholder as default value (required for @make_plan functions)"
            )
    return res


def make_plan(func: Callable) -> Plan:
    inputs = build_plan_inputs(func)
    vm = PLAN_BUILDER_VM
    vm.record_actions()
    res = func(**inputs)
    vm.stop_recording()
    plan = Plan(actions=vm.recorded_actions, inputs=inputs, outputs=res)
    # cleanup
    vm.recorded_actions = []
    return plan
