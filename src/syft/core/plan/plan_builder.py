# stdlib
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

# syft relative
from ...core.node.common.client import Client
from ...core.node.vm.plan_vm import PlanVirtualMachine
from ...core.pointer.pointer import Pointer
from .plan import Plan

PLAN_BUILDER_VM: PlanVirtualMachine = PlanVirtualMachine(name="plan_vm")
ROOT_CLIENT: Client = PLAN_BUILDER_VM.get_root_client()


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


def map_in2out(inputs: dict, outputs: tuple) -> dict:
    in2out_map = {}
    for k, v in inputs.items():
        input_id_at_location = getattr(v, "id_at_location", None)
        for i, o in enumerate(outputs):
            if o.id_at_location == input_id_at_location:
                in2out_map[k] = i
    return in2out_map


def make_plan(func: Callable, inputs: Optional[Dict[str, Any]] = None) -> Plan:
    if inputs is None:
        inputs = build_plan_inputs(func)
    vm = PLAN_BUILDER_VM
    code = inspect.getsource(func)
    vm.record_actions()
    res = func(**inputs)
    vm.stop_recording()
    plan = Plan(
        actions=vm.recorded_actions,
        inputs=inputs,
        outputs=res,
        i2o_map=map_in2out(inputs, res),
        code=code,
    )
    # cleanup
    vm.recorded_actions = []
    return plan
