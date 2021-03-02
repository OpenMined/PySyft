from syft.core.node.vm.plan_vm import PlanVirtualMachine
from syft.core.plan.plan import Plan

import inspect

PLAN_BUILDER_VM = PlanVirtualMachine(name="alice")
ROOT_CLIENT = PLAN_BUILDER_VM.get_root_client()


def build_plan_inputs(forward_func):
    signature = inspect.signature(forward_func)
    res = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            res[k] = v.default.send(ROOT_CLIENT)
        else:
            raise ValueError(f"arg {k} has no placeholder as default value (required for @make_plan functions)")
    return res


def make_plan(func):
    inputs = build_plan_inputs(func)
    vm = PLAN_BUILDER_VM
    vm.record_actions()
    res = func(**inputs)
    vm.stop_recording()
    plan = Plan(actions=vm.recorded_actions, inputs=inputs, outputs=res)
    #cleanup
    vm.recorded_actions=[]
    return plan
