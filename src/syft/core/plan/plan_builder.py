# stdlib
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

# syft relative
from ...core.node.common.client import Client
from ...core.node.vm.plan_vm import PlanVirtualMachine
from ...core.pointer.pointer import Pointer
from ...logger import info
from ...logger import traceback_and_raise
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


def make_plan(func: Callable) -> Plan:
    inputs = build_plan_inputs(func)
    vm = PLAN_BUILDER_VM
    code = inspect.getsource(func)
    vm.record_actions()
    res = func(**inputs)
    vm.stop_recording()
    plan = Plan(actions=vm.recorded_actions, inputs=inputs, outputs=res, code=code)
    # cleanup
    vm.recorded_actions = []
    return plan


class PlanBuilder:
    def __init__(self, vm: PlanVirtualMachine = PLAN_BUILDER_VM):
        self.vm = vm

        try:
            self.plan: Plan = self.build()
        except Exception as e:
            traceback_and_raise(ValueError(f"Failed to build plan. {e}."))

        self.plan_pointer = None

    def build(self, *args: Tuple[Any, ...]) -> Plan:
        inputs = build_plan_inputs(self.forward)

        self.vm.record_actions()
        res = self.forward(**inputs)
        self.vm.stop_recording()
        plan: Plan = Plan(actions=self.vm.recorded_actions, inputs=inputs, outputs=res)
        self.vm.recorded_actions = []
        return plan

    def forward(self, *args: Tuple[Any, ...], **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> Any:
        if self.plan_pointer is None:
            client: Client = PlanVirtualMachine(name="alice").get_root_client()
            info("Model is not remote yet, sending to a new VM")
            self.send_plan(client)

        if self.plan_pointer is None:
            traceback_and_raise(ValueError("Unable to call Plan. Send failed."))

        return self.plan_pointer(**kwargs)  # type: ignore

    def send_plan(self, client: Client) -> None:
        self.plan_pointer = self.plan.send(client)  # type: ignore
