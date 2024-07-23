# stdlib
from collections.abc import Callable
import inspect
from typing import Any

# relative
from ... import ActionObject
from ... import Worker
from ...client.client import SyftClient
from ...serde.recursive import recursive_serde_register
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from .action_object import Action
from .action_object import TraceResultRegistry


class Plan(SyftObject):
    __canonical_name__ = "Plan"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_passthrough_attrs: list[str] = [
        "inputs",
        "outputs",
        "code",
        "actions",
        "client",
    ]

    inputs: dict[str, ActionObject]
    outputs: list[ActionObject]
    actions: list[Action]
    code: str
    client: SyftClient | None = None

    def __repr__(self) -> str:
        obj_str = "Plan"

        inp_str = "Inputs:\n"
        inp_str += "\n".join(
            [f"\t\t{k}: {v.__class__.__name__}" for k, v in self.inputs.items()]
        )

        act_str = f"Actions:\n\t\t{len(self.actions)} Actions"

        out_str = "Outputs:\n"
        out_str += "\n".join([f"\t\t{o.__class__.__name__}" for o in self.outputs])

        plan_str = "Plan code:\n"
        plan_str += f'"""\n{self.code}\n"""' if self.code is not None else ""

        return f"{obj_str}\n{inp_str}\n{act_str}\n{out_str}\n\n{plan_str}"

    def remap_actions_to_inputs(self, **new_inputs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ActionObject | list[ActionObject]:
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs


def planify(func: Callable) -> ActionObject:
    TraceResultRegistry.reset_result_for_thread()
    # TraceResult.reset()
    ActionObject.add_trace_hook()
    worker = Worker.named(name="plan_building", reset=True, processes=0)
    client = worker.root_client
    if client is None:
        raise ValueError("Not able to get client for plan building")
    # if client.settings is not None:
    #     client.settings.enable_eager_execution(enable=True) # NOTE: Disabled until we bring back eager execution
    TraceResultRegistry.set_trace_result_for_current_thread(client=client)
    try:
        # TraceResult._client = client
        plan_kwargs = build_plan_inputs(func, client)
        outputs = func(**plan_kwargs)
        if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
            outputs = [outputs]
        ActionObject.remove_trace_hook()
        actions = TraceResultRegistry.get_trace_result_for_thread().result  # type: ignore
        TraceResultRegistry.reset_result_for_thread()
        code = inspect.getsource(func)
        for a in actions:
            if a.create_object is not None:
                # warmup cache
                a.create_object.syft_action_data  # noqa: B018
        plan = Plan(inputs=plan_kwargs, actions=actions, outputs=outputs, code=code)
        return ActionObject.from_obj(plan)
    finally:
        TraceResultRegistry.reset_result_for_thread()


def build_plan_inputs(
    forward_func: Callable, client: SyftClient
) -> dict[str, ActionObject]:
    signature = inspect.signature(forward_func)
    res = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            default_value = v.default
            if not isinstance(default_value, ActionObject):
                default_value = ActionObject.from_obj(default_value)
            res[k] = default_value.send(client)
        else:
            raise ValueError(
                f"arg {k} has no placeholder as default value (required for @make_plan functions)"
            )
    return res


recursive_serde_register(Plan)
