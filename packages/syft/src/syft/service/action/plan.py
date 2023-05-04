# stdlib
import inspect
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

# relative
from ... import ActionObject
from ... import Worker
from ...client.client import SyftClient
from ...serde.recursive import recursive_serde_register
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from .action_object import Action
from .action_object import TraceResult


class Plan(SyftObject):
    __canonical_name__ = "Plan"
    __version__ = SYFT_OBJECT_VERSION_1
    syft_passthrough_attrs = ["inputs", "outputs", "code", "actions", "client"]

    inputs: Dict[str, ActionObject]
    outputs: List[ActionObject]
    actions: List[Action]
    code: str
    client: Optional[SyftClient] = None

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

    def remap_actions_to_inputs(self, **new_inputs):
        pass

    def __call__(self, *args, **kwargs):
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

        # return self.outputs

    # def __call__(self, *args, **kwargs):
    #     # todo, fix
    #     if self.client is None:
    #         raise ValueError("first set client")
    #     if args != ():
    #         raise ValueError(
    #             f"Only kwargs are allowed for plan execution, found {args}"
    #         )

    #     for k, v_action in kwargs.items():
    #         if not isinstance(v_action, ActionObject):
    #             v_action = ActionObject.from_obj(v_action)
    #         v_action.id = self.inputs[k].id
    #         self.client.api.services.action.set(v_action)

    #     for a in self.actions:
    #         self.client.api.services.action.execute(a)
    #     outputs = [
    #         self.client.api.services.action.get_pointer(x.id) for x in self.outputs
    #     ]
    #     if len(outputs) == 1:
    #         return outputs[0]
    #     else:
    #         return outputs[1]


def planify(func):
    TraceResult.reset()
    ActionObject.add_trace_hook()
    worker = Worker.named("plan_building", reset=True, processes=0)
    client = worker.root_client
    plan_kwargs = build_plan_inputs(func, client)
    outputs = func(**plan_kwargs)
    if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
        outputs = [outputs]
    ActionObject.remove_trace_hook()
    actions = TraceResult.result
    TraceResult.reset()
    code = inspect.getsource(func)
    plan = Plan(inputs=plan_kwargs, actions=actions, outputs=outputs, code=code)
    return ActionObject.from_obj(plan)


def build_plan_inputs(
    forward_func: Callable, client: SyftClient
) -> Dict[str, ActionObject]:
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
