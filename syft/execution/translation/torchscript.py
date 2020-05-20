from torch import jit
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation.abstract import AbstractPlanTranslator


class PlanTranslatorTorchscript(AbstractPlanTranslator):
    """Performs translation from 'list of ops' Plan into torchscript Plan"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate(self):
        translation_plan = self.plan.copy()
        translation_plan.forward = None

        args_shape = translation_plan.get_args_shape()
        args = PlaceHolder.create_placeholders(args_shape)

        # To avoid storing Plan state tensors in torchscript, they will be send as parameters
        # we trace wrapper func, which accepts state parameters as last arg
        # and sets them into the Plan before executing the Plan
        def wrap_stateful_plan(*args):
            role = translation_plan.role
            state = args[-1]
            if 0 < len(role.state.state_placeholders) == len(state) and isinstance(
                state, (list, tuple)
            ):
                state_placeholders = tuple(
                    role.placeholders[ph.id.value] for ph in role.state.state_placeholders
                )
                PlaceHolder.instantiate_placeholders(role.state.state_placeholders, state)
                PlaceHolder.instantiate_placeholders(state_placeholders, state)

            return translation_plan(*args[:-1])

        plan_params = translation_plan.parameters()
        if len(plan_params) > 0:
            torchscript_plan = jit.trace(wrap_stateful_plan, (*args, plan_params))
        else:
            torchscript_plan = jit.trace(translation_plan, args)

        self.plan.torchscript = torchscript_plan
        return self.plan

    def remove(self):
        self.plan.torchscript = None

        return self.plan
