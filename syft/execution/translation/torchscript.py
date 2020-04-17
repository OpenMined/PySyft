from torch import jit
from syft.execution.plan import Plan
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation.abstract import AbstractPlanTranslator


class PlanTranslatorTorchscript(AbstractPlanTranslator):
    """Performs translation from 'list of ops' Plan into torchscript Plan"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate(self):
        plan = self.plan

        args_shape = plan.get_args_shape()
        args = PlaceHolder.create_placeholders(args_shape)

        # Temporarily remove reference to original function
        tmp_forward = plan.forward
        plan.forward = None

        # To avoid storing Plan state tensors inside the torchscript,
        # we trace wrapper func, which accepts state parameters as last arg
        # and sets them into the Plan before executing the Plan
        def wrap_stateful_plan(*args):
            role = plan.role
            state = args[-1]
            if 0 < len(role.state.state_placeholders) == len(state) and isinstance(
                state, (list, tuple)
            ):
                state_placeholders = tuple(
                    role.placeholders[ph.id.value] for ph in role.state.state_placeholders
                )
                PlaceHolder.instantiate_placeholders(role.state.state_placeholders, state)
                PlaceHolder.instantiate_placeholders(state_placeholders, state)

            return plan(*args[:-1])

        plan_params = plan.parameters()
        if len(plan_params) > 0:
            torchscript_plan = jit.trace(wrap_stateful_plan, (*args, plan_params))
        else:
            torchscript_plan = jit.trace(plan, args)
        plan.torchscript = torchscript_plan
        plan.forward = tmp_forward

        return plan

    def remove(self):
        plan = self.plan
        plan.torchscript = None

        return plan


# Register translators that should apply at Plan build time
Plan.register_build_translator(PlanTranslatorTorchscript)
