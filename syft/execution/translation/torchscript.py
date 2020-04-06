from torch import jit
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

        # To avoid storing Plan state tensors in torchscript, they will be send as parameters
        plan_params = plan.parameters()
        if len(plan_params) > 0:
            args = (*args, plan_params)
        torchscript_plan = jit.trace(plan, args)
        plan.torchscript = torchscript_plan
        plan.forward = tmp_forward

        return plan

    def remove(self):
        plan = self.plan
        plan.torchscript = None

        return plan
