from torch import jit
from syft.execution.plan import Plan
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
        plan_params = translation_plan.parameters()
        if len(plan_params) > 0:
            args = (*args, plan_params)
        torchscript_plan = jit.trace(translation_plan, args)

        self.plan.torchscript = torchscript_plan

        return self.plan

    def remove(self):
        self.plan.torchscript = None

        return self.plan


# Register translators that should apply at Plan build time
Plan.register_build_translator(PlanTranslatorTorchscript)
