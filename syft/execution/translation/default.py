from syft.execution.translation.abstract import AbstractPlanTranslator


class PlanTranslatorDefault(AbstractPlanTranslator):
    """Performs translation from 'list of ops' Plan into torchscript Plan"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate(self):
        #  do nothing, Plan is built in default "list of ops" variant
        return self.plan

    def remove(self):
        plan = self.plan
        plan.role.actions = []

        return plan
