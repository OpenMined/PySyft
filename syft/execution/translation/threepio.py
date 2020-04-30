from torch import jit
from pythreepio.threepio import Threepio
from pythreepio.utils import Command
from pythreepio.errors import TranslationMissing
from syft.execution.translation import TranslationTarget
from syft.execution.translation.abstract import AbstractPlanTranslator


class PlanTranslatorThreepio(AbstractPlanTranslator):
    """Parent translator class for all Threepio supported frameworks"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate_action(self, action, to_framework):
        threepio = Threepio(self.plan.base_framework, to_framework, None)
        function_name = action.name.split(".")[-1]
        if action.target is None:
            # Translate normally if action isn't a method of a tensor
            args = action.args
            cmd = threepio.translate(Command(function_name, action.args, action.kwargs))
        else:
            # Otherwise reformat into proper translation
            args = [action.target, *action.args]
            cmd = threepio.translate(Command(function_name, args, action.kwargs))

        new_action = action.copy()
        new_action.name = ".".join(cmd.attrs)
        new_action.args = cmd.args
        new_action.kwargs = cmd.kwargs
        new_action.target = None
        return new_action

    def translate_framework(self, to_framework):
        plan = self.plan.copy()
        # Check to see if plan has been translated to this framework yet
        if plan.role.operations.get(to_framework, None) is not None:
            plan.role.actions = plan.role.operations[to_framework]
            return plan

        new_actions = []
        for action in plan.role.actions:
            new_actions.append(self.translate_action(action, to_framework))
        plan.role.actions = new_actions
        plan.role.operations[to_framework] = new_actions
        return plan


class PlanTranslatorTfjs(PlanTranslatorThreepio):
    """Performs translation from 'list of ops' Plan into 'list of ops in tfjs' Plan"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate(self):
        return self.translate_framework(TranslationTarget.TENSORFLOW_JS.value)
