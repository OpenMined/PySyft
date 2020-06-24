from pythreepio.threepio import Threepio
from pythreepio.command import Command
from syft.execution.action import Action
from syft.execution.role import Role
from syft.execution.translation import TranslationTarget
from syft.execution.translation.abstract import AbstractPlanTranslator


class PlanTranslatorThreepio(AbstractPlanTranslator):
    """Parent translator class for all Threepio supported frameworks"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate_action(self, action: Action, to_framework: str) -> Action:
        """Uses threepio to perform command level translation given a specific action"""
        threepio = Threepio(self.plan.base_framework, to_framework, None)
        function_name = action.name.split(".")[-1]
        args = action.args if action.target is None else (action.target, *action.args)
        translated_cmds = threepio.translate(Command(function_name, args, action.kwargs))

        actions = []
        for cmd in translated_cmds:
            new_action = action.copy()
            new_action.name = ".".join(cmd.attrs)
            new_action.args = tuple(cmd.args)
            new_action.kwargs = cmd.kwargs
            new_action.target = None
            actions.append(new_action)
        return actions

    def translate_framework(self, to_framework: str) -> Role:
        """Translates current plan's Role to specified framework"""
        role = self.plan.role.copy()
        plan = self.plan.copy()
        # Check to see if plan has been translated to this framework yet
        if plan.roles.get(to_framework, None) is not None:
            plan.default_framework = to_framework
            return plan

        new_actions = []
        for action in role.actions:
            translated_actions = self.translate_action(action, to_framework)
            new_actions = [*new_actions, *translated_actions]
        role.actions = tuple(new_actions)
        return role


class PlanTranslatorTfjs(PlanTranslatorThreepio):
    """Performs translation from 'list of ops' Plan into 'list of ops in tfjs' Plan"""

    framework = TranslationTarget.TENSORFLOW_JS.value

    def __init__(self, plan):
        super().__init__(plan)

    def translate(self) -> Role:
        """Translate role of given plan to tensorflow.js"""
        return self.translate_framework(TranslationTarget.TENSORFLOW_JS.value)
