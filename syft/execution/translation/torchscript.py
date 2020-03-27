import torch
from torch import jit
from syft.execution import Plan
from syft.execution.translation.abstract import AbstractPlanTranslator


class PlanTranslatorTorchscript(AbstractPlanTranslator):
    """Performs translation from 'list of ops' Plan into torchscript Plan"""

    def __init__(self, plan):
        super().__init__(plan)

    def translate(self):
        plan = self.plan.copy()
        args_shape = plan.get_args_shape()
        args = Plan._create_placeholders(args_shape)

        # Remove reference to original function
        plan.forward = None
        torchscript_plan = jit.trace_module(TorchscriptPlan(plan), {"forward": args})
        plan.torchscript = torchscript_plan

        # Remove actions and state, these should be captured in torchscript now
        plan.actions = []
        plan.state = []
        return plan


class TorchscriptPlan(torch.nn.Module):
    """nn.Module wrapper for Plan that registers state tensors of Plan into torchscript"""

    def __init__(self, plan: "Plan"):
        super(TorchscriptPlan, self).__init__()
        # Add state tensors as nn.Parameter on nn.Module to make it available in torchscript
        for idx, param in enumerate(plan.parameters()):
            setattr(self, "param%d" % idx, param)
        self.plan = plan

    def forward(self, *args):
        return self.plan(*args)
