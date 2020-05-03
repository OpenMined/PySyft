import syft as sy
import torch


def test_plan_module_tracing():
    @sy.func2plan(args_shape=[(1,)])
    def plan_test(x, torch=torch):
        y = torch.rand([1])
        return x + y

    p = plan_test(torch.tensor([3]))
    assert len(plan_test.role.actions) == 2
