# third party
import torch

# syft absolute
import syft as sy


def torch_grad_test(client: sy.VirtualMachineClient) -> None:
    x = client.torch.Tensor([[1, 1], [1, 1]])
    x.requires_grad = True
    gt = client.torch.Tensor([[1, 1], [1, 1]]) * 16 - 0.5

    loss_fn = client.torch.nn.MSELoss()

    v = x + 2
    y = v ** 2

    loss = loss_fn(y, gt)
    loss.backward()

    assert x.grad.get().equal(torch.Tensor([[-19.5, -19.5], [-19.5, -19.5]]))
    assert x.data.get().equal(torch.Tensor([[1, 1], [1, 1]]))
