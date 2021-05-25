# third party
import numpy as np
import torch as th

# syft absolute
# import syft as sy
# from syft.core.adp.entity import Entity
from syft.core.tensor.tensor import Tensor


def test_mnist() -> None:
    data_batch = np.random.rand(4, 28 * 28)
    label_batch = np.random.rand(4, 10)

    # bob = Entity(unique_name="Bob")

    data = Tensor(data_batch).autograd(
        requires_grad=True
    )  # .private(0.01,1,entity=bob)
    target = Tensor(label_batch).autograd(
        requires_grad=True
    )  # .private(0.01,1,entity=bob)

    weights = np.random.rand(28 * 28, 10)

    pred = data.dot(weights)
    diff = target - pred
    pre_loss = np.square(diff)

    pre_loss.backward()
    pred.grad

    th_data = th.tensor(data.child.child, requires_grad=True)
    th_target = th.tensor(target.child.child, requires_grad=True)
    th_weights = th.tensor(weights, requires_grad=True)

    th_pred = th.tensor(th_data.mm(th_weights), requires_grad=True)
    th_diff = th_target - th_pred
    th_pre_loss = th_diff * th_diff
    # _another_loss = _diff * _diff

    th_pre_loss.sum().backward()
    th_pred.grad

    assert diff == th_diff.clone().detach().numpy()
    assert pre_loss == th_pre_loss.clone().detach().numpy()
    assert pred == th_pred.clone().detach().numpy()
    assert pred.grad == th_pred.grad.clone().detach().numpy()
