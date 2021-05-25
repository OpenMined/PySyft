# third party
import numpy as np
import torch as th

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.tensor.tensor import Tensor


def test_vs_torch() -> None:
    data_batch = np.random.rand(4, 28 * 28)
    label_batch = np.random.rand(4, 10)
    data = Tensor(data_batch).autograd(requires_grad=True)
    target = Tensor(label_batch).autograd(requires_grad=True)

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

    th_pre_loss.sum().backward()
    th_pred.grad

    assert diff == th_diff.clone().detach().numpy()
    assert pre_loss == th_pre_loss.clone().detach().numpy()
    assert pred == th_pred.clone().detach().numpy()
    assert pred.grad == th_pred.grad.clone().detach().numpy()


def test_train_mnist() -> None:
    data_batch = np.random.rand(4, 28 * 28)
    label_batch = np.random.rand(4, 10)

    bob = Entity(unique_name="Bob")

    data = Tensor(data_batch).private(0.01, 1, entity=bob).autograd(requires_grad=True)
    target = (
        Tensor(label_batch).private(0.01, 1, entity=bob).autograd(requires_grad=True)
    )

    weights = Tensor(np.random.rand(28 * 28, 10)).autograd(requires_grad=True)

    for i in range(10):

        pred = data.dot(weights)
        diff = target - pred
        pre_loss = np.square(diff)
        loss = np.mean(pre_loss)
        extraneous_thing = -diff
        loss.backward()

        wdiff = weights.grad * 0.01
        weights = -wdiff + weights

    assert loss.data_child < 10.0
