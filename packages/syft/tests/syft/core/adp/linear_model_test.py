# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.adversarial_accountant import AdversarialAccountant
from syft.core.adp.entity import Entity
from syft.core.adp.publish import publish
from syft.core.adp.scalar import PhiScalar
from syft.core.tensor.tensor import Tensor


def test_autodp_phiscalar_publish() -> None:
    x = PhiScalar(0, 0.01, 1)
    y = PhiScalar(0, 0.02, 1)
    z = PhiScalar(0, 0.02, 1)

    o = x * x + y * y + z
    z = o * o * o

    acc = AdversarialAccountant(max_budget=10)
    z.publish(acc=acc, sigma=0.2)

    publish([z, z], acc=acc, sigma=0.2)

    acc.print_ledger()
    assert len(acc.entities) == 3


@pytest.mark.xfail
def test_autodp_train_linear_model() -> None:
    # entities = [
    #     Entity(name="Tudor"),
    #     Entity(name="Madhava"),
    #     Entity(name="Kritika"),
    #     Entity(name="George"),
    # ]

    entity = Entity(name="Trask")

    x = (
        Tensor(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]))
        .private(min_val=0, max_val=1, entity=entity)
        .autograd(requires_grad=True)
    )
    y = (
        Tensor(np.array([[1], [1], [0], [0]]))
        .private(min_val=0, max_val=1, entity=entity)
        .autograd(requires_grad=True)
    )

    weights = Tensor(np.random.uniform(size=(2, 1))).autograd(requires_grad=True)
    # print("type _weights ", type(_weights))
    # weights = _weights + 0

    acc = AdversarialAccountant(max_budget=7)

    for _ in range(1):
        batch_loss = 0

        pred = x.dot(weights)
        loss = np.mean(np.square(y - pred))
        loss.backward()

        weight_grad = weights.grad * 0.5
        weight_grad = weight_grad.publish(acc=acc, sigma=0.1)

        weights = weights - weight_grad
        batch_loss += loss.value

        acc.print_ledger()

    assert len(acc.entities) == 4
    assert batch_loss > 0
    assert True is False


@pytest.mark.xfail
def test_adding_scalars() -> None:
    weights = Tensor(np.random.uniform(size=(2, 1))).autograd(requires_grad=True)
    weights = weights + 0  # 0 has no requires_grad so this fails in the add op
    weights.backward()
