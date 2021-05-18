# third party
import numpy as np

# syft absolute
from syft.lib.adp.adversarial_accountant import AdversarialAccountant
from syft.lib.adp.entity import Entity
from syft.lib.adp.publish import publish
from syft.lib.adp.scalar import PhiScalar
from syft.lib.adp.tensor import Tensor


def test_autodp_train_linear_model() -> None:
    x = PhiScalar(0, 0.01, 1)
    y = PhiScalar(0, 0.02, 1)
    z = PhiScalar(0, 0.02, 1)

    entities = [
        Entity(unique_name="Tudor"),
        Entity(unique_name="Madhava"),
        Entity(unique_name="Kritika"),
        Entity(unique_name="George"),
    ]
    # a = Tensor(np.array([[1, 1], [1, 0], [0, 1], [0, 0]])).private(
    #     min_val=0, max_val=1, entities=entities, is_discrete=True
    # )
    # b = Tensor(np.array([[1], [1], [0], [0]])).private(
    #     min_val=0, max_val=1, entities=entities, is_discrete=False
    # )

    o = x * x + y * y + z
    z = o * o * o

    acc = AdversarialAccountant(max_budget=10)
    z.publish(acc=acc, sigma=0.2)

    # the data scientist should never see this
    acc = AdversarialAccountant(max_budget=10)
    publish([z, z], acc=acc, sigma=0.2)

    acc = AdversarialAccountant(max_budget=3000000)

    entities = [
        Entity(unique_name="Tudor"),
        Entity(unique_name="Madhava"),
        Entity(unique_name="Kritika"),
        Entity(unique_name="George"),
    ]

    x = Tensor(np.array([[1, 1], [1, 0], [0, 1], [0, 0]])).private(
        min_val=0, max_val=1, entities=entities, is_discrete=True
    )
    y = Tensor(np.array([[1], [1], [0], [0]])).private(
        min_val=0, max_val=1, entities=entities, is_discrete=False
    )

    _weights = Tensor(np.random.uniform(size=(2, 1)))

    weights = _weights + 0
    # acc = AdversarialAccountant(max_budget=7)  # causes error at end of budget
    acc = AdversarialAccountant(max_budget=3000000)

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

    assert batch_loss > 0
