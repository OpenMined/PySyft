# third party
import numpy as np

# syft absolute
from syft.core.adp.adversarial_accountant import AdversarialAccountant
from syft.core.adp.entity import Entity
from syft.core.adp.tensor import Tensor

entities = [
    Entity(name="Tudor"),
    Entity(name="Madhava"),
    Entity(name="Kritika"),
    Entity(name="George"),
]

x = Tensor(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]) * 0.01).private(
    min_val=0, max_val=0.01, entities=entities
)
y = Tensor(np.array([[1], [1], [0], [0]]) * 0.01).private(
    min_val=0, max_val=0.01, entities=entities
)

acc = AdversarialAccountant(max_budget=3000)

weights = Tensor(np.random.uniform(size=(2, 1)))

for i in range(10):
    batch_loss = 0
    for row in range(int(len(x) / 2)):
        start = row * 2
        end = row * 2 + 2

        pred = x[start:end].dot(weights)
        loss = np.mean(np.square(y[start:end] - pred))
        loss.backward()

        weight_grad = weights.grad * 0.1

        weight_grad = weight_grad.publish(acc=acc, sigma=0.1)

        weights = weights - weight_grad
        batch_loss += loss.value
    print(batch_loss)
acc.print_ledger()
