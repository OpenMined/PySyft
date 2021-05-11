import syft as sy
import numpy as np
from syft.lib.adp.tensor import Tensor
from syft.lib.adp.entity import Entity
from syft.lib.adp.adversarial_accountant import AdversarialAccountant

entities = [Entity(name="Tudor"), Entity(name="Madhava"), Entity(name="Kritika"), Entity(name="George")]

x = Tensor(np.array([[1,1],[1,0],[0,1],[0,0]])).private(min_val=0, max_val=1, entities=entities)
y = Tensor(np.array([[1],[1],[0],[0]])).private(min_val=0, max_val=1, entities=entities)

acc = AdversarialAccountant(max_budget=3000)

weights = Tensor(np.random.uniform(size=(2,1)))

for i in range(10):
    
    pred = x.dot(weights)
    loss = np.mean(np.square(y-pred))
    loss.backward()

    weight_grad = (weights.grad * 1)
    print(weight_grad)
    
    weight_grad = weight_grad.publish(acc=acc, sigma=0.1)
    
    weights = weights - weight_grad
    print(loss.value)
    acc.print_ledger()