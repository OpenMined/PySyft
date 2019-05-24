import torch as th
import torch.nn.functional as F
from torch import nn

import syft as sy
from syft import workers
from syft.frameworks.torch import pointers

use_cuda = th.cuda.is_available()
th.manual_seed(1)
device = th.device("cuda" if use_cuda else "cpu")

hook = sy.TorchHook(th)
me = hook.local_worker

kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": False}
alice = workers.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)

# Loss function
@th.jit.script
def loss_fn(real, pred):
    return ((real - pred) ** 2).mean()


# Model
class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
data = th.tensor(th.tensor([[-10, -2.0], [1, 1.1], [11, 22.1], [-10, 1.2]]))

traced_model = th.jit.trace(model, data)

model_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=traced_model)
loss_fn_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=loss_fn)

model_ptr = me.send(model_with_id, alice)
loss_fn_ptr = me.send(loss_fn_with_id, alice)

# Create and send train config
train_config = sy.TrainConfig(
    model_id=model_ptr.id_at_location, loss_plan_id=loss_fn_ptr.id_at_location, batch_size=2
)
train_config.send(alice)

for epoch in range(5):
    loss = alice.fit(dataset_key="vectors", return_id=88 + epoch)
    print("-" * 50)
    print("Iteration %s: alice's loss: %s" % (epoch, loss))

new_model = model_ptr.get()
data = th.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
target = th.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

print("\nEvaluation before training")
pred = model(data)
loss = loss_fn(real=target, pred=pred)
print("Loss: {}".format(loss))

print("\nEvaluation after training:")
pred = new_model(data)
loss = loss_fn(real=target, pred=pred)
print("Loss: {}".format(loss))
