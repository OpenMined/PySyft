import torch as th
import torch.nn.functional as F
from torch import nn

import syft as sy
from syft.workers import WebsocketClientWorker

use_cuda = th.cuda.is_available()
th.manual_seed(1)
device = th.device("cuda" if use_cuda else "cpu")

hook = sy.TorchHook(th)
hook.local_worker.is_client_worker = False
me = hook.local_worker


kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": False}
alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)


# TODO: maybe by default is_client_worker could be False?
# for plans the client_worker needs to be able to register tensors
# since it needs to build the plan.
# Loss function
@sy.func2plan
def loss_fn(real, pred):
    return ((real - pred) ** 2).mean()


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

    @sy.method2plan
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Force build
# TODO: this should be done automatically
loss_fn(th.tensor([1.0]), th.tensor([1.0]))

model = Net()

# TODO: this line should not be needed at all.
model.train()
model.send(me)

# Force build
# TODO: this should be done automatically
model(th.tensor([1.0, 2]))

# TODO: this line should not be needed at all.
model.get()

# Create and send train config
train_config = sy.TrainConfig(model=model, loss_plan=loss_fn, batch_size=2)
train_config.send(alice)

# TODO: Returns a tensor when it should actually return a Pointer to the result Tensor.

for epoch in range(5):
    loss = alice.fit(dataset_key="vectors")
    print("-" * 50)
    print("Iteration %s: alice's loss: %s" % (epoch, loss))
