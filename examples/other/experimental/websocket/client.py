import torch

from syft.core.hooks import TorchHook
from syft.core.workers import WebSocketWorker

hook = TorchHook(local_worker=WebSocketWorker(id=0, port=1111, verbose=True))
remote_client = WebSocketWorker(
    hook=hook, id=2, port=1112, is_pointer=True, verbose=True
)
hook.local_worker.add_worker(remote_client)
x = torch.FloatTensor([1, 2, 3, 4, 5]).send(remote_client)
x2 = torch.FloatTensor([1, 2, 3, 4, 4]).send(remote_client)
y = x + x2 + x
print(y.owners)
print(y.get())
