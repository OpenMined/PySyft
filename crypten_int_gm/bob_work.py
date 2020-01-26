import syft as sy
import torch

from syft.workers.websocket_client import WebsocketClientWorker


hook = sy.TorchHook(torch)
kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": True}

bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)

# Bob uses rank 1
bob.connect_to_mpc(1, 2)

# Should be blocked here until alice puts something in the store
print(bob.get_from_store_mpc("test"))

bob.put_to_store_mpc("Hello", "World")
