import syft as sy
import torch

from syft.workers.websocket_client import WebsocketClientWorker

hook = sy.TorchHook(torch)
kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": True}

alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)

alice.connect_to_mpc(0,2)
alice.put_to_store_mpc("test", "PySyft + Crypten")

print(alice.get_from_store_mpc("Hello"))
