import torch
import syft as sy

hook = sy.TorchHook(torch=torch)

kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}
server = sy.workers.WebsocketServerWorker(**kwargs)
server.start()