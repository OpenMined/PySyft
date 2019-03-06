import syft as sy
import torch as th

host = "localhost"
port = 8765

hook = sy.TorchHook(th)
kwargs = {"id": "fed1", "host": host, "port": port, "hook": hook}
server = sy.workers.WebsocketServerWorker(**kwargs)
server.start()
