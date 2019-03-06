import syft as sy
import torch as th
import sys

host = "localhost"
port = sys.argv[1]

hook = sy.TorchHook(th)
kwargs = {"id": host+":"+str(port), "host": host, "port": port, "hook": hook}
server = sy.workers.WebsocketServerWorker(**kwargs)
server.start()
