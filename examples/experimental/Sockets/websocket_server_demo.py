import torch
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

if __name__ == "__main__":
    hook = sy.TorchHook(torch)
    websocket_server = WebsocketServerWorker(hook, "localhost", 8765, "pysyft_socket_server")
    websocket_server.start()
