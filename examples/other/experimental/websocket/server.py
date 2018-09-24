from syft.core.hooks import TorchHook
from syft.core.workers import WebSocketWorker

hook = TorchHook()


local_worker = WebSocketWorker(
    hook=hook,
    id=2,
    port=1112,
    is_pointer=False,
    is_client_worker=False,
)
