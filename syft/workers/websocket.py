import torch
import asyncio
import websockets
import syft as sy
from syft.workers.base import BaseWorker


class WebsocketCode(Enum):
    LOCAL = 1
    REMOTE = 2


class WebsocketWorker(BaseWorker):
    def __init__(
        self,
        hook,
        id=0,
        is_client_worker=False,
        log_msgs=False,
        verbose=False,
        connection_params={},
        data={},
    ):
        # TODO get angry when we have no connection params
        self.port = connection_params["port"]
        self.host = connection_params["host"]
        self.connections = set()
        self.broadcast_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)
        self.start()

    def _send_msg(self, message, location):
        return location._recv_msg(message)

    def _recv_msg(self, message):
        self.broadcast_queue.put(message)
        # TODO figure out how to get response from other worker efficiently
        return self.recv_msg(message)  # This line is wrong and must be fixed

    async def consumer_handler(self, websocket, cid):
        while True:
            msg = await websocket.recv()
            await self.broadcast_queue.put((WebsocketCode.LOCAL, msg))

    async def producer_handler(self, websocket, cid):
        while True:
            code, message = await self.broadcast_queue.get()

            for idx, ws in enumerate(self.connections):
                # TODO investigate using a binary flag for local to save cycles
                if code == WebsocketCode.LOCAL:
                    response = self.recv_msg(message)
                    await ws.send(response)
                elif code == WebsocketCode.REMOTE:
                    await ws.send(message)
                else:
                    raise RuntimeError("Invalid Websocket Code")

    async def handler(self, websocket, path):
        cid = len(self.connections)
        self.connections.add(websocket)
        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket, cid))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket, cid))

        done, pending = await asyncio.wait(
            [consumer_task, producer_task], return_when=asyncio.FIRST_COMPLETED
        )
        print("Connection closed, canceling pending tasks")
        for task in pending:
            task.cancel()

    def start(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
