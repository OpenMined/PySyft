import torch
import asyncio
import websockets
from websocket import create_connection
import syft as sy
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.codes import MSGTYPE


class WebsocketClientWorker(BaseWorker):
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
        self.uri = f"ws://{self.host}:{self.port}"
        self.ws = create_connection(self.uri)
        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

    def search(self, query):
        return self.send_msg(MSGTYPE.SEARCH, query, location=self)

    def _send_msg(self, message, location):
        return location._recv_msg(message)

    def _recv_msg(self, message):
        print("msg", message)
        self.ws.send(message)
        return self.ws.recv()


class WebsocketServerWorker(VirtualWorker):
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
        self.broadcast_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)
        self.start()

    async def consumer_handler(self, websocket):
        while True:
            msg = await websocket.recv()
            await self.broadcast_queue.put(msg)

    async def producer_handler(self, websocket):
        while True:
            message = await self.broadcast_queue.get()
            print("server side message", message)
            response = self.recv_msg(message)
            await ws.send(response)

    async def handler(self, websocket, path):
        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket))

        done, pending = await asyncio.wait(
            [consumer_task, producer_task], return_when=asyncio.ALL_COMPLETED
        )
        for task in pending:
            task.cancel()

    def start(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
