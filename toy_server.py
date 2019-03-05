import time
import asyncio
import websockets
import syft as sy
from syft.serde import serialize


class FederatedLearningServer:
    def __init__(self, id, connection_params, hook, loop=None):
        self.port = connection_params['port']
        self.host = connection_params['host']
        self.id = id
        self.broadcast_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop() if loop is None else loop

    async def consumer_handler(self, websocket):
        while True:
            msg = await websocket.recv()
            print(f'[{self.id} | RCV] {msg}')
            await self.broadcast_queue.put(msg)


    async def producer_handler(self, websocket):
        while True:
            message = await self.broadcast_queue.get()
            await websocket.send(f'Hi, {message}')

    async def handler(self, websocket, path):
        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket))

        done, pending = await asyncio.wait([consumer_task, producer_task] , return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()


    def start(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        print('server')
        asyncio.get_event_loop().run_forever()
