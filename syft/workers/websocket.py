import torch
import asyncio
import websockets
import syft as sy
from enum import Enum
from syft.workers.base import BaseWorker
from threading import Thread
from syft.codes import MSGTYPE


class WebsocketCode(Enum):
    LOCAL = 1
    REMOTE = 2

class WebsocketClientWorker(BaseWorker):
    def __init__(
        self, hook, id=0, is_client_worker=False,
        log_msgs=False, verbose=False, connection_params={}, data={}
        ):
        # TODO get angry when we have no connection params
        self.port = connection_params['port']
        self.host = connection_params['host']
        self.current_status = 'ready'
        self.connections = set()
        self.send_queue = asyncio.Queue()
        self.recv_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        self.uri = f'ws://{self.host}:{self.port}'
        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)
        self.start()

    def msg(self, msg):
        return f'[{self.id}] {msg}'

    def worker_metadata(self):
        return [ obj.shape for key, obj in self._objects.items() ]

    async def consumer_handler(self, websocket):
        """ receive messages """
        while True:
            if not websocket.open:
                websocket = await websockets.connect(self.uri)
            print("connected", websocket)
            msg = await websocket.recv()
            print(f'[{self.id} | RCV] {msg}')
            await self.recv_queue.put(msg)


    async def producer_handler(self, websocket):
        """ send messages """
        while True:
            msg = await self.send_queue.get()
            await websocket.send(msg)

    async def handler(self, websocket):
        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket))

        done, pending = await asyncio.wait([consumer_task, producer_task]
                                        , return_when=asyncio.FIRST_COMPLETED)
        print("Connection closed, canceling pending tasks")
        for task in pending:
            task.cancel()

    async def connect(self):
        async with websockets.connect(self.uri) as websocket:
            while True:
                if not websocket.open:
                    websocket = await websockets.connect(self.uri)

                print("connected...")
                # here is where we setup the worker datastructures
                await self.handler(websocket)

    def start(self):
        def do_it():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.connect())
        t = Thread(target=do_it)
        t.start()

    def search(self, query):
        return self.send_msg(MSGTYPE.SEARCH, query, location=self)

    def _send_msg(self, message, location):
        print('_send_msg', message)
        return location._recv_msg(message)

    async def write_message(self, message):
        print("1 aww", message)
        await self.send_queue.put(message)
        recv_obj = await self.recv_queue.get()
        print("3 aww", recv_obj)
        return recv_obj

    async def get_chat_id(self, name):
        await asyncio.sleep(3)
        return "chat-%s" % name


    def _recv_msg(self, message):
        print("ruc")
        result = asyncio.get_event_loop().run_until_complete(self.get_chat_id('fooo'))
        print("2 ruc")
        return result


class WebsocketServerWorker(BaseWorker):
    def __init__(
        self,
        hook,
        id=0,
        is_client_worker=False,
        log_msgs=False,
        verbose=False,
        connection_params={},
        data={}
    ):
        # TODO get angry when we have no connection params
        self.port = connection_params["port"]
        self.host = connection_params["host"]
        self.connections = set()
        self.broadcast_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)
        print(" i am server")
        self.start()

    def _send_msg(self, message, location):
        return location._recv_msg(message)

    def _recv_msg(self, message):
        self.broadcast_queue.put((WebsocketCode.REMOTE, message))
        # TODO figure out how to get response from other worker efficiently
        return self.recv_msg(message)  # This line is wrong and must be fixed

    async def consumer_handler(self, websocket, cid):
        while True:
            msg = await websocket.recv()
            print(f"SRV - RCV {msg}")
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
            [consumer_task, producer_task], return_when=asyncio.ALL_COMPLETED
        )
        print("Connection closed, canceling pending tasks")
        for task in pending:
            task.cancel()

    def start(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
