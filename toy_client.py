import time
import asyncio
import websockets
import syft as sy
import aiohttp
from syft.serde import serialize
from websocket import create_connection

class FederatedLearningClient:
    def __init__(self, connection_params):
        self.port = connection_params['port']
        self.host = connection_params['host']
        self.current_status = 'ready'
        self.msg_queue = asyncio.Queue()
        self.uri = f'ws://{self.host}:{self.port}'
        self.ws = create_connection(self.uri)

    def send_msg(self, msg):
        self.ws.send(msg)
        return self.ws.recv()
