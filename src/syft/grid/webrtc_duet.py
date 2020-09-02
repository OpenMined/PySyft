import asyncio
from http_client import HTTPClient
import syft as sy
from syft.core.io.address import Address
from syft.core.node.domain.client import DomainClient
from syft.core.node.domain.domain import Domain

from syft.core.io.route import SoloRoute

from nacl.signing import SigningKey
from nacl.signing import VerifyKey

from syft.grid.services.signaling_service import (
    SignalingOfferMessage,
    SignalingAnswerMessage,
    OfferPullRequestMessage,
    AnswerPullRequestMessage
)

from syft.grid.connections.webrtc import WebRTCConnection
import asyncio
import nest_asyncio
import weakref

nest_asyncio.apply()


class Duet(DomainClient):

    def __init__(
            self,
            node: Domain,
            address: Address,
            network_client: HTTPClient,
            offer: bool = True):
        
        # Generate a signing key
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

        # Async Queues
        self._push_msg_queue = asyncio.Queue()
        self._pull_msg_queue = asyncio.Queue()

        self.node = weakref.proxy(node)

        # WebRTCConnection instance ( Bidirectional Connection )
        self.connection = WebRTCConnection(self.node)

        # Client used to exchange signaling messages in order to establish a connection
        self.network_client = network_client
        
        if not offer:
            # Start adding an OfferPullRequest in order to verify if anyone wants to connect with you.
            self._pull_msg_queue.put_nowait(
                    OfferPullRequestMessage(address=address, reply_to=self.node.address)
            )
        else:
            # Push a WebRTC offer to the address.
            self.send_offer(address)
        
        self._available = True

        # Wait until the WebRTCConnection procotol finishes.
        asyncio.run( self.notify() )
        
        spec_location, name, client_id = DomainClient.deserialize_client_metadata_from_node(metadata=self._client_metadata)
        route = SoloRoute(destination=spec_location, connection=self.connection)

        super().__init__(
            domain=spec_location,
            name=name,
            routes=[route],
            signing_key=self.signing_key,
            verify_key=self.verify_key,
        )
        self.connection._client_address = self.address
        #print("My Self address name: ", self.address.name)

    async def notify(self):
        push_task = asyncio.ensure_future(self.push(route="/signaling/push"))
        pull_task = asyncio.ensure_future(self.pull(route="/signaling/pull"))

        # Wait until one of them finishes
        done, pending = await asyncio.wait(
            [pull_task, push_task], return_when=asyncio.FIRST_COMPLETED
        )
        
        # Finish the pending one.
        for task in pending:
            task.cancel()

    async def push(self, route: str):
        try:
            while self._available:
                msg = await self._push_msg_queue.get()
                self.network_client.send_immediate_msg_with_reply(msg=msg, route=route)
        except Exception:
           self._available = False
    
    async def pull(self, route: str):
        try:
            while self._available:
                msg = await self._pull_msg_queue.get()
                _response = self.network_client.send_immediate_msg_with_reply(msg=msg, route=route)
                
                task = None
                if isinstance(_response, SignalingOfferMessage):
                    task = self._send_answer
                elif isinstance(_response, SignalingAnswerMessage):
                    task = self._ack
                else:
                    self._pull_msg_queue.put_nowait(msg)

                if task:
                    await task(msg=_response)

                self._available = self._update_availability()
        except Exception:
                self._available = False
    
    def send_offer(self, address: Address):
        payload = asyncio.run(self.connection._set_offer())
        signaling_offer = SignalingOfferMessage(
            address=address,
            payload=payload,
            target_metadata=self.node.get_metadata_for_client(),
            reply_to=self.node.address
        )
        self._push_msg_queue.put_nowait(signaling_offer)
        
        self._pull_msg_queue.put_nowait(
                AnswerPullRequestMessage(address=address, reply_to=self.node.address)
        )
        
    async def _send_answer(self, msg: SignalingOfferMessage):
        payload = asyncio.run(self.connection._set_answer(msg.payload) )
        self._client_metadata = msg.target_metadata
        signaling_answer = SignalingAnswerMessage(
                address=msg.reply_to,
                payload=payload,
                target_metadata=self.node.get_metadata_for_client(),
                reply_to=self.node.address)
        await self._push_msg_queue.put(signaling_answer)

    async def _ack(self, msg: SignalingAnswerMessage):
        self._client_metadata = msg.target_metadata
        await self.connection._process_answer(msg.payload)


    def _update_availability(self) -> bool:
        return (not self._pull_msg_queue.empty() and self.connection.peer_connection)

