"""
PySyft Duet (WebRTC)

This class aims to implement the PySyft Duet concept by using WebRTC protocol as a
connection channel in order to allow two different users to establish a direct
connection with high-quality Real-time Communication using private addresses.

The most common example showing how it can be used is the notebook demo example:

Two different jupyter / collab notebooks in different machines using private addresses
behind routers, proxies and firewalls can connect to each other using a full-duplex
channel in order to perform machine learning and data science tasks, working as a client
and server at the same time.

PS 1: You need a signaling server running somewhere.
If you don't know any public address running this service, or want to set up your own
signaling network you can use PyGrid's network app.

For local development you can run:
$ python src/syft/grid/example_nodes/network.py

PS 2: The PyGrid's dev / main branches are still supporting PySyft 0.2.x,
To use this feature you must use the pygrid_0.3.0 branch.
(https://github.com/OpenMined/PyGrid/tree/pygrid_0.3.0)

You can get more details about all this process, in the syft/grid/connections/webrtc.py
source code.
"""

# stdlib
import asyncio

# third party
from nacl.signing import SigningKey

# syft relative
from ...core.io.route import SoloRoute
from ...core.node.domain.client import DomainClient
from ...core.node.domain.domain import Domain
from ...decorators.syft_decorator_impl import syft_decorator
from ..connections.webrtc import WebRTCConnection
from ..duet.signaling_client import SignalingClient
from ..services.signaling_service import AnswerPullRequestMessage
from ..services.signaling_service import InvalidLoopBackRequest
from ..services.signaling_service import OfferPullRequestMessage
from ..services.signaling_service import SignalingAnswerMessage
from ..services.signaling_service import SignalingOfferMessage


class Duet(DomainClient):
    def __init__(
        self,
        node: Domain,
        target_id: str,
        signaling_client: SignalingClient,
        offer: bool = True,
    ):
        # Generate a signing key
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

        # Async Queues
        # These queues will be used in order to enqueue/dequeue
        # messages to be sent to the signaling server.
        self._push_msg_queue: asyncio.Queue = asyncio.Queue()
        self._pull_msg_queue: asyncio.Queue = asyncio.Queue()

        # As we need to inject a node instance inside of
        # a bidirectional connection in order to allow this
        # connection to work as a client and server using the
        # same channel. We need to be aware about forwarding
        # node instance references in order to avoid multiple
        # references to the same object (this makes the garbage
        # collecting difficult).
        # A good solution to avoid this problem is forward just
        # weak references. These references works like a proxy
        # not creating a  strong reference to the object.
        # So, If we delete the real object instance, the
        # garbage collect will call the __del__ method without problem.
        self.node = node

        # WebRTCConnection instance ( Bidirectional Connection )
        self.connection = WebRTCConnection(node=self.node)

        # Client used to exchange signaling messages in order to establish a connection
        # NOTE: In the future it may be a good idea to modularize this client to make
        # it pluggable using different connection protocols.
        self.signaling_client = signaling_client

        # If this peer will not start the signaling process
        if not offer:
            # Start adding an OfferPullRequest in order to verify if
            # the desired address pushed an offer request to connect with you.
            # This will trigger the pull async task to be check signaling notifications
            self._pull_msg_queue.put_nowait(
                OfferPullRequestMessage(
                    address=self.signaling_client.address,
                    target_peer=target_id,
                    host_peer=self.signaling_client.duet_id,
                    reply_to=self.signaling_client.address,
                )
            )
        else:
            # Push a WebRTC offer request to the address.
            self.send_offer(target_id=target_id)

        # This flag is used in order to finish the signaling process gracefully
        # While self._available is True, the pull/push tasks will be running
        # This flag will be setted to false when:
        # 1 - End of the signaling process (checked by _update_availability()).
        # 2 - Any Exception raised during these tasks.
        self._available = True

        # This attribute will be setted during the signaling messages exchange,
        # and used to create a SoloRoute for the both sides.
        self._client_metadata = ""

        # Start async tasks and wait until one of them finishes.
        # As mentioned before, these tasks can be finished by two reasons:
        # 1 - Signaling process ends
        # 2 - Unexpected Exception
        asyncio.run(self.notify())

        # If client_metadata != None, then the connection was created successfully.
        if self._client_metadata:
            # Deserialize client's metadata in order to obtain
            # PySyft's location structure
            (
                spec_location,
                name,
                client_id,
            ) = DomainClient.deserialize_client_metadata_from_node(
                metadata=self._client_metadata
            )

            # Create a SoloRoute
            route = SoloRoute(destination=spec_location, connection=self.connection)

            # Intialize the super class
            super().__init__(
                domain=spec_location,
                name=name,
                routes=[route],
                signing_key=self.signing_key,
                verify_key=self.verify_key,
            )
            self.connection._client_address = self.address
        # If client_metada is None, then an exception occurred during the process
        # The exception has been caught and saved in self._exception
        else:
            # NOTE: Maybe we should create a custom exception type.
            raise Exception(
                f"Something went wrong during the Duet init process. {self._exception}"
            )

    @syft_decorator(typechecking=True)
    async def notify(self) -> None:
        # Enqueue Pull/Push async tasks
        push_task = asyncio.ensure_future(self.push())
        pull_task = asyncio.ensure_future(self.pull())

        # Wait until one of them finishes
        done, pending = await asyncio.wait(
            [pull_task, push_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Finish the pending one.
        for task in pending:
            task.cancel()

    def close(self) -> None:
        self.connection.close()

    @syft_decorator(typechecking=True)
    async def push(self) -> None:
        # This task is responsible for pushing offer/answer messages.
        try:
            while self._available:
                # If push_msg_queue is empty,
                # give up task queue priority, giving
                # computing time to the next task.
                msg = await self._push_msg_queue.get()

                # If self.push_msg_queue.get() returned a message (SignalingOfferMessage,SignalingAnswerMessage)
                # send it to the signaling server.
                self.signaling_client.send_immediate_msg_without_reply(msg=msg)
        except Exception as e:
            # If any exception raises, set the self._available flag to False
            # in order to finish gracefully all the async tasks and save the exception.
            self._available = False
            self._exception: Exception = e

    @syft_decorator(typechecking=True)
    async def pull(self) -> None:
        try:
            while self._available:
                # If pull_msg_queue is empty,
                # give up task queue priority, giving
                # computing time to the next task.
                msg = await self._pull_msg_queue.get()

                # If self.push_msg_queue.get() returned a message (OfferPullRequestMessage,AnswerPullRequestMessage)
                # send it to the signaling server.
                _response = self.signaling_client.send_immediate_msg_with_reply(msg=msg)

                task = None
                # If Signaling Offer Message was found
                if isinstance(_response, SignalingOfferMessage):
                    task = self._send_answer

                # If Signaling Answer Message was found
                elif isinstance(_response, SignalingAnswerMessage):
                    task = self._ack

                # If LoopBack Message it was a loopback request
                elif isinstance(_response, InvalidLoopBackRequest):
                    raise Exception(
                        "You can't perform p2p connection using your current node address as a destination peer."
                    )

                # If Signaling Message weren't found
                else:
                    # Just enqueue the request to be processed later.
                    self._pull_msg_queue.put_nowait(msg)

                # If we have tasks to execute
                if task:
                    # Execute task using the received message.
                    await task(msg=_response)

                # Checks if the signaling process is over.
                self._available = self._update_availability()
                await asyncio.sleep(0.5)
        except Exception as e:
            # If any exception raises, set the self._available flag to False
            # in order to finish gracefully all the async tasks and save the exception.
            self._available = False
            self._exception = e

    @syft_decorator(typechecking=True)
    def send_offer(self, target_id: str) -> None:
        """Starts a new signaling process by creating a new
        offer message and pushing it to the Signaling Server."""

        # Generates an offer request payload containing
        # local network description data/metadata (IP, MAC, Mask, etc...)
        payload = asyncio.run(self.connection._set_offer())

        # Creates a PySyft's SignalingOfferMessage
        signaling_offer = SignalingOfferMessage(
            address=self.signaling_client.address,  # Target's address
            payload=payload,  # Offer Payload
            host_metadata=self.node.get_metadata_for_client(),  # Own Node Metadata
            target_peer=target_id,
            host_peer=self.signaling_client.duet_id,  # Own Node ID
        )

        # Enqueue it in push msg queue to be sent to the signaling server.
        self._push_msg_queue.put_nowait(signaling_offer)

        # Create/enqueue a new AnswerPullRequest in order to wait for signaling response.
        self._pull_msg_queue.put_nowait(
            AnswerPullRequestMessage(
                address=self.signaling_client.address,
                target_peer=target_id,
                host_peer=self.signaling_client.duet_id,
                reply_to=self.signaling_client.address,
            )
        )

    @syft_decorator(typechecking=True)
    async def _send_answer(self, msg: SignalingOfferMessage) -> None:
        """Process SignalingOfferMessage and create a new
        SignalingAnswerMessage as a response"""

        # Process received offer message updating target's remote address
        # Generates an answer request payload containing
        # local network description data/metadata (IP, MAC, Mask, etc...)
        payload = asyncio.run(self.connection._set_answer(payload=msg.payload))

        # Save remote node's metadata in roder to create a SoloRoute.
        self._client_metadata = msg.host_metadata

        # Create a new SignalingAnswerMessage
        signaling_answer = SignalingAnswerMessage(
            address=self.signaling_client.address,
            payload=payload,  # Signaling answer payload
            host_metadata=self.node.get_metadata_for_client(),  # Own Node Metadata
            target_peer=msg.host_peer,  # Remote Node ID
            host_peer=self.signaling_client.duet_id,
        )

        # Enqueue it in the push msg queue to be sent to the signaling server.
        await self._push_msg_queue.put(signaling_answer)

    @syft_decorator(typechecking=True)
    async def _ack(self, msg: SignalingAnswerMessage) -> None:
        """Last signaling message, stores remote Node
        metadata and updates target's remote address"""

        # Save remote node's metadata in roder to create a SoloRoute.
        self._client_metadata = msg.host_metadata

        # Process received offer message updating target's remote address
        await self.connection._process_answer(payload=msg.payload)

    @syft_decorator(typechecking=True)
    def _update_availability(self) -> bool:
        """Method used to check if the signaling process is over.
        :return: Boolean flag, True if it's NOT over, and False if it's over.
        :rtype: Boolean
        """
        return (
            not self._pull_msg_queue.empty()
            and self.connection.peer_connection is not None
        )
