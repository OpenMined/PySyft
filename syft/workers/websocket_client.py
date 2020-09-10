import binascii
from typing import Union
from typing import List

import torch
import websocket
import websockets
import logging
import ssl
import time
import asyncio

import syft as sy

from syft.exceptions import ResponseSignatureError

from syft.messaging.message import Message
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import SearchMessage
from syft.messaging.message import TensorCommandMessage
from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.workers.base import BaseWorker

logger = logging.getLogger(__name__)

TIMEOUT_INTERVAL = 60


class WebsocketClientWorker(BaseWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        secure: bool = False,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
        timeout: int = None,
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        self.port = port
        self.host = host
        self.timeout = TIMEOUT_INTERVAL if timeout is None else timeout

        super().__init__(
            hook=hook,
            id=id,
            data=data,
            is_client_worker=is_client_worker,
            log_msgs=log_msgs,
            verbose=verbose,
        )

        # creates the connection with the server which gets held open until the
        # WebsocketClientWorker is garbage collected.
        # Secure flag adds a secure layer applying cryptography and authentication
        self.secure = secure
        self.ws = None
        self.connect()

    @property
    def url(self):
        return f"wss://{self.host}:{self.port}" if self.secure else f"ws://{self.host}:{self.port}"

    def connect(self):
        args_ = {"max_size": None, "timeout": self.timeout, "url": self.url}

        if self.secure:
            args_["sslopt"] = {"cert_reqs": ssl.CERT_NONE}

        self.ws = websocket.create_connection(**args_)
        self._log_msgs_remote(self.log_msgs)

    def close(self):
        self.ws.shutdown()

    def search(self, query):
        # Prepare a message requesting the websocket server to search among its objects
        message = SearchMessage(query)
        serialized_message = sy.serde.serialize(message)
        # Send the message and return the deserialized response.
        response = self._send_msg(serialized_message)
        return sy.serde.deserialize(response)

    def _send_msg(self, message: bin, location=None) -> bin:
        return self._recv_msg(message)

    def _forward_to_websocket_server_worker(self, message: bin) -> bin:
        """
        Note: Is subclassed by the node client when you use the GridNode
        """
        self.ws.send(str(binascii.hexlify(message)))
        response = binascii.unhexlify(self.ws.recv()[2:-1])
        return response

    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketServerWorker"""
        response = self._forward_to_websocket_server_worker(message)
        if not self.ws.connected:
            logger.warning("Websocket connection closed (worker: %s)", self.id)
            self.ws.shutdown()
            time.sleep(0.1)
            # Avoid timing out on the server-side
            self.ws = websocket.create_connection(self.url, max_size=None, timeout=self.timeout)
            logger.warning("Created new websocket connection")
            time.sleep(0.1)
            response = self._forward_to_websocket_server_worker(message)
            if not self.ws.connected:
                raise RuntimeError(
                    "Websocket connection closed and creation of new connection failed."
                )
        return response

    def _send_msg_and_deserialize(self, command_name: str, *args, **kwargs):
        message = self.create_worker_command_message(command_name=command_name, *args, **kwargs)

        # Send the message and return the deserialized response.
        serialized_message = sy.serde.serialize(message)
        response = self._send_msg(serialized_message)
        return sy.serde.deserialize(response)

    def list_tensors_remote(self):
        return self._send_msg_and_deserialize("list_tensors")

    def tensors_count_remote(self):
        return self._send_msg_and_deserialize("tensors_count")

    def list_objects_remote(self):
        return self._send_msg_and_deserialize("list_objects")

    def objects_count_remote(self):
        return self._send_msg_and_deserialize("objects_count")

    def _get_msg_remote(self, index):
        return self._send_msg_and_deserialize("_get_msg", index=index)

    def _log_msgs_remote(self, value=True):
        return self._send_msg_and_deserialize("_log_msgs", value=value)

    def clear_objects_remote(self):
        return self._send_msg_and_deserialize("clear_objects", return_self=False)

    async def async_dispatch(self, workers, commands):
        results = await asyncio.gather(
            *[
                worker.async_send_command(message=command)
                for worker, command in zip(workers, commands)
            ]
        )
        return results

    async def async_send_msg(self, message: Message) -> object:
        """Asynchronous version of send_msg."""
        if self.verbose:
            print("async_send_msg", message)

        async with websockets.connect(
            self.url, timeout=self.timeout, max_size=None, ping_timeout=self.timeout
        ) as websocket:
            # Step 1: serialize the message to a binary
            bin_message = sy.serde.serialize(message, worker=self)

            # Step 2: send the message
            await websocket.send(bin_message)

            # Step 3: wait for a response
            bin_response = await websocket.recv()

            # Step 4: deserialize the response
            response = sy.serde.deserialize(bin_response, worker=self)

        return response

    async def async_send_command(
        self, message: tuple, return_ids: str = None, return_value: bool = False
    ) -> Union[List[PointerTensor], PointerTensor]:
        """
        Sends a command through a message to the server part attached to the client
        Args:
            message: A tuple representing the message being sent.
            return_ids: A list of strings indicating the ids of the
                tensors that should be returned as response to the command execution.
        Returns:
            A list of PointerTensors or a single PointerTensor if just one response is expected.
        Note: this is the async version of send_command, with the major difference that you
        directly call it on the client worker (so we don't have the recipient kw argument)
        """

        if return_ids is None:
            return_ids = (sy.ID_PROVIDER.pop(),)

        name, target, args_, kwargs_ = message

        # Close the existing websocket connection in order to open a asynchronous connection
        self.close()
        try:
            message = TensorCommandMessage.computation(
                name, target, args_, kwargs_, return_ids, return_value
            )
            ret_val = await self.async_send_msg(message)

        except ResponseSignatureError as e:
            ret_val = None
            return_ids = e.ids_generated
        # Reopen the standard connection
        self.connect()

        if ret_val is None or type(ret_val) == bytes:
            responses = []
            for return_id in return_ids:
                response = PointerTensor(
                    location=self,
                    id_at_location=return_id,
                    owner=sy.local_worker,
                    id=sy.ID_PROVIDER.pop(),
                )
                responses.append(response)

            if len(return_ids) == 1:
                responses = responses[0]
        else:
            responses = ret_val
        return responses

    async def async_fit(self, dataset_key: str, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit function on the remote location.

        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.

        Returns:
            See return value of the FederatedClient.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
            self.url, timeout=self.timeout, max_size=None, ping_timeout=self.timeout
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit", return_ids=return_ids, dataset_key=dataset_key, device=device
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)
            await websocket.send(str(binascii.hexlify(serialized_message)))
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)

        # Return the deserialized response.
        return sy.serde.deserialize(response)

    def fit(self, dataset_key: str, **kwargs):
        """Call the fit() method on the remote worker (WebsocketServerWorker instance).

        Note: The argument return_ids is provided as kwargs as otherwise there is a miss-match
        with the signature in VirtualWorker.fit() method. This is important to be able to switch
        between virtual and websocket workers.

        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            **kwargs:
                return_ids: List[str]
        """
        return_ids = kwargs["return_ids"] if "return_ids" in kwargs else [sy.ID_PROVIDER.pop()]

        self._send_msg_and_deserialize("fit", return_ids=return_ids, dataset_key=dataset_key)

        msg = ObjectRequestMessage(return_ids[0], None, "")
        # Send the message and return the deserialized response.
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)
        return sy.serde.deserialize(response)

    def evaluate(
        self,
        dataset_key: str,
        return_histograms: bool = False,
        nr_bins: int = -1,
        return_loss=True,
        return_raw_accuracy: bool = True,
        device: str = "cpu",
    ):
        """Call the evaluate() method on the remote worker (WebsocketServerWorker instance).

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            return_histograms: If True, calculate the histograms of predicted classes.
            nr_bins: Used together with calculate_histograms. Provide the number of classes/bins.
            return_loss: If True, loss is calculated additionally.
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions
            device: "cuda" or "cpu"

        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
        """

        return self._send_msg_and_deserialize(
            "evaluate",
            dataset_key=dataset_key,
            return_histograms=return_histograms,
            nr_bins=nr_bins,
            return_loss=return_loss,
            return_raw_accuracy=return_raw_accuracy,
            device=device,
        )

    def __str__(self):
        """Returns the string representation of a Websocket worker.

        A to-string method for websocket workers that includes information from the websocket server

        Returns:
            The Type and ID of the worker

        """
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " #tensors local:" + str(len(self.object_store._tensors))
        out += " #tensors remote: " + str(self.tensors_count_remote())
        out += ">"
        return out
