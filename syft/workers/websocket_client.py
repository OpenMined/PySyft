import binascii
from typing import Union
from typing import List

import torch
import websocket
import websockets
import logging
import ssl
import time

import syft as sy
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import SearchMessage
from syft.generic.tensor import AbstractTensor
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
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        self.port = port
        self.host = host

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
        args = {"max_size": None, "timeout": TIMEOUT_INTERVAL, "url": self.url}

        if self.secure:
            args["sslopt"] = {"cert_reqs": ssl.CERT_NONE}

        self.ws = websocket.create_connection(**args)

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
            self.ws = websocket.create_connection(self.url, max_size=None, timeout=TIMEOUT_INTERVAL)
            logger.warning("Created new websocket connection")
            time.sleep(0.1)
            response = self._forward_to_websocket_server_worker(message)
            if not self.ws.connected:
                raise RuntimeError(
                    "Websocket connection closed and creation of new connection failed."
                )
        return response

    def _send_msg_and_deserialize(self, command_name: str, *args, **kwargs):
        message = self.create_message_execute_command(
            command_name=command_name, command_owner="self", *args, **kwargs
        )

        # Send the message and return the deserialized response.
        serialized_message = sy.serde.serialize(message)
        response = self._send_msg(serialized_message)
        return sy.serde.deserialize(response)

    def list_objects_remote(self):
        return self._send_msg_and_deserialize("list_objects")

    def objects_count_remote(self):
        return self._send_msg_and_deserialize("objects_count")

    def clear_objects_remote(self):
        return self._send_msg_and_deserialize("clear_objects", return_self=False)

    async def async_fit(self, dataset_key: str, return_ids: List[int] = None):
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
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_message_execute_command(
                command_name="fit",
                command_owner="self",
                return_ids=return_ids,
                dataset_key=dataset_key,
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)
            await websocket.send(str(binascii.hexlify(serialized_message)))
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage((return_ids[0], None, ""))
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

        msg = ObjectRequestMessage((return_ids[0], None, ""))
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
        out += " #objects local:" + str(len(self._objects))
        out += " #objects remote: " + str(self.objects_count_remote())
        out += ">"
        return out
