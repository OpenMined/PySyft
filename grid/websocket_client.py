import binascii
import json
import os
import requests
from requests_toolbelt.multipart import encoder, decoder
import sys

from typing import List, Union
from urllib.parse import urlparse

import websocket
import torch
from gevent import monkey

import syft as sy
from syft.messaging.message import Message, PlanCommandMessage
from syft.generic.tensor import AbstractTensor
from syft.workers.base import BaseWorker
from syft import WebsocketClientWorker
from syft.federated.federated_client import FederatedClient
from syft.codes import MSGTYPE
from syft.messaging.message import Message

from grid import utils as gr_utils


MODEL_LIMIT_SIZE = (1024 ** 2) * 64  # 64MB


class WebsocketGridClient(WebsocketClientWorker, FederatedClient):
    """Websocket Grid Client."""

    def __init__(
        self,
        hook,
        address,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
        chunk_size: int = MODEL_LIMIT_SIZE,
    ):
        """
        Args:
            hook : a normal TorchHook object
            address : the address this client connects to
            id : the unique id of the worker (string or int)
            log_msgs : whether or not all messages should be
                saved locally for later inspection.
            verbose : a verbose option - will print all messages
                sent/received to stdout
            data : any initial tensors the server should be
                initialized with (such as datasets)
        """
        self.address = address
        self.secure, self.host, self.port = self.parse_address(address)
        super().__init__(
            hook,
            self.host,
            self.port,
            self.secure,
            id,
            is_client_worker,
            log_msgs,
            verbose,
            data,
        )
        self.id = self.get_node_id()
        self._encoding = "ISO-8859-1"
        self._chunk_size = chunk_size

    @property
    def url(self):
        if self.port:
            return (
                f"wss://{self.host}:{self.port}"
                if self.secure
                else f"ws://{self.host}:{self.port}"
            )
        else:
            return self.address

    def parse_address(self, address):
        url = urlparse(address)
        secure = True if url.scheme == "wss" else False
        return (secure, url.hostname, url.port)

    def get_node_id(self):
        message = {"type": "get-id"}
        self.ws.send(json.dumps(message))
        response = json.loads(self.ws.recv())
        return response["id"]

    def connect_nodes(self, node):
        message = {"type": "connect-node", "address": node.address, "id": node.id}
        self.ws.send(json.dumps(message))
        return json.loads(self.ws.recv())

    def _forward_to_websocket_server_worker(self, message: bin) -> bin:
        self.ws.send_binary(message)
        response = self.ws.recv()
        return response

    def serve_model(
        self,
        model,
        model_id: str = None,
        allow_download: bool = False,
        allow_remote_inference: bool = False,
    ):
        if model_id is None:
            if isinstance(model, sy.Plan):
                model_id = model.id
            else:
                raise ValueError("Model id argument is mandatory for jit models.")

        # If the model is a Plan we send the model
        # and host the plan version created after
        # the send operation
        if isinstance(model, sy.Plan):
            # We need to use the same id in the model
            # as in the POST request.
            model.id = model_id
            model.send(self)
            res_model = model.ptr_plans[self.id]
        else:
            res_model = model

        # Send post
        serialized_model = sy.serde.serialize(res_model)

        # If the model is smaller than a chunk size
        if sys.getsizeof(serialized_model) <= self._chunk_size:
            self.ws.send(
                json.dumps(
                    {
                        "type": "host-model",
                        "encoding": self._encoding,
                        "model_id": model_id,
                        "allow_download": str(allow_download),
                        "allow_remote_inference": str(allow_remote_inference),
                        "model": serialized_model.decode(self._encoding),
                    }
                )
            )
            response = json.loads(self.ws.recv())
            if response["success"]:
                return True
            else:
                raise RuntimeError(response["error"])
        else:
            # HUGE Models
            # TODO: Replace to websocket protocol
            response = json.loads(
                self._send_streaming_post(
                    "serve-model/",
                    data={
                        "model": (
                            model_id,
                            serialized_model,
                            "application/octet-stream",
                        ),
                        "encoding": self._encoding,
                        "model_id": model_id,
                        "allow_download": str(allow_download),
                        "allow_remote_inference": str(allow_remote_inference),
                    },
                )
            )
            return self._return_bool_result(response)

    def _generate_model_chunks(self, serialized_model):
        return [
            serialized_model[i : i + self._chunk_size]
            for i in range(0, len(serialized_model), self._chunk_size)
        ]

    def run_remote_inference(self, model_id, data, N: int = 1):
        serialized_data = sy.serde.serialize(data).decode(self._encoding)
        payload = {
            "type": "run-inference",
            "model_id": model_id,
            "data": serialized_data,
            "encoding": self._encoding,
        }
        self.ws.send(json.dumps(payload))
        response = json.loads(self.ws.recv())
        if response["success"]:
            return torch.tensor(response["prediction"])
        else:
            raise RuntimeError(response["error"])

    def search(self, *query):
        # Prepare a message requesting the websocket server to search among its objects
        message = Message(MSGTYPE.SEARCH, query)
        serialized_message = sy.serde.serialize(message)

        # Send the message and return the deserialized response.
        response = self._recv_msg(serialized_message)
        return sy.serde.deserialize(response)

    def _return_bool_result(self, result, return_key=None):
        if result["success"]:
            return result[return_key] if return_key is not None else True
        elif result["error"]:
            raise RuntimeError(result["error"])
        else:
            raise RuntimeError(
                "Something went wrong, check the server logs for more information."
            )

    def _send_http_request(
        self,
        route,
        data,
        request,
        N: int = 10,
        unhexlify: bool = True,
        return_response_text: bool = True,
    ):
        """Helper function for sending http request to talk to app.

        Args:
            route: App route.
            data: Data to be sent in the request.
            request: Request type (GET, POST, PUT, ...).
            N: Number of tries in case of fail. Default is 10.
            unhexlify: A boolean indicating if we should try to run unhexlify on the response or not.
            return_response_text: If True return response.text, return raw response otherwise.
        Returns:
            If return_response_text is True return response.text, return raw response otherwise.
        """
        url = (
            f"https://{self.host}:{self.port}"
            if self.secure
            else f"http://{self.host}:{self.port}"
        )
        url = os.path.join(url, "{}".format(route))
        r = request(url, data=data) if data else request(url)
        r.encoding = self._encoding
        response = r.text if return_response_text else r

        # Try to request the message `N` times.
        for _ in range(N):
            try:
                if unhexlify:
                    response = binascii.unhexlify(response[2:-1])
                return response
            except:
                if self.verbose:
                    print(response)
                response = None
                r = request(url, data=data) if data else request(url)
                response = r.text

        return response

    def _send_streaming_post(self, route, data=None):
        """ Used to send large models / datasets using stream channel.

            Args:
                route : Service endpoint
                data : tensors / models to be uploaded.
            Return:
                response : response from server
        """
        # Build URL path
        url = os.path.join(self.address, "{}".format(route))

        # Send data
        session = requests.Session()
        form = encoder.MultipartEncoder(data)
        headers = {"Prefer": "respond-async", "Content-Type": form.content_type}
        resp = session.post(url, headers=headers, data=form)
        session.close()
        return resp.content

    def _send_get(self, route, data=None, **kwargs):
        return self._send_http_request(route, data, requests.get, **kwargs)

    @property
    def models(self, N: int = 1):
        self.ws.send(json.dumps({"type": "list-models"}))
        response = json.loads(self.ws.recv())
        return response["models"]

    def delete_model(self, model_id):
        message = {"type": "delete-model", "model_id": model_id}
        self.ws.send(json.dumps(message))
        response = json.loads(self.ws.recv())
        return self._return_bool_result(response)

    def download_model(self, model_id: str):
        """Downloads a model to run it  locally."""

        def _is_large_model(result):
            return "multipart/form-data" in result.headers["Content-Type"]

        # Check if we can get a copy of this model
        # TODO: We should remove this endpoint and verify download permissions during /get_model request / fetch_plan.
        # If someone performs request/fetch outside of this function context, they'll get the model.
        result = json.loads(
            self._send_get("is_model_copy_allowed/{}".format(model_id), unhexlify=False)
        )

        if not result["success"]:
            raise RuntimeError(result["error"])

        try:
            # If the model is a plan we can just call fetch
            return sy.hook.local_worker.fetch_plan(model_id, self, copy=True)
        except AttributeError:
            # Try download model by websocket channel
            self.ws.send(json.dumps({"type": "download-model", "model_id": model_id}))
            response = json.loads(self.ws.recv())

            # If we can download model (small models) by sockets
            if response.get("serialized_model", None):
                serialized_model = result["serialized_model"].encode(self._encoding)
                model = sy.serde.deserialize(serialized_model)
                return model

            # If it isn't possible, try download model by HTTP protocol
            # TODO: This flow need to be removed when sockets can download huge models
            result = self._send_get(
                "get_model/{}".format(model_id),
                unhexlify=False,
                return_response_text=False,
            )
            if result:
                if _is_large_model(result):
                    # If model is large, receive it by a stream channel
                    multipart_data = decoder.MultipartDecoder.from_response(result)
                    model_bytes = b"".join(
                        [part.content for part in multipart_data.parts]
                    )
                    serialized_model = model_bytes.decode("utf-8").encode(
                        self._encoding
                    )
                else:
                    # If model is small, receive it by a standard json
                    result = json.loads(result.text)
                    serialized_model = result["serialized_model"].encode(self._encoding)

                model = sy.serde.deserialize(serialized_model)
                return model
            else:
                raise RuntimeError(
                    "There was a problem while getting the model, check the server logs for more information."
                )

    def serve_encrypted_model(self, encrypted_model: sy.messaging.plan.Plan):
        """Serve a model in a encrypted fashion using SMPC.

        A wrapper for sending the model. The worker is responsible for sharing the model using SMPC.

        Args:
            encrypted_model: A pĺan already shared with workers using SMPC.

        Returns:
            True if model was served successfully, raises a RunTimeError otherwise.
        """
        # Send the model
        encrypted_model.send(self)
        res_model = encrypted_model.ptr_plans[self.id]

        # Serve the model so we can have a copy saved in the database
        serialized_model = sy.serde.serialize(res_model).decode(self._encoding)
        result = self.serve_model(
            serialized_model,
            res_model.id,
            allow_download=True,
            allow_remote_inference=False,
        )
        return result
