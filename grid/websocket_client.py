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
from syft.messaging.message import Message

from grid import utils as gr_utils
from grid.auth import search_credential
from grid.grid_codes import REQUEST_MSG, RESPONSE_MSG

MODEL_LIMIT_SIZE = (1024 ** 2) * 64  # 64MB


class WebsocketGridClient(WebsocketClientWorker, FederatedClient):
    """Websocket Grid Client."""

    def __init__(
        self,
        hook,
        address,
        id: Union[int, str] = 0,
        auth: dict = None,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        chunk_size: int = MODEL_LIMIT_SIZE,
    ):
        """
        Args:
            hook : a normal TorchHook object
            address : the address this client connects to
            id : the unique id of the worker (string or int)
            auth : An optional dict parameter give authentication credentials,
                to perform authentication process during node connection process.
                If not defined, we'll work with a public grid node version, otherwise,
                we'll work with a private version of the same grid node.
            is_client_worker : An optional boolean parameter to indicate
                whether this worker is associated with an end user client. If
                so, it assumes that the client will maintain control over when
                variables are instantiated or deleted as opposed to handling
                tensor/variable/model lifecycle internally. Set to True if this
                object is not where the objects will be stored, but is instead
                a pointer to a worker that eists elsewhere.
                log_msgs : whether or not all messages should be
                saved locally for later inspection.
            verbose : a verbose option - will print all messages
                sent/received to stdout
        """
        self.address = address

        # Parse address string to get scheme, host and port
        self.secure, self.host, self.port = self.parse_address(address)

        # Initialize WebsocketClientWorker / Federated Client
        super().__init__(
            hook,
            self.host,
            self.port,
            self.secure,
            id,
            is_client_worker,
            log_msgs,
            verbose,
            None,  # initial data
        )

        # Update Node reference using node's Id given by the remote grid node
        self._update_node_reference(self.get_node_id())
        self.credentials = None

        # If auth mode enabled, perform authentication
        if auth:
            self.authenticate(auth)

        self._encoding = "ISO-8859-1"
        self._chunk_size = chunk_size

    @property
    def url(self) -> str:
        """ Get Node URL Address.
            
            Returns:
                address (str) : Node's address.
        """
        if self.port:
            return (
                f"wss://{self.host}:{self.port}"
                if self.secure
                else f"ws://{self.host}:{self.port}"
            )
        else:
            return self.address

    def _update_node_reference(self, new_id: str):
        """ Update worker references changing node id references at hook structure.
            
            Args:
                new_id (str) : New worker ID.
        """
        del self.hook.local_worker._known_workers[self.id]
        self.id = new_id
        self.hook.local_worker._known_workers[new_id] = self

    def parse_address(self, address: str) -> tuple:
        """ Parse Address string to define secure flag and split into host and port.
            
            Args:
                address (str) : Adress of remote worker.
        """
        url = urlparse(address)
        secure = True if url.scheme == "wss" else False
        return (secure, url.hostname, url.port)

    def get_node_id(self) -> str:
        """ Get Node ID from remote node worker
            
            Returns:
                node_id (str) : node id used by remote worker.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.GET_ID}
        response = self._forward_json_to_websocket_server_worker(message)
        return response.get(RESPONSE_MSG.NODE_ID, None)

    def connect_nodes(self, node) -> dict:
        """ Connect two remote workers between each other.
            If this node is authenticated, use the same credentials to authenticate the candidate node.
            
            Args:
                node (WebsocketGridClient) : Node that will be connected with this remote worker.
            Returns:
                node_response (dict) : node response.
        """
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.CONNECT_NODE,
            "address": node.address,
            "id": node.id,
        }
        if node.credentials:
            message["auth"] = node.credentials
        return self._forward_json_to_websocket_server_worker(message)

    def authenticate(self, user: Union[str, dict]):
        """ Perform Authentication Process using credentials grid credentials.
            Grid credentials can be loaded calling the function gr.load_credentials().

            Args:
                user : String containing the username of a loaded credential or a credential's dict.
            Raises:
                RuntimeError : If authentication process fail.
        """
        cred_dict = None
        # If user is a string
        if isinstance(user, str):
            cred = search_credential(user)
            cred_dict = cred.json()
        # If user is a dict structure
        elif isinstance(user, dict):
            cred_dict = user

        if cred_dict:
            # Prepare a authentication request to remote grid node
            cred_dict[REQUEST_MSG.TYPE_FIELD] = REQUEST_MSG.AUTHENTICATE
            response = self._forward_json_to_websocket_server_worker(cred_dict)
            # If succeeded, update node's reference and update client's credential.
            node_id = self._return_bool_result(response, RESPONSE_MSG.NODE_ID)
            if node_id:
                self._update_node_reference(node_id)
                self.credentials = cred_dict
        else:
            raise RuntimeError("Invalid user.")

    def _forward_json_to_websocket_server_worker(self, message: dict) -> dict:
        """ Prepare/send a JSON message to a remote grid node and receive the response.
            
            Args:
                message (dict) : message payload.
            Returns:
                node_response (dict) : response payload.
        """
        self.ws.send(json.dumps(message))
        return json.loads(self.ws.recv())

    def _forward_to_websocket_server_worker(self, message: bin) -> bin:
        """ Prepare/send a binary message to a remote grid node and receive the response.
            Args:
                message (bytes) : message payload.
            Returns:
                node_response (bytes) : response payload.
        """
        self.ws.send_binary(message)
        response = self.ws.recv()
        return response

    def serve_model(
        self,
        model,
        model_id: str = None,
        mpc: bool = False,
        allow_download: bool = False,
        allow_remote_inference: bool = False,
    ) -> bool:
        """ Hosts the model and optionally serve it using a Socket / Rest API.
            
            Args:
                model : A jit model or Syft Plan.
                model_id (str): An integer or string representing the model id used to retrieve the model
                    later on using the Rest API. If this is not provided and the model is a Plan
                    we use model.id, if the model is a jit model we raise an exception.
                allow_download (bool) : If other workers should be able to fetch a copy of this model to run it locally set this to True.
                allow_remote_inference (bool) : If other workers should be able to run inference using this model through a Rest API interface set this True.
            Returns:
                result (bool) : True if model was served sucessfully, raises a RunTimeError otherwise.
            Raises:
                ValueError: if model_id is not provided and model is a jit model (aka does not have an id attribute).
                RunTimeError: if there was a problem during model serving.
        """

        # If the model is a Plan we send the model
        # and host the plan version created after
        # the send operation
        if isinstance(model, sy.Plan):
            # We need to use the same id in the model
            # as in the POST request.
            p_model = model.send(self)
            res_model = p_model
        else:
            res_model = model

        # Send post
        serialized_model = sy.serde.serialize(res_model)

        # If the model is smaller than a chunk size
        if sys.getsizeof(serialized_model) <= self._chunk_size:
            message = {
                REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.HOST_MODEL,
                "encoding": self._encoding,
                "model_id": model_id,
                "allow_download": str(allow_download),
                "allow_remote_inference": str(allow_remote_inference),
                "model": serialized_model.decode(self._encoding),
                "mpc": str(mpc),
            }
            response = self._forward_json_to_websocket_server_worker(message)
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
                        "mpc": str(mpc),
                    },
                )
            )
        return self._return_bool_result(response)

    def run_remote_inference(self, model_id, data):
        """ Run a dataset inference using a remote model.
            
            Args:
                model_id (str) : Model ID.
                data (Tensor) : dataset to be inferred.
            Returns:
                inference (Tensor) : Inference result
            Raises:
                RuntimeError : If an unexpected behavior happen, It will forward the error message.
        """
        serialized_data = sy.serde.serialize(data).decode(self._encoding)
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.RUN_INFERENCE,
            "model_id": model_id,
            "data": serialized_data,
            "encoding": self._encoding,
        }
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response, RESPONSE_MSG.INFERENCE_RESULT)

    def _return_bool_result(self, result, return_key=None):
        if result.get("success"):
            return result[return_key] if return_key is not None else True
        elif result.get("error"):
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
    ) -> bool:
        """Helper function for sending http request to talk to app.

            Args:
                route (str) : App route.
                data (str) : Data to be sent in the request.
                request (str) : Request type (GET, POST, PUT, ...).
                N (int) : Number of tries in case of fail. Default is 10.
                unhexlify (bool) : A boolean indicating if we should try to run unhexlify on the response or not.
                return_response_text (bool): If True return response.text, return raw response otherwise.
            Returns:
                response (bool) : If return_response_text is True return response.text, return raw response otherwise.
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

    def _send_streaming_post(self, route: str, data: dict = None) -> str:
        """ Used to send large models / datasets using stream channel.

            Args:
                route (str) : Service endpoint
                data (dict) : dict with tensors / models to be uploaded.
            Returns:
                response (str) : response from server
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
    def models(self) -> list:
        """ Get models stored at remote grid node.
            
            Returns:
                models (List) : List of models stored in this grid node.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.LIST_MODELS}
        response = self._forward_json_to_websocket_server_worker(message)
        return response.get(RESPONSE_MSG.MODELS, None)

    def delete_model(self, model_id: str) -> bool:
        """ Delete a model previously registered.
            
            Args:
                model_id (String) : ID of the model that will be deleted.
            Returns:
                result (bool) : If succeeded, return True.
        """
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.DELETE_MODEL,
            "model_id": model_id,
        }
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response)

    def download_model(self, model_id: str):
        """ Download a model to run it locally.
        
            Args:
                model_id (str) : ID of the model that will be downloaded.
            Returns:
                model : Model to be downloaded.
            Raises:
                RuntimeError : If an unexpected behavior happen, It will forward the error message.
        """

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
            message = {
                REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.DOWNLOAD_MODEL,
                "model_id": model_id,
            }
            response = self._forward_json_to_websocket_server_worker(message)

            # If we can download model (small models) by sockets
            if response.get("serialized_model", None):
                serialized_model = response["serialized_model"].encode(self._encoding)
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

    def serve_encrypted_model(self, encrypted_model: sy.messaging.plan.Plan) -> bool:
        """ Serve a model in a encrypted fashion using SMPC.

            A wrapper for sending the model. The worker is responsible for sharing the model using SMPC.

            Args:
                encrypted_model (syft.Plan) : A pĺan already shared with workers using SMPC.
            Returns:
                result (bool) : True if model was served successfully, raises a RunTimeError otherwise.
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
            mpc=True,
        )
        return result

    def __str__(self) -> str:
        return "Grid Worker < id: " + self.id + " >"
