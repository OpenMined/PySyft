import json
import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

from typing import Union
from urllib.parse import urlparse

# Syft imports
from syft.serde import serialize
from syft.version import __version__
from syft.execution.plan import Plan
from syft.codes import REQUEST_MSG, RESPONSE_MSG
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker


class DataCentricFLClient(WebsocketClientWorker):
    """Federated Node Client."""

    def __init__(
        self,
        hook,
        address,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        encoding: str = "ISO-8859-1",
        timeout: int = None,
    ):
        """
        Args:
            hook : a normal TorchHook object.
            address : Address used to connect with remote node.
            id : the unique id of the worker (string or int)
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
                sent/received to stdout.
            encoding : Encoding pattern used to send/retrieve models.
            timeout : connection's timeout with the remote worker.
        """
        self.address = address
        self.encoding = encoding

        # Parse address string to get scheme, host and port
        self.secure, self.host, self.port = self._parse_address(address)

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
            timeout,
        )

        # Update Node reference using node's Id given by the remote node
        self._update_node_reference(self._get_node_infos())

    @property
    def url(self) -> str:
        """Get Node URL Address.

        Returns:
            address (str) : Node's address.
        """
        if self.port:
            return (
                f"wss://{self.host}:{self.port}" if self.secure else f"ws://{self.host}:{self.port}"
            )
        else:
            return self.address

    @property
    def models(self) -> list:
        """Get models stored at remote node.

        Returns:
            models (List) : List of models stored in this node.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.LIST_MODELS}
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response, RESPONSE_MSG.MODELS)

    def _update_node_reference(self, new_id: str):
        """Update worker references changing node id references at hook structure.

        Args:
            new_id (str) : New worker ID.
        """
        del self.hook.local_worker._known_workers[self.id]
        self.id = new_id
        self.hook.local_worker._known_workers[new_id] = self

    def _parse_address(self, address: str) -> tuple:
        """Parse Address string to define secure flag and split into host and port.

        Args:
            address (str) : Adress of remote worker.
        """
        url = urlparse(address)
        secure = True if url.scheme == "wss" else False
        return (secure, url.hostname, url.port)

    def _get_node_infos(self) -> str:
        """Get Node ID from remote node worker

        Returns:
            node_id (str) : node id used by remote worker.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.GET_ID}
        response = self._forward_json_to_websocket_server_worker(message)
        node_version = response.get(RESPONSE_MSG.SYFT_VERSION, None)
        if node_version != __version__:
            raise RuntimeError(
                "Library version mismatch, The PySyft version of your environment is "
                + __version__
                + " the Grid Node Syft version is "
                + node_version
            )

        return response.get(RESPONSE_MSG.NODE_ID, None)

    def _forward_json_to_websocket_server_worker(self, message: dict) -> dict:
        """Prepare/send a JSON message to a remote node and receive the response.

        Args:
            message (dict) : message payload.
        Returns:
            node_response (dict) : response payload.
        """
        self.ws.send(json.dumps(message))
        return json.loads(self.ws.recv())

    def _forward_to_websocket_server_worker(self, message: bin) -> bin:
        """Send a bin message to a remote node and receive the response.

        Args:
            message (bytes) : message payload.
        Returns:
            node_response (bytes) : response payload.
        """
        self.ws.send_binary(message)
        response = self.ws.recv()
        return response

    def _return_bool_result(self, result, return_key=None):
        if result.get(RESPONSE_MSG.SUCCESS):
            return result[return_key] if return_key is not None else True
        elif result.get(RESPONSE_MSG.ERROR):
            raise RuntimeError(result[RESPONSE_MSG.ERROR])
        else:
            raise RuntimeError("Something went wrong.")

    def connect_nodes(self, node) -> dict:
        """Connect two remote workers between each other.

        Args:
            node (WebsocketFederatedClient) : Node that will be connected with this remote worker.
        Returns:
            node_response (dict) : node response.
        """
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.CONNECT_NODE,
            "address": node.address,
            "id": node.id,
        }
        return self._forward_json_to_websocket_server_worker(message)

    def serve_model(
        self,
        model,
        model_id: str = None,
        mpc: bool = False,
        allow_download: bool = False,
        allow_remote_inference: bool = False,
    ):
        """Hosts the model and optionally serve it using a Socket / Rest API.
        Args:
            model : A jit model or Syft Plan.
            model_id (str): An integer/string representing the model id.
            If it isn't provided and the model is a Plan we use model.id,
            if the model is a jit model we raise an exception.
            allow_download (bool) : Allow to copy the model to run it locally.
            allow_remote_inference (bool) : Allow to run remote inferences.
        Returns:
            result (bool) : True if model was served sucessfully.
        Raises:
            ValueError: model_id isn't provided and model is a jit model.
            RunTimeError: if there was a problem during model serving.
        """

        # If the model is a Plan we send the model
        # and host the plan version created after
        # the send action
        if isinstance(model, Plan):
            # We need to use the same id in the model
            # as in the POST request.
            pointer_model = model.send(self)
            res_model = pointer_model
        else:
            res_model = model

        serialized_model = serialize(res_model).decode(self.encoding)

        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.HOST_MODEL,
            "encoding": self.encoding,
            "model_id": model_id,
            "allow_download": str(allow_download),
            "mpc": str(mpc),
            "allow_remote_inference": str(allow_remote_inference),
            "model": serialized_model,
        }

        url = self.address.replace("ws", "http") + "/data_centric/serve-model/"

        # Multipart encoding
        form = MultipartEncoder(message)
        upload_size = form.len

        # Callback that shows upload progress
        def progress_callback(monitor):
            upload_progress = "{} / {} ({:.2f} %)".format(
                monitor.bytes_read, upload_size, (monitor.bytes_read / upload_size) * 100
            )
            print(upload_progress, end="\r")
            if monitor.bytes_read == upload_size:
                print()

        monitor = MultipartEncoderMonitor(form, progress_callback)
        headers = {"Prefer": "respond-async", "Content-Type": monitor.content_type}

        session = requests.Session()
        response = session.post(url, headers=headers, data=monitor).content
        session.close()
        return self._return_bool_result(json.loads(response))

    def run_remote_inference(self, model_id, data):
        """Run a dataset inference using a remote model.
        Args:
            model_id (str) : Model ID.
            data (Tensor) : dataset to be inferred.
        Returns:
            inference (Tensor) : Inference result
        Raises:
            RuntimeError : If an unexpected behavior happen.
        """
        serialized_data = serialize(data).decode(self.encoding)
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.RUN_INFERENCE,
            "model_id": model_id,
            "data": serialized_data,
            "encoding": self.encoding,
        }
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response, RESPONSE_MSG.INFERENCE_RESULT)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model previously registered.
        Args:
            model_id (String) : ID of the model that will be deleted.
        Returns:
            result (bool) : If succeeded, return True.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.DELETE_MODEL, "model_id": model_id}
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response)

    def __str__(self) -> str:
        return f"<Federated Worker id:{self.id}>"

    @staticmethod
    def simplify(_worker: AbstractWorker, worker: "VirtualWorker") -> tuple:
        return BaseWorker.simplify(_worker, worker)

    @staticmethod
    def detail(worker: AbstractWorker, worker_tuple: tuple) -> Union["VirtualWorker", int, str]:
        detailed = BaseWorker.detail(worker, worker_tuple)

        if isinstance(detailed, int):
            result = VirtualWorker(id=detailed, hook=worker.hook)
        else:
            result = detailed

        return result
