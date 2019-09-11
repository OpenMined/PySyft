import binascii
import json
import os
import requests
from requests_toolbelt.multipart import encoder
import sys

import torch as th
import syft as sy
from syft.workers.base import BaseWorker

from grid import utils as gr_utils

MODEL_LIMIT_SIZE = (1024 ** 2) * 100  # 100MB


class GridClient(BaseWorker):
    """GridClient."""

    def __init__(self, addr: str, verbose: bool = True, hook=None, id="grid", **kwargs):
        hook = sy.hook if hook is None else hook
        super().__init__(hook=hook, id=id, verbose=verbose, **kwargs)
        print(
            "WARNING: Grid nodes publish datasets online and are for EXPERIMENTAL use only."
            "Deploy nodes at your own risk. Do not use OpenGrid with any data/models you wish to "
            "keep private.\n"
        )
        self.addr = addr
        self._verify_identity()
        # We use ISO encoding for some serialization/deserializations
        # due to non-ascii characters.
        self._encoding = "ISO-8859-1"

    def _verify_identity(self):
        r = requests.get(self.addr + "/identity/")
        if r.text != "OpenGrid":
            raise PermissionError("App is not an OpenGrid app.")

    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        raise NotImplementedError

    def _send_http_request(
        self, route, data, request, N: int = 10, unhexlify: bool = True
    ):
        url = os.path.join(self.addr, "{}".format(route))
        r = request(url, data=data) if data else request(url)
        response = r.text
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
        url = os.path.join(self.addr, "{}".format(route))

        # Send data
        session = requests.Session()
        form = encoder.MultipartEncoder(data)
        headers = {"Prefer": "respond-async", "Content-Type": form.content_type}
        resp = session.post(url, headers=headers, data=form)
        session.close()
        return resp.content

    def _send_post(self, route, data=None, N: int = 10, unhexlify: bool = True):
        return self._send_http_request(
            route, data, requests.post, N=N, unhexlify=unhexlify
        )

    def _send_get(self, route, data=None, N: int = 10, unhexlify: bool = True):
        return self._send_http_request(
            route, data, requests.get, N=N, unhexlify=unhexlify
        )

    def _recv_msg(self, message: bin, N: int = 10) -> bin:
        message = str(binascii.hexlify(message))
        return self._send_post("cmd", data={"message": message}, N=N)

    def destroy(self):
        grid_name = self.addr.split("//")[1].split(".")[0]
        gr_utils.execute_command(
            "heroku destroy " + grid_name + " --confirm " + grid_name
        )
        if self.verbose:
            print("Destroyed node: " + str(grid_name))

    @property
    def models(self, N: int = 1):
        return json.loads(self._send_get("models/", N=N))

    def delete_model(self, model_id):
        return json.loads(
            self._send_post(
                "delete_model/", data={"model_id": model_id}, unhexlify=False
            )
        )

    def serve_model(self, model, model_id=None, is_private_model=False):
        """Hosts the model and optionally serve it using a Rest API.

        Args:
            model: A jit model or Syft Plan.
            model_id: An integer or string representing the model id used to retrieve the model
                later on using the Rest API. If this is not provided and the model is a Plan
                we use model.id, if the model is a jit model we raise an exception.
            is_private_model: A boolean indicating if the model is private or not. If the model
                is private the user does not intend to serve it using a Rest API due to privacy reasons.

        Returns:
            None if is_private_model is True, otherwise it returns a json object representing a Rest API response.

        Raises:
            ValueError: if model_id is not provided and model is a jit model (does not have an id attribute).
        """

        # If model is private just send the nodel and return None
        if is_private_model:
            model.send(self)
            return None

        if model_id is None:
            if isinstance(model, sy.Plan):
                model_id = model.id
            else:
                raise ValueError("Model id argument is mandatory for jit models.")

        # If the model is a Plan we send the model
        # and host the plan version created after
        # the send operation
        if isinstance(model, sy.Plan):
            model.send(self)
            res_model = model.ptr_plans[self.id]
        else:
            res_model = model

        serialized_model = sy.serde.serialize(res_model).decode(self._encoding)

        if sys.getsizeof(serialized_model) >= MODEL_LIMIT_SIZE:
            return json.loads(
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
                    },
                )
            )
        else:
            return json.loads(
                self._send_post(
                    "serve-model/",
                    data={
                        "model": serialized_model,
                        "model_id": model_id,
                        "encoding": self._encoding,
                    },
                    unhexlify=False,
                )
            )

    def run_inference(self, model_id, data, N: int = 1):
        serialized_data = sy.serde.serialize(data)
        return json.loads(
            self._send_get(
                "models/{}".format(model_id),
                data={
                    "data": serialized_data.decode(self._encoding),
                    "encoding": self._encoding,
                },
                N=N,
            )
        )
