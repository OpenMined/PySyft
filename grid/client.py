import binascii
import json
import os
import requests


import torch as th
import syft as sy
from syft.workers import BaseWorker

from grid import utils as gr_utils


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
        models = json.loads(self._send_get("models/", N=N))["models"]
        return models

    def serve_model(self, model, model_id):
        # If the model is a Plan we send the model
        # and host the plan version created after
        # the send operation
        if isinstance(model, sy.Plan):
            _ = model.send(self)
            res_model = model.ptr_plans[self.id]
        else:
            res_model = model

        serialized_model = sy.serde.serialize(res_model).decode(self._encoding)
        return self._send_post(
            "serve-model/",
            data={
                "model": serialized_model,
                "model_id": model_id,
                "encoding": self._encoding,
            },
            unhexlify=False,
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
