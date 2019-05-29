import torch as th
import syft as sy
import binascii
import requests
import os
from syft.workers import BaseWorker

from grid import utils as gr_utils


class GridClient(BaseWorker):
    """GridClient."""

    def __init__(self, addr: str, verbose: bool = True):
        super().__init__(hook=sy.hook, id="grid", verbose=verbose)
        print(
            "WARNING: Grid nodes publish datasets online and are for EXPERIMENTAL use only."
            "Deploy nodes at your own risk. Do not use OpenGrid with any data/models you wish to "
            "keep private.\n"
        )
        self.addr = addr
        self._verify_identity()

    def _verify_identity(self):
        r = requests.get(self.addr + "/identity/")
        if r.text != "OpenGrid":
            raise PermissionError("App is not an OpenGrid app.")

    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        raise NotImplementedError

    def _recv_msg(self, message: bin, N: int = 10) -> bin:
        message = str(binascii.hexlify(message))

        # Try to request the message `N` times.
        for _ in range(N):
            r = requests.post(self.addr + "/cmd/", data={"message": message})

            response = r.text
            try:
                response = binascii.unhexlify(response[2:-1])
            except:
                if self.verbose:
                    print(response)
                response = None
                continue

            return response

    def destroy(self):
        grid_name = self.addr.split("//")[1].split(".")[0]
        gr_utils.exec_os_cmd("heroku destroy " + grid_name + " --confirm " + grid_name)
        if self.verbose:
            print("Destroyed node: " + str(grid_name))
