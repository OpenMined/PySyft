import torch as th
import syft as sy
import binascii
import requests
import os
from syft.workers import BaseWorker

from grid import utils as gr_utils


class GridClient(BaseWorker):
    """GridClient."""

    def __init__(self, addr, verbose: bool = True):
        super().__init__(hook=sy.hook, id="grid", verbose=verbose)
        print("WARNING: Grid nodes publish datasets online and are for EXPERIMENTAL use only."
              "Deploy nodes at your own risk. Do not use OpenGrid with any data/models you wish to "
              "keep private.\n")
        self.addr = addr

    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        raise NotImplementedError

    def _recv_msg(self, message: bin) -> bin:
        message = str(binascii.hexlify(message))

        # Try the message 10 times before quitting
        for _ in range(10):
            r = requests.post(self.addr + '/cmd/', data={'message': message})

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
        res = gr_utils.exec_os_cmd("heroku destroy " + grid_name + " --confirm " + grid_name)
        if self.verbose:
            print("Destroyed node: " + str(grid_name))


