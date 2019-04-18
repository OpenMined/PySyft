import torch as th
import syft as sy
import binascii
import requests
from syft.workers import BaseWorker

class GridClient(BaseWorker):

    def __init__(self, addr):
        super().__init__(hook=sy.hook, id="grid")

        self.addr = addr

    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        raise NotImplementException

    def _recv_msg(self, message: bin) -> bin:
        message = str(binascii.hexlify(message))

        # Try the message 10 times before quitting
        for i in range(10):
            r = requests.post(self.addr + '/cmd/', data={'message': message})

            response = r.text
            try:
                response = binascii.unhexlify(response[2:-1])
            except:
                print(response)
                response = None
                continue

            return response

        
