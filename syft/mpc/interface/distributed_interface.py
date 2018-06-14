from interface.base_interface import BaseInterface
import os
import torch
import torch.distributed as dist


class DistributedInterface(object):

    def __init__(self,party, master_addr='127.0.0.1', master_port='29500'):
        self.party=party
        if party:
            self.other = 0
        else:
            self.other = 1
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        # currently only supports sending between 2 parties over tcp
        dist.init_process_group('tcp', rank=party, world_size=2)

    def send(self, var):
        dist.send(tensor=var, dst=self.other)

    def recv(self, var):
        dist.recv(tensor=var, src=self.other)
        return var
    def get_party(self):
        return self.party
