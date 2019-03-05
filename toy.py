import time
import os
import asyncio
from multiprocessing import Process
import syft as sy
import torch
import numpy as np
from toy_server import FederatedLearningServer
from toy_client import FederatedLearningClient
from threading import Thread


def start_proc(participant, kwargs):
    """ helper function for spinning up a websocket participant """
    def target():
        server = participant(**kwargs)
        server.start()
    p = Process(target=target)
    p.start()
    return p

def main():
    hook = sy.TorchHook(torch)
    kwargs = { "id": "fed1", "connection_params": { 'host': 'localhost', 'port': 8765 }, "hook": hook }
    server = start_proc(FederatedLearningServer, kwargs)

    time.sleep(1)
    t = torch.ones(5)

    client = FederatedLearningClient(kwargs['connection_params'])

    resp = client.send_msg('foobar')
    print('res', resp)
    f = client.send_msg('baz')
    print('res', f)
    f = client.send_msg('333')
    print('res', f)

    server.kill()

if __name__ == "__main__":
    main()
