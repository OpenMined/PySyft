import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import syft as sy
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker
from multiprocessing import Process
import threading
import asyncio

import numpy as np
from collections import ChainMap as merge

hook = sy.TorchHook(torch)


def start_proc(participant, kwargs):
    server = participant(**kwargs)

    def target():
        server.start()

    p = Process(target=target)
    p.start()
    return p, server


@sy.func2plan
def plan_double_abs(x):
    x = x + x
    x = torch.abs(x)
    return x


kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}
_, server = start_proc(WebsocketServerWorker, kwargs)

time.sleep(1)

x_ptr = torch.tensor([-1, 7, 3]).tag("x").send(server)
print("server has x")
