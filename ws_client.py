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
import binascii
from syft.serde import serialize, deserialize


@sy.func2plan
def plan_double_abs(x):
    x = x + x
    x = torch.abs(x)
    return x


kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}

socket_pipe = WebsocketClientWorker(**kwargs)

serialized_plan = str(binascii.hexlify(serialize(plan_double_abs)))
plan_ptr = plan_double_abs.send(socket_pipe)

x_ptr = torch.tensor([-1, 7, 3]).send(socket_pipe)
while True:
    time.sleep(1)
    p = plan_ptr(x_ptr).get()
    print(p)
    print(socket_pipe.search("x"))  # Empty
