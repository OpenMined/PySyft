import logging
import syft as sy
from syft.workers import WebsocketServerWorker
import torch
import argparse
from torchvision import datasets
from torchvision import transforms
import numpy as np
from syft.frameworks.torch.federated import utils

import subprocess
import sys
from pathlib import Path

python = Path(sys.executable).name

FILE_PATH = Path(__file__).resolve().parent.joinpath("run_websocket_server.py")

call_alice = [python, FILE_PATH, "--port", "8777", "--id", "alice", "--pytest_testing"]

print("Starting server for Alice")
subprocess.Popen(call_alice)
