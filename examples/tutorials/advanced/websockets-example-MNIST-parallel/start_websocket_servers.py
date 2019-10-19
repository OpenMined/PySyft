import subprocess

from torchvision import datasets
from torchvision import transforms

import signal
import sys


# Downloads MNIST dataset
mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

call_alice = [
    "python",
    "run_websocket_server.py",
    "--port",
    "8777",
    "--id",
    "alice",
    "--host",
    "0.0.0.0",
]

call_bob = [
    "python",
    "run_websocket_server.py",
    "--port",
    "8778",
    "--id",
    "bob",
    "--host",
    "0.0.0.0",
]

call_charlie = [
    "python",
    "run_websocket_server.py",
    "--port",
    "8779",
    "--id",
    "charlie",
    "--host",
    "0.0.0.0",
]

call_testing = [
    "python",
    "run_websocket_server.py",
    "--port",
    "8780",
    "--id",
    "testing",
    "--testing",
    "--host",
    "0.0.0.0",
]

print("Starting server for Alice")
process_alice = subprocess.Popen(call_alice)

print("Starting server for Bob")
process_bob = subprocess.Popen(call_bob)

print("Starting server for Charlie")
process_charlie = subprocess.Popen(call_charlie)

print("Starting server for Testing")
process_testing = subprocess.Popen(call_testing)


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    for p in [process_alice, process_bob, process_charlie, process_testing]:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

signal.pause()
