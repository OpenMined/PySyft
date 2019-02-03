"""Some syft imports..."""
from . import frameworks
from . import workers
from . import serde
from . import codes

# CONVENIENCE HOOKS
# The purpose of the following import section is to increase the convenience of using
# PySyft by making it possible to import the most commonly used objects from syft
# directly (i.e., syft.TorchHook or syft.VirtualWorker or syft.LoggingTensor)

# Import Hook
from syft.frameworks.torch import TorchHook

# Import Worker Types
from syft.workers import VirtualWorker

# Import Tensor Types
from syft.frameworks.torch.tensors import LoggingTensor
from syft.frameworks.torch.tensors import PointerTensor

# import modules
from syft.frameworks.torch import optim

__all__ = [
    "frameworks",
    "workers",
    "serde",
    "TorchHook",
    "VirtualWorker",
    "codes",
    "LoggingTensor",
    "PointerTensor",
    "optim",
]

local_worker = None
torch = None


def create_sandbox(gbs):
    """There's some boilerplate stuff that most people who are
    just playing around would like to have. This will create
    that for you"""

    try:
        torch = gbs["torch"]
    except:
        torch = gbs["th"]

    global hook
    global bob
    global theo
    global alice
    global andy
    global jason
    global jon

    print("Setting up Sandbox...")

    print("\t- Hooking PyTorch")
    hook = TorchHook(torch)

    print("\t- Creating Virtual Workers:")
    print("\t\t- bob")
    bob = VirtualWorker(hook, id="bob")
    print("\t\t- theo")
    theo = VirtualWorker(hook, id="theo")
    print("\t\t- jason")
    jason = VirtualWorker(hook, id="jason")
    print("\t\t- alice")
    alice = VirtualWorker(hook, id="alice")
    print("\t\t- andy")
    andy = VirtualWorker(hook, id="andy")
    print("\t\t- jon")
    jon = VirtualWorker(hook, id="jon")

    print("Storing hook and workers as global variables...")
    gbs["hook"] = hook
    gbs["bob"] = bob
    gbs["theo"] = theo
    gbs["jason"] = jason
    gbs["alice"] = alice
    gbs["andy"] = andy
    gbs["jon"] = jon
    print("Done!")
