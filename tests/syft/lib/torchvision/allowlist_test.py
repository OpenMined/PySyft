# allowlist_temp

# stdlib
from os import path
import os.path
from typing import Dict
from typing import Union

# third party
from packaging import version

# third party libraries
import pytest
import torch
import torchvision as tv

# syft absolute
import syft as sy
from syft.lib.torchvision.allowlist import allowlist

# PySyft/tests/syft/lib/torchvision
# PySyft/src/syft/lib/torchvision
fileName = "imageTensor.pt"

TORCHVISION_VERSION = version.parse(tv.__version__)


@pytest.fixture(scope="function")
def tens() -> torch.Tensor:
    if path.isfile("imageTensor.pt"):
        return torch.load("imageTensor.pt")
    else:
        cwd = os.getcwd()
        path_file = cwd + "/tests/syft/lib/torchvision/" + fileName
        return torch.load(path_file)


@pytest.fixture(scope="function")
def alice() -> sy.VirtualMachine:
    return sy.VirtualMachine(name="alice")


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        if "min_version" not in support_dict.keys():
            return True
        return TORCHVISION_VERSION >= version.parse(support_dict["min_version"])


def test_allowlist(alice: sy.VirtualMachine) -> None:
    alice_client = alice.get_root_client()
    torchvision = alice_client.torchvision
    torch = alice_client.torch
    try:
        tx = torch.rand(4)
        tx = tx * 2
    except Exception as e:
        print(e)
    transforms = torchvision.transforms
    transforms.RandomAffine(2)
    for item in allowlist:

        if isinstance(allowlist[item], dict):
            # print(item)
            if version_supported(support_dict=allowlist[item]):
                if "test_parameters" in allowlist[item].keys():
                    print(item)
                    exec(item + allowlist[item]["test_parameters"])


# bob = sy.VirtualMachine(name="alice")
# test_allowlist(bob)
