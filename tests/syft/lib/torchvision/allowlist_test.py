# stdlib
from os import path
import os.path
from typing import Dict
from typing import Union

# third party
from packaging import version
import pytest
import torch
import torchvision as tv

# syft absolute
import syft as sy
from syft.lib.torchvision.allowlist import allowlist

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


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        if "min_version" not in support_dict.keys():
            return True
        return TORCHVISION_VERSION >= version.parse(support_dict["min_version"])


def test_allowlist(root_client: sy.VirtualMachineClient, tens: torch.Tensor) -> None:
    torchvision = root_client.torchvision
    torch = root_client.torch
    try:
        tx = torch.rand(4)
        tx = tx * 2
    except Exception as e:
        print(e)
    transforms = torchvision.transforms
    transforms.RandomAffine(2)
    for item in allowlist:
        arr = item.split(".")
        # print(item)
        if (
            arr[1] == "datasets"
            and len(arr) <= 3
            and isinstance(allowlist[item], dict)
            and "test_parameters" in allowlist[item].keys()
            and version_supported(support_dict=allowlist[item])
        ):
            print(item)
            try:
                exec(item + allowlist[item]["test_parameters"])
            except RuntimeError as e:
                assert (
                    "not found" in str(e)
                    or "not present in the root directory" in str(e)
                    or "does not exist" in str(e)
                )
            except FileNotFoundError as e:
                assert "No such file or directory" in str(
                    e
                ) or "cannot find the path" in str(e)
            except ModuleNotFoundError as e:
                assert "No module named" in str(e)
            except KeyError:
                pass
        elif (
            isinstance(allowlist[item], dict)
            and version_supported(support_dict=allowlist[item])
            and "test_parameters" in allowlist[item].keys()
        ):
            exec(item + allowlist[item]["test_parameters"])
