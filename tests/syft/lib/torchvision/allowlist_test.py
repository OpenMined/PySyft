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


def test_transforms_functional(alice: sy.VirtualMachine, tens: torch.Tensor) -> None:
    alice_client = alice.get_root_client()
    torchvision = alice_client.torchvision
    torchvision.transforms.functional.adjust_brightness(tens, 0.2)
    string_function = "torchvision.transforms.functional.adjust_brightness"
    parameters_test = "(tens, 0.2)"
    exec(string_function + parameters_test)
    # torchvision.transforms.functional.rotate(tens, 0.2)
    torchvision.transforms.functional.adjust_gamma(tens, 0.2)
    torchvision.transforms.functional.adjust_hue(tens, 0.2)
    torchvision.transforms.functional.adjust_saturation(tens, 0.2)
    torchvision.transforms.functional.erase(tens, 0, 10, 15, 20, 25)


def test_transforms(alice: sy.VirtualMachine) -> None:
    alice_client = alice.get_root_client()
    torchvision = alice_client.torchvision
    torchvision.transforms.RandomErasing(
        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
    )
    torchvision.transforms.Normalize(0, 1, inplace=False)
    torchvision.transforms.RandomOrder([1, 2])
    torchvision.transforms.RandomChoice([1, 2])
    torchvision.transforms.TenCrop(10, vertical_flip=False)
    torchvision.transforms.Resize(10, interpolation=2)
    torchvision.transforms.RandomVerticalFlip(p=0.5)
    torchvision.transforms.RandomRotation(
        10, resample=False, expand=False, center=None, fill=None
    )
    torchvision.transforms.RandomResizedCrop(
        10, scale=(0.08, 1.0), ratio=(0.75, 1.25), interpolation=2
    )
    torchvision.transforms.RandomPerspective(
        distortion_scale=0.5, p=0.5, interpolation=2, fill=0
    )
    torchvision.transforms.RandomHorizontalFlip(p=0.5)
    torchvision.transforms.RandomGrayscale(p=0.1)
    torchvision.transforms.RandomCrop(
        10, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"
    )
    torchvision.transforms.RandomApply([1, 2], p=0.5)
    torchvision.transforms.RandomAffine(2)
    # torchvision.transforms.Pad([1, 2], fill=0, padding_mode="constant")
    torchvision.transforms.Grayscale(num_output_channels=1)
    torchvision.transforms.CenterCrop(10)
    torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    torchvision.transforms.FiveCrop(10)


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
