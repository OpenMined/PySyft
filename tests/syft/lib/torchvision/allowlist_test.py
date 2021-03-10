# stdlib
from os import path
import os.path

# third party
# third party libraries
import pytest
import torch

# syft absolute
import syft as sy

fileName = "imageTensor.pt"


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


def test_transforms_functional(alice: sy.VirtualMachine, tens: torch.Tensor) -> None:
    alice_client = alice.get_root_client()
    torchvision = alice_client.torchvision
    torchvision.transforms.functional.adjust_brightness(tens, 0.2)
    torchvision.transforms.functional.adjust_contrast(tens, 0.2)
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
    torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0))
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
    torchvision.transforms.Pad([1, 2], fill=0, padding_mode="constant")
    torchvision.transforms.Grayscale(num_output_channels=1)
    torchvision.transforms.CenterCrop(10)
    torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    torchvision.transforms.FiveCrop(10)
