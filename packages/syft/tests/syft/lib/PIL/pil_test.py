# third party
import pytest

# syft absolute
import syft as sy
from syft.grid.duet.ui import LOGO_URL


@pytest.mark.vendor(lib="PIL")
def test_send_and_get(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import PIL

    sy.load("PIL")

    im = PIL.Image.open(LOGO_URL)
    remote_im = im.send(root_client)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None


@pytest.mark.vendor(lib="PIL")
def test_remote_create(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import PIL
    import numpy as np
    import torch

    sy.load("PIL")

    remote_torchvision = root_client.torchvision

    im = PIL.Image.open(LOGO_URL)
    im_array = np.array(im)
    im_tensor = torch.Tensor(im_array).permute(2, 0, 1)
    remote_tensor = im_tensor.send(root_client)
    remote_im = remote_torchvision.transforms.functional.to_pil_image(remote_tensor)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None
