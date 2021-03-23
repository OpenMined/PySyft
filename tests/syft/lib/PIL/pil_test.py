# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="PIL")
def test_send_and_get() -> None:
    # stdlib
    import io

    # third party
    import PIL
    import requests

    sy.load("PIL")

    data_owner = sy.VirtualMachine().get_root_client()

    im_url = "https://www.python.org/static/community_logos/python-logo.png"
    im = PIL.Image.open(io.BytesIO(requests.get(im_url).content))
    remote_im = im.send(data_owner)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None


@pytest.mark.vendor(lib="PIL")
def test_remote_create() -> None:
    # stdlib
    import io

    # third party
    import PIL
    import numpy as np
    import requests
    import torch

    sy.load("PIL")

    data_owner = sy.VirtualMachine().get_root_client()
    remote_torchvision = data_owner.torchvision

    im_url = "https://www.python.org/static/community_logos/python-logo.png"
    im = PIL.Image.open(io.BytesIO(requests.get(im_url).content))
    im_array = np.array(im)
    im_tensor = torch.Tensor(im_array).permute(2, 0, 1)
    remote_tensor = im_tensor.send(data_owner)
    remote_im = remote_torchvision.transforms.functional.to_pil_image(remote_tensor)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None
