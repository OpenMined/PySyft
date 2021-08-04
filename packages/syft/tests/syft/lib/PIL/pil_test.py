# third party
import pytest
import torch

# syft absolute
import syft as sy
from syft.grid.duet.ui import LOGO_URL

PIL = pytest.importorskip("PIL")
np = pytest.importorskip("numpy")


# MADHAVA: this needs fixing
@pytest.mark.skip(
    reason="This was broken when we switched from using a Dictionary obj store to a SQL one which means"
    + "that there's missing serialization functionality. Please address when you can."
)
@pytest.mark.vendor(lib="PIL")
def test_send_and_get(root_client: sy.VirtualMachineClient) -> None:
    im = PIL.Image.open(LOGO_URL)
    remote_im = im.send(root_client)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None


# MADHAVA: this needs fixing
@pytest.mark.skip(
    reason="This was broken when we switched from using a Dictionary obj store to a SQL one which means"
    + "that there's missing serialization functionality. Please address when you can."
)
@pytest.mark.vendor(lib="PIL")
def test_remote_create(root_client: sy.VirtualMachineClient) -> None:
    remote_torchvision = root_client.torchvision

    im = PIL.Image.open(LOGO_URL)
    im_array = np.array(im)
    im_tensor = torch.Tensor(im_array).permute(2, 0, 1)
    remote_tensor = im_tensor.send(root_client)
    remote_im = remote_torchvision.transforms.functional.to_pil_image(remote_tensor)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None
