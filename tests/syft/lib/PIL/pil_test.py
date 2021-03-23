# stdlib
import io

# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="PIL")
def test_remote_engine_simple() -> None:
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
