# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="PIL")
def test_remote_engine_simple() -> None:
    # third party
    import PIL

    sy.load("PIL")

    data_owner = sy.VirtualMachine().get_root_client()

    im = PIL.Image.open("logo.png")
    im = im.resize((64, 32))
    remote_im = im.send(data_owner)
    received_im = remote_im.get()

    assert PIL.ImageChops.difference(im, received_im).getbbox() is None
