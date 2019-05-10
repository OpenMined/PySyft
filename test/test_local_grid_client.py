import syft as sy
import torch as th

from grid.client import GridClient


def test_local_grid_client():
    hook = sy.TorchHook(th)
    gr_client = GridClient(addr="http://127.0.0.1:5000")
    assert True
