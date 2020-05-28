import syft as sy
import torch  # noqa: F401

# Import Hook
from syft.frameworks.torch.hook.hook import TorchHook

# Import grids
from syft.grid.private_grid import PrivateGridNetwork


def test_sandbox():
    sy.create_sandbox(globals(), download_data=False)

    assert alice == alice  # noqa: F821
    assert isinstance(alice, sy.VirtualWorker)  # noqa: F821
    assert andy == andy  # noqa: F821
    assert isinstance(andy, sy.VirtualWorker)  # noqa: F821
    assert bob == bob  # noqa: F821
    assert isinstance(bob, sy.VirtualWorker)  # noqa: F821
    assert jason == jason  # noqa: F821
    assert isinstance(jason, sy.VirtualWorker)  # noqa: F821
    assert jon == jon  # noqa: F821
    assert isinstance(jon, sy.VirtualWorker)  # noqa: F821
    assert theo == theo  # noqa: F821
    assert isinstance(theo, sy.VirtualWorker)  # noqa: F821

    assert hook == hook  # noqa: F821
    assert isinstance(hook, TorchHook)  # noqa: F821

    assert grid == grid  # noqa: F821
    assert isinstance(grid, PrivateGridNetwork)  # noqa: F821

    assert workers == [bob, theo, jason, alice, andy, jon]  # noqa: F821
