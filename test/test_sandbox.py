import pytest
import torch
import syft as sy


def test_sandbox():
    sy.create_sandbox(globals(), download_data=False)

    # check to make sure global variable gets set for alice
    assert alice == alice  # noqa: F821
    assert isinstance(alice, sy.VirtualWorker)  # noqa: F821
