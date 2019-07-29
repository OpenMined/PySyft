from syft.frameworks.torch.hook import hook_args
from syft.frameworks.torch import pointers
import torch
import numpy as np


def test_build_rule_syft_tensors_and_pointers():
    pointer = pointers.PointerTensor(
        id=1000, location="location", owner="owner", garbage_collect_data=False
    )
    result = hook_args.build_rule(([torch.tensor([1, 2]), pointer], 42))
    assert result == ([1, 1], 0)


def test_build_rule_numpy():
    arr = np.array([2.0, 3.0, 4.0])
    result = hook_args.build_rule([arr, arr + 2, [2, 4, "string"]])
    assert result == [1, 1, [0, 0, 0]]
