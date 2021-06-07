# third party
import pytest
import torch

# syft absolute
import syft as sy

input_list = [torch.nn.Linear(100, 10)]


@pytest.mark.parametrize("target_layer", input_list)
def test_weights_and_bias(
    root_client: sy.VirtualMachineClient, target_layer: torch.nn.Module
) -> None:
    original_bias = target_layer.bias
    original_weight = target_layer.weight

    remote_target_mod = target_layer.send(root_client)

    remote_original_bias = remote_target_mod.bias
    remote_original_weight = remote_target_mod.weight

    assert torch.equal(original_bias, remote_original_bias.get())
    assert torch.equal(original_weight, remote_original_weight.get())

    new_bias = torch.nn.Parameter(torch.zeros_like(original_bias))
    new_weight = torch.nn.Parameter(torch.zeros_like(original_weight))

    remote_target_mod.weight = new_weight
    remote_target_mod.bias = new_bias

    assert torch.equal(remote_target_mod.weight.get(), new_weight)
    assert torch.equal(remote_target_mod.bias.get(), new_bias)
