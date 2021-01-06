# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="opacus", python={"min_version": (3, 6, 9)})
def test_remote_engine_simple() -> None:
    sy.load_lib("opacus")

    data_owner = sy.VirtualMachine().get_root_client()
    remote_opacus = data_owner.opacus
    remote_torch = data_owner.torch

    model_ptr = remote_torch.nn.Linear(1, 1)
    batch_size = 16
    sample_size = 16
    alphas = [2, 3, 4]
    noise_multiplier = 1.0
    max_grad_norm = 1.0

    privacy_engine_ptr = remote_opacus.privacy_engine.PrivacyEngine(
        model_ptr, batch_size, sample_size, alphas, noise_multiplier, max_grad_norm
    )

    assert privacy_engine_ptr.__name__ == "PrivacyEnginePointer"
