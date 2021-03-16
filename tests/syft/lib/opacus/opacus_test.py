# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="opacus")
def test_remote_engine_simple() -> None:
    sy.load("opacus")

    data_owner = sy.VirtualMachine().get_root_client()
    remote_opacus = data_owner.opacus
    remote_torch = data_owner.torch

    model_ptr = remote_torch.nn.Linear(1, 1)
    batch_size = 16
    sample_size = 16
    noise_multiplier = 1.0
    max_grad_norm = 1.0

    privacy_engine_ptr = remote_opacus.privacy_engine.PrivacyEngine(
        model_ptr,
        batch_size=batch_size,
        sample_size=sample_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    assert privacy_engine_ptr.__name__ == "PrivacyEnginePointer"
