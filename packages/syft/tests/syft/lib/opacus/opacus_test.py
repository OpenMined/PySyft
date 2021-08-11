# third party
import pytest

# syft absolute
import syft as sy

opacus = pytest.importorskip("opacus")


# MADHAVA: this needs fixing
@pytest.mark.xfail(
    reason="This was broken when we switched from using a Dictionary obj store to a SQL one which means"
    + "that there's missing serialization functionality. Please address when you can."
)
@pytest.mark.vendor(lib="opacus")
def test_remote_engine_simple(root_client: sy.VirtualMachineClient) -> None:
    remote_opacus = root_client.opacus
    remote_torch = root_client.torch

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
