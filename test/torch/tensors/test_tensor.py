import torch
import syft


def test_init():
    hook = syft.TorchHook(torch, verbose=True)
    tensor_extension = torch.Tensor()
    assert tensor_extension.id is not None
    assert tensor_extension.owner is not None


def test_decrypt_mpc(workers):
    alice = workers.get("alice")
    bob = workers.get("bob")
    cp = workers.get("charlie")
    t = torch.tensor(73)
    # without grad
    t_encrypted = t.encrypt(protocol="mpc", workers=[alice, bob], crypto_provider=cp)
    assert t_encrypted.decrypt() == t
    # with grad
    t_encrypted = t.encrypt(
        protocol="mpc", workers=[alice, bob], crypto_provider=cp, requires_grad=True
    )
    assert t_encrypted.decrypt() == t
