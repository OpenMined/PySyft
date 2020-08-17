import pytest
from os import name as os_name

import torch
import syft
if os_name != "nt":
    import crypten


@pytest.fixture(scope="function")
@pytest.mark.skipif("os_name == 'nt'")
def model():
    l_in, l_h, l_out = 32, 16, 2
    model = crypten.nn.Sequential(
        crypten.nn.Linear(l_in, l_h), crypten.nn.ReLU(), crypten.nn.Linear(l_h, l_out)
    )
    return model


@pytest.mark.skipif("os_name == 'nt'")
def test_send_module(workers, model):
    alice = workers["alice"]
    model_cmp = model.copy()
    params = [p.data for p in model_cmp.parameters()]

    assert model.location is None
    assert model.owner == syft.local_worker
    model.send(alice)
    assert model.location == alice
    assert model.owner == syft.local_worker
    model.get()
    assert model.location is None
    assert model.owner == syft.local_worker

    for p, p_cmp in zip(model.parameters(), params):
        assert torch.all(p.data == p_cmp)


@pytest.mark.skipif("os_name == 'nt'")
def test_share_module(workers, model):
    alice = workers["alice"]
    bob = workers["bob"]
    cp = workers["charlie"]

    model.fix_prec()
    model.share(alice, bob, crypto_provider=cp)
    for p in model.parameters():
        assert isinstance(p.child, syft.FixedPrecisionTensor)
        assert isinstance(p.child.child, syft.AdditiveSharingTensor)

    model.get()
    model.float_prec()
    for p in model.parameters():
        assert isinstance(p, torch.Tensor)


@pytest.mark.skipif("os_name == 'nt'")
def test_move_module(workers, model):
    alice = workers["alice"]
    bob = workers["bob"]

    assert model.location is None
    model.send(alice)
    assert model.location == alice
    model.move(bob)
    assert model.location == bob
    model.get()
    assert model.location is None


@pytest.mark.skipif("os_name == 'nt'")
def test_copy(model):
    copy_model = model.copy()
    with torch.no_grad():
        for p in model.parameters():
            assert isinstance(p, torch.Tensor)
            p.set_(torch.zeros_like(p))

        for p in copy_model.parameters():
            assert not torch.all(p == 0)


@pytest.mark.skipif("os_name == 'nt'")
def test_encrypted_model(model):
    crypten.init()
    model.encrypt()
    with pytest.raises(RuntimeError):
        model._check_encrypted()
    crypten.uninit()
