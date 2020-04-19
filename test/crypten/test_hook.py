import pytest
import crypten
import torch
import syft


@pytest.fixture(scope="function")
def model():
    l_in, l_h, l_out = 32, 16, 2
    model = crypten.nn.Sequential(
        [crypten.nn.Linear(l_in, l_h), crypten.nn.ReLU(), crypten.nn.Linear(l_h, l_out)]
    )
    return model


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
