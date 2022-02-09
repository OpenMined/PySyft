# third party
import numpy as np
import pytest

# syft absolute
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
from syft.core.tensor.tensor import Tensor


@pytest.mark.smpc
def test_repeat(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 2))

    res = mpc_tensor.repeat(3)
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.repeat(3)

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_copy(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 2))

    res = mpc_tensor.copy()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = mpc_tensor.reconstruct()

    # we cannot check id for copy as the values are in different locations
    assert (res == exp_res).all()


@pytest.mark.smpc
def test_diagonal(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[0, 1], [2, 3]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 2))

    res = mpc_tensor.diagonal()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.diagonal()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_flatten(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(
        np.array([[89, 12, 54], [412, 89, 42], [87, 32, 58]], dtype=np.int32)
    )

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 3))

    res = mpc_tensor.flatten()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.flatten()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_transpose(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(
        np.array([[89, 12, 54], [412, 89, 42], [87, 32, 58]], dtype=np.int32)
    )

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 3))

    res = mpc_tensor.transpose()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.transpose()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_resize(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[89, 12], [412, 89], [87, 32]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 2))

    res = mpc_tensor.resize((2, 3))
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.resize((2, 3))

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_ravel(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[8, 1, 5], [4, 8, 4], [7, 2, 27]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 3))

    res = mpc_tensor.ravel()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.ravel()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_compress(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 2))

    res = mpc_tensor.compress([0, 1], axis=0)
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.compress([0, 1], axis=0)

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_reshape(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 2))

    res = mpc_tensor.reshape((2, 3))
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.reshape((2, 3))

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_squeeze(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[7], [6], [72]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 1))

    res = mpc_tensor.squeeze()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.squeeze()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_swapaxes(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[613, 645, 738], [531, 412, 658]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 3))

    res = mpc_tensor.swapaxes(0, 1)
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.swapaxes(0, 1)

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_pos(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[5, 2], [3, 7]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 2))

    res = mpc_tensor.__pos__()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.__pos__()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_put(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[5, 2], [3, 7]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 2))

    res = mpc_tensor.put([0, 1], 7)
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    value.put([0, 1], 7)
    exp_res = value  # inplace ops

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_neg(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[-5, 2], [-3, 7]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 2))

    res = -mpc_tensor
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = -value

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_take(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([-5, 2, -3, 7, 132, 54, 27], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(7,))

    res = mpc_tensor.take([5, 1, 6])
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.take([5, 1, 6])

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_abs(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[-32, -54, 98], [12, -108, 27]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 3))

    res = mpc_tensor.__abs__()
    res.block_with_timeout(secs=120)
    res = res.reconstruct()

    exp_res = value.__abs__()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_sign(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[-32, -54, 98], [12, -108, 27]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 3))

    res = mpc_tensor.sign()
    res.block_with_timeout(secs=120)
    res = res.reconstruct()

    exp_res = np.array([[-1, -1, 1], [1, -1, 1]], dtype=np.int32)

    assert (res == exp_res).all()


@pytest.mark.parametrize("power", [4, 7])
@pytest.mark.smpc
def test_pow(get_clients, power) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([1, -2, 3], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3,))

    res = mpc_tensor**power
    res.block_with_timeout(secs=40)
    res = res.reconstruct()

    exp_res = value**power

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_cumsum(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([-5, 2, -3, 7, 132, 54, 27], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(7,))

    res = mpc_tensor.cumsum()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.cumsum()

    assert (res == exp_res.child).all()


@pytest.mark.smpc
def test_trace(get_clients) -> None:
    clients = get_clients(2)
    value = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32))

    remote_value = value.send(clients[0])

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(3, 3))

    res = mpc_tensor.trace()
    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    exp_res = value.trace()

    assert (res == exp_res.child).all()
