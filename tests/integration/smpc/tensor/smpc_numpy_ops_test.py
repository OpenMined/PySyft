# # third party
# import numpy as np
# import pytest

# # syft absolute
# from syft.core.tensor.smpc.mpc_tensor import MPCTensor


# @pytest.mark.skip
# def test_repeat(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[1, 2], [3, 4]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(2, 2), seed_shares=42
#     )

#     res = mpc_tensor.repeat(3).reconstruct()
#     exp_res = value.repeat(3)

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_copy(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[1, 2], [3, 4]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(2, 2), seed_shares=42
#     )

#     res = mpc_tensor.copy().reconstruct()
#     exp_res = mpc_tensor.reconstruct()

#     # we cannot check id for copy as the values are in different locations
#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_diagonal(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[0, 1], [2, 3]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(2, 2), seed_shares=42
#     )

#     res = mpc_tensor.diagonal().reconstruct()
#     exp_res = value.diagonal()

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_flatten(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[89, 12, 54], [412, 89, 42], [87, 32, 58]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 3), seed_shares=42
#     )

#     res = mpc_tensor.flatten().reconstruct()
#     exp_res = value.flatten()

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_transpose(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[89, 12, 54], [412, 89, 42], [87, 32, 58]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 3), seed_shares=42
#     )

#     res = mpc_tensor.transpose().reconstruct()
#     exp_res = value.transpose()

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_resize(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[89, 12], [412, 89], [87, 32]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 2), seed_shares=42
#     )

#     res = mpc_tensor.resize(2, 3).reconstruct()
#     value.resize(2, 3)
#     exp_res = value  # inplace op

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_ravel(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[8, 1, 5], [4, 8, 4], [7, 2, 27]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 3), seed_shares=42
#     )

#     res = mpc_tensor.ravel().reconstruct()
#     exp_res = value.ravel()

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_compress(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 2), seed_shares=42
#     )

#     res = mpc_tensor.compress([0, 1], axis=0).reconstruct()
#     exp_res = value.compress([0, 1], axis=0)

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_reshape(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 2), seed_shares=42
#     )

#     res = mpc_tensor.reshape((2, 3)).reconstruct()
#     exp_res = value.reshape((2, 3))

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_squeeze(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[7], [6], [72]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(3, 1), seed_shares=42
#     )

#     res = mpc_tensor.squeeze().reconstruct()
#     exp_res = value.squeeze()

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_swapaxes(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[613, 645, 738], [531, 412, 658]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(2, 3), seed_shares=42
#     )

#     res = mpc_tensor.swapaxes(0, 1).reconstruct()
#     exp_res = value.swapaxes(0, 1)

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_pos(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[5, 2], [3, 7]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(2, 2), seed_shares=42
#     )

#     res = mpc_tensor.__pos__().reconstruct()
#     exp_res = value.__pos__()

#     assert (res == exp_res).all()


# @pytest.mark.skip
# def test_put(get_clients) -> None:
#     clients = get_clients(2)
#     value = np.array([[5, 2], [3, 7]], dtype=np.int32)

#     remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

#     mpc_tensor = MPCTensor(
#         parties=clients, secret=remote_value, shape=(2, 2), seed_shares=42
#     )

#     res = mpc_tensor.put([0, 1], 7).reconstruct()
#     value.put([0, 1], 7)
#     exp_res = value  # inplace ops

#     assert (res == exp_res).all()
