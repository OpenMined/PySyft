# # stdlib
# from typing import Any
# from typing import Dict
# from typing import Iterable
# from typing import List
# from typing import Tuple

# # third party
# import numpy as np
# import pytest

# # syft absolute
# from syft import Tensor

# # absolute
# from syft.core.smpc.store import CryptoPrimitiveProvider
# from syft.core.smpc.store import register_primitive_generator
# from syft.core.smpc.store import register_primitive_store_add
# from syft.core.smpc.store import register_primitive_store_get
# from syft.core.tensor import ShareTensor

# PRIMITIVE_NR_ELEMS = 4

# # Rasswanth : Fix tests after solving .get() issues


# @pytest.mark.skip
# @register_primitive_generator("test")
# def provider_test(nr_parties: int, nr_instances: int) -> List[Tuple[int]]:
#     """This function will generate the values:

#     [((0, 0, 0, 0), (0, 0, 0, 0), ...), ((1, 1, 1, 1), (1, 1, 1, 1)),
#     ...]
#     """
#     primitives = [
#         tuple(
#             tuple(
#                 ShareTensor(
#                     rank=i,
#                     nr_parties=nr_parties,
#                     value=Tensor(np.array([[4, 5], [7, 27]], dtype=np.int32)),
#                 )
#                 for _ in range(PRIMITIVE_NR_ELEMS)
#             )
#             for _ in range(nr_instances)
#         )
#         for i in range(nr_parties)
#     ]
#     return primitives


# @pytest.mark.skip
# @register_primitive_store_get("test")
# def provider_test_get(
#     store: Dict[str, List[Any]], nr_instances: int
# ) -> List[Tuple[int]]:

#     return [store["test_key"][i] for i in range(nr_instances)]


# @pytest.mark.skip
# @register_primitive_store_add("test")
# def provider_test_add(
#     store: Dict[str, List[Any]], primitives: Iterable[Any]
# ) -> List[Tuple[int]]:
#     store["test_key"] = primitives


# @pytest.mark.skip
# def test_exception_init() -> None:
#     with pytest.raises(ValueError):
#         CryptoPrimitiveProvider()


# @pytest.mark.skip
# def test_generate_primitive_exception() -> None:
#     with pytest.raises(ValueError):
#         CryptoPrimitiveProvider.generate_primitives(op_str="SMPC", parties=[])


# @pytest.mark.skip
# def test_transfer_primitives_type_exception() -> None:
#     with pytest.raises(ValueError):
#         """Primitives should be a list."""
#         CryptoPrimitiveProvider._transfer_primitives_to_parties(
#             op_str="test", primitives=50, parties=[], p_kwargs={}
#         )


# @pytest.mark.skip
# def test_transfer_primitives_mismatch_len_exception() -> None:
#     with pytest.raises(ValueError):
#         """Primitives and Parties should have the same len."""
#         CryptoPrimitiveProvider._transfer_primitives_to_parties(
#             op_str="test", primitives=[1], parties=[], p_kwargs={}
#         )


# @pytest.mark.skip
# def test_register_primitive() -> None:

#     val = CryptoPrimitiveProvider.get_state()
#     expected_providers = "test"

#     assert expected_providers in val, "Test Provider not registered"


# @pytest.mark.skip
# @pytest.mark.parametrize("nr_instances", [1, 5, 100])
# @pytest.mark.parametrize("nr_parties", [2, 3, 4])
# def test_generate_primitive(get_clients, nr_parties: int, nr_instances: int) -> None:
#     parties = get_clients(nr_parties)
#     g_kwargs = {"nr_instances": nr_instances}
#     res = CryptoPrimitiveProvider.generate_primitives(
#         "test",
#         parties=parties,
#         g_kwargs=g_kwargs,
#         p_kwargs=None,
#     )

#     assert isinstance(res, list)
#     assert len(res) == nr_parties

#     for i, primitives in enumerate(res):
#         for primitive in primitives:
#             assert primitive == tuple(
#                 ShareTensor(
#                     nr_parties=nr_parties,
#                     value=Tensor(np.array([[4, 5], [7, 27]], dtype=np.int32)),
#                     rank=i,
#                 )
#                 for _ in range(PRIMITIVE_NR_ELEMS)
#             )


# @pytest.mark.skip
# @pytest.mark.parametrize(
#     ("nr_instances", "nr_instances_retrieve"),
#     [(1, 1), (5, 4), (5, 5), (100, 25), (100, 100)],
# )
# @pytest.mark.parametrize("nr_parties", [2, 3, 4])
# def test_generate_and_transfer_primitive(
#     get_clients,
#     nr_parties: int,
#     nr_instances: int,
#     nr_instances_retrieve: int,
# ) -> None:
#     parties = get_clients(nr_parties)
#     g_kwargs = {"nr_instances": nr_instances}
#     CryptoPrimitiveProvider.generate_primitives(
#         "test",
#         parties=parties,
#         g_kwargs=g_kwargs,
#         p_kwargs={},
#     )

#     for i, party in enumerate(parties):
#         remote_crypto_store = CryptoPrimitiveProvider.cache_store[party]
#         primitives = remote_crypto_store.get_primitives_from_store(
#             op_str="test", nr_instances=nr_instances_retrieve
#         ).get()
#         assert primitives == [
#             tuple(
#                 ShareTensor(
#                     nr_parties=nr_parties,
#                     value=Tensor(np.array([[4, 5], [7, 27]], dtype=np.int32)),
#                     rank=i,
#                 )
#                 for _ in range(PRIMITIVE_NR_ELEMS)
#             )
#             for _ in range(nr_instances_retrieve)
#         ]
