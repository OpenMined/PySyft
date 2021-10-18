# # stdlib
# from typing import Any
# from typing import Dict
# from typing import Iterable
# from typing import List
# from typing import Tuple

# # third party
# import pytest

# # syft absolute
# from syft.core.smpc.store import CryptoStore
# from syft.core.smpc.store import register_primitive_store_add
# from syft.core.smpc.store import register_primitive_store_get

# # Rasswanth : Fix tests after solving .get() issues


# @pytest.mark.skip
# @register_primitive_store_get("test_crypto_store")
# def provider_test_get(
#     store: Dict[str, List[Any]], nr_instances: int
# ) -> List[Tuple[int]]:

#     return [store["test_key_store"][i] for i in range(nr_instances)]


# @pytest.mark.skip
# @register_primitive_store_add("test_crypto_store")
# def provider_test_add(
#     store: Dict[str, List[Any]], primitives: Iterable[Any]
# ) -> List[Tuple[int]]:
#     store["test_key_store"] = primitives


# @pytest.mark.skip
# def test_add_store() -> None:
#     crypto_store = CryptoStore()

#     primitives = list(range(100))
#     crypto_store.populate_store("test_crypto_store", primitives)

#     crypto_store.store["test_key_store"] == primitives


# @pytest.mark.skip
# @pytest.mark.parametrize("nr_instances", [1, 5, 7, 100])
# def test_get_store(nr_instances: int) -> None:
#     crypto_store = CryptoStore()

#     primitives = list(range(100))
#     crypto_store.store["test_key_store"] = primitives

#     primitives_store = crypto_store.get_primitives_from_store(
#         "test_crypto_store", nr_instances
#     )

#     assert primitives[:nr_instances] == primitives_store
