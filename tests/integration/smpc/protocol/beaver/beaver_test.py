# # third party
# import pytest

# # syft absolute
# from syft.core.smpc.store import CryptoPrimitiveProvider

# # Rasswanth : Fix tests after solving .get() issues


# @pytest.mark.skip
# def test_beaver_mul(get_clients) -> None:
#     parties = get_clients(3)
#     a_shape = (2, 2)
#     b_shape = (2, 2)
#     g_kwargs = {"a_shape": a_shape, "b_shape": b_shape}

#     # For other ops (ex: conv2d) g_kwargs,p_kwargs are different.
#     p_kwargs = g_kwargs

#     primitives = CryptoPrimitiveProvider.generate_primitives(
#         op_str="beaver_mul", parties=parties, g_kwargs=g_kwargs, p_kwargs=p_kwargs
#     )
#     cache_store = CryptoPrimitiveProvider.cache_store

#     for party, primitive in zip(parties, primitives):
#         crypto_store = cache_store[party]
#         triple = crypto_store.get_primitives_from_store(
#             "beaver_mul", a_shape, b_shape
#         ).get()
#         primitive = primitive[0]
#         for x, y in zip(triple, primitive):
#             assert x == y
