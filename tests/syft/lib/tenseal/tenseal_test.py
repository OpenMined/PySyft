# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_loaded_before() -> None:
    # third party
    import tenseal as ts

    sy.load_lib("tenseal")

    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    context = ts.Context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )

    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(alice_client, searchable=True)
    enc_v1_ptr = enc_v1.send(alice_client, searchable=True)
    enc_v2_ptr = enc_v2.send(alice_client, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result_dec_ptr = result_enc_ptr.decrypt()
    result = result_dec_ptr.get()
    assert result == [4.0, 4.0, 4.0, 4.0, 4.0]  # ~ [4, 4, 4, 4, 4]

    result_enc_ptr2 = enc_v1_ptr.dot(enc_v2_ptr)
    result_dec_ptr2 = result_enc_ptr2.decrypt()
    result2 = result_dec_ptr2.get()
    assert [round(i) for i in result2] == [10]  # ~ [10]

    matrix = [
        [73, 0.5, 8],
        [81, -5, 66],
        [-100, -78, -2],
        [0, 9, 17],
        [69, 11, 10],
    ]

    result_enc_ptr3 = enc_v1_ptr.matmul(matrix)

    result_dec_ptr3 = result_enc_ptr3.decrypt()
    result3 = result_dec_ptr3.get()
    assert [round(i) for i in result3] == [157, -90, 153]  # ~ [157, -90, 153]


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_loaded_after() -> None:
    # third party
    import tenseal as ts

    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    sy.load_lib("tenseal")

    context = ts.Context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )

    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(alice_client, searchable=True)
    enc_v1_ptr = enc_v1.send(alice_client, searchable=True)
    enc_v2_ptr = enc_v2.send(alice_client, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result_dec_ptr = result_enc_ptr.decrypt()
    result = result_dec_ptr.get()
    assert result == [4.0, 4.0, 4.0, 4.0, 4.0]  # ~ [4, 4, 4, 4, 4]

    result_enc_ptr2 = enc_v1_ptr.dot(enc_v2_ptr)

    result_dec_ptr2 = result_enc_ptr2.decrypt()
    result2 = result_dec_ptr2.get()
    assert [round(i) for i in result2] == [10]  # ~ [10]

    matrix = [
        [73, 0.5, 8],
        [81, -5, 66],
        [-100, -78, -2],
        [0, 9, 17],
        [69, 11, 10],
    ]

    result_enc_ptr3 = enc_v1_ptr.matmul(matrix)

    result_dec_ptr3 = result_enc_ptr3.decrypt()
    result3 = result_dec_ptr3.get()
    assert [round(i) for i in result3] == [157, -90, 153]  # ~ [157, -90, 153]
