from syft.frameworks.torch.tensors.interpreters import LargePrecisionTensor


def test_split_restore():
    bits = 32

    result_128 = LargePrecisionTensor._split_number(87721325272084551684339671875103718004, bits)
    result_64 = LargePrecisionTensor._split_number(4755382571665082714, bits)
    result_32 = LargePrecisionTensor._split_number(1107198784, bits)

    assert len(result_128) == 4
    assert len(result_64) == 2
    assert len(result_32) == 1

    assert (
        LargePrecisionTensor._restore_number(result_128, bits)
        == 87721325272084551684339671875103718004
    )
    assert LargePrecisionTensor._restore_number(result_64, bits) == 4755382571665082714
    assert LargePrecisionTensor._restore_number(result_32, bits) == 1107198784


def test_add():
    bits = 16
    expected = 9510765143330165428
    lpt1 = LargePrecisionTensor([4755382571665082714], to_bits=bits)
    lpt2 = LargePrecisionTensor([4755382571665082714], to_bits=bits)
    result = lpt1.add(lpt2)
    # The same number can be factorized in different ways. The sum of two matrices can be different to the
    # factorisation of the number the sum represents
    assert LargePrecisionTensor._restore_number(result.tolist(), bits) == expected
