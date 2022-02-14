# third party
# third party
import numpy as np
import pytest

# syft absolute
# absolute
from syft.core.tensor.smpc.share_tensor import ShareTensor


@pytest.mark.smpc
def test_bit_extraction() -> None:
    share = ShareTensor(rank=0, parties_info=[], ring_size=2**32)
    data = np.array([[21, 32], [-54, 89]], dtype=np.int32)
    share.child = data
    exp_res1 = np.array([[False, False], [True, False]], dtype=np.bool_)
    res = share.bit_extraction(31).child

    assert (res == exp_res1).all()

    exp_res2 = np.array([[True, False], [False, False]], dtype=np.bool_)
    res = share.bit_extraction(2).child
    assert (res == exp_res2).all()


@pytest.mark.smpc
def test_bit_extraction_exception() -> None:
    share = ShareTensor(rank=0, parties_info=[], ring_size=2**32)
    data = np.array([[21, 32], [-54, 89]], dtype=np.int32)
    share.child = data

    with pytest.raises(Exception):
        share >> 33

    with pytest.raises(Exception):
        share >> -1
