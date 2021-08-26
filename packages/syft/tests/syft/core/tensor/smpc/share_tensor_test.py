# third party
import pytest

# syft absolute
from syft.core.tensor.smpc.share_tensor import ShareTensor


# TODO: This needs more tests (Added issue)


def test_publish_raise_not_implemented() -> None:
    share = ShareTensor(rank=0, nr_parties=1)

    with pytest.raises(NotImplementedError) as exp:
        share.publish()
