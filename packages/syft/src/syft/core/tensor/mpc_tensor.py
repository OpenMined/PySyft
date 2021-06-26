# stdlib
from functools import lru_cache

# third party
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor


class MPCTensor(PassthroughTensor):
    def __init__(self, shares):
        super().__init__(shares)

    def reconstruct(self):
        local_shares = [share.get() for share in self.child]

        result_fp = sum(local_shares)
        result = result_fp.decode()
        return result
