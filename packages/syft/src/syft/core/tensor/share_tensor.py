# stdlib
from functools import lru_cache

# third party
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor


class ShareTensor(PassthroughTensor):
    def __init__(
        self, rank, ring_size=2 ** 64, value=None, seed=None, seed_generators=None
    ):
        if seed_generators is None:
            self.seed_generators = [0, 1]
        else:
            self.seed_generators = seed_generators

        if seed is None:
            self.seed = 42
        else:
            self.seed = seed

        # TODO: This is not secure
        self.generators_przs = [
            np.random.default_rng(seed) for seed in self.seed_generators
        ]
        self.generator_ids = np.random.default_rng(self.seed)
        self.rank = rank
        self.ring_size = ring_size
        self.min_value, self.max_value = ShareTensor.compute_min_max_from_ring(
            self.ring_size
        )
        super().__init__(value)

    @staticmethod
    @lru_cache(32)
    def compute_min_max_from_ring(ring_size=2 ** 64):
        min_value = (-ring_size) // 2
        max_value = (ring_size - 1) // 2
        return min_value, max_value

    @staticmethod
    def generate_shares(secret, nr_shares, ring_size=2 ** 64):
        # syft relative
        from .fixed_precision_tensor import FixedPrecisionTensor

        if not isinstance(secret, np.ndarray):
            secret = np.array([secret])

        if not isinstance(secret, FixedPrecisionTensor):
            secret = FixedPrecisionTensor(value=secret)

        shape = secret.shape
        min_value, max_value = ShareTensor.compute_min_max_from_ring(ring_size)

        generator_shares = np.random.default_rng()

        random_shares = []
        for i in range(nr_shares):
            random_value = generator_shares.integers(
                low=min_value, high=max_value, size=shape
            )
            fpt_value = FixedPrecisionTensor(value=random_value)
            random_shares.append(fpt_value)

        shares_fpt = []
        for i in range(nr_shares):
            if i == 0:
                share = value = random_shares[i]
            elif i < nr_shares - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]

            shares_fpt.append(share)

        # Add the ShareTensor class between them
        shares = []
        for rank, share_fpt in enumerate(shares_fpt):
            share_fpt.child = ShareTensor(rank=rank, value=share_fpt.child)
            shares.append(share_fpt)

        return shares
