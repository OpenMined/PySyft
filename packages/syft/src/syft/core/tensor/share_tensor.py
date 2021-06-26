# third party
import numpy as np
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

        self.ring_size = ring_size
        self.min_value = (-ring_size) // 2
        self.max_value = (ring_size - 1) // 2

        super().__init__(value)

    def generate_przs(self, shape):
        if self.child is None:
            self.child = np.zeros(shape)

        share_1 = self.generators_przs[0].integers(
            low=self.min_value, high=self.max_value
        )
        share_2 = self.generators_przs[1].integers(
            low=self.min_value, high=self.max_value
        )
        self.child = self.child + share_1 - share_2
