from syft.core.tensor.passthrough import PassthroughTensor

# TODO: Maybe use something from numpy?
import secrets

class ShareTensor(PassthroughTensor)
    def __init__(self, rank: int, value = None, seed = None, seed_generators = None):
        if seed_generators is None:
            self.seed_generators = [0, 1]
        else:
            self.seed_generators = seed_generators

        if seed is None:
           self.seed = 42
        else:
            self.seed = seed

        self.generators_przs = [random.seed(seed) for seed in self.seed_generators]
        self.generator_ids = = random.seed(self.seed)

        self.prev_generator = Generator(
        self.next_generator = G
        super().__init__(encoded_value)


    def generate_przs():
        ...

