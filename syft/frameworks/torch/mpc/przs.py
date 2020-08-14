from collections import defaultdict
from syft.frameworks.torch.mpc.primitives import PrimitiveStorage
from syft.generic.utils import remote, allow_command

import torch
import syft

RING_SIZE = 2 ** 64
ERR_MSG = "You must call PRZS.setup because the seeds where not shared between workers"


class PRZS:
    def __init__(self):
        self.generators = {}

    @property
    def generators(self):
        return self.__generators

    @generators.setter
    def generators(self, generators):
        self.__generators = generators

    @staticmethod
    def setup(workers):
        seed_max = 2 ** 32
        paired_workers = list(zip(workers, workers[1:]))
        paired_workers.append((workers[-1], workers[0]))

        workers_ptr = defaultdict(dict)

        for cur_worker, next_worker in paired_workers:
            ptr = cur_worker.remote.torch.randint(-seed_max, seed_max - 1, size=(1,))
            ptr_next = ptr.copy().move(next_worker)

            workers_ptr[cur_worker]["cur_seed"] = ptr
            workers_ptr[next_worker]["prev_seed"] = ptr_next

        for worker, seeds in workers_ptr.items():
            cur_seed = seeds["cur_seed"]
            prev_seed = seeds["prev_seed"]
            func_remote = remote(_initialize_generators, location=worker)
            func_remote(cur_seed, prev_seed)

    def generate_alpha_3of3(self):
        """
            This function should be called only on a local worker
            Generate a random number (alpha) using the two generators
            * generator cur - represents a generator initialized with this worker (i) seed
            * generator prev - represents a generator initialized with
                        the previous worker (i-1) seed
        """
        assert self.generators, ERR_MSG
        cur_gen = self.generators["cur"]
        prev_gen = self.generators["prev"]

        alpha = self.__get_next_elem(cur_gen) - self.__get_next_elem(prev_gen)
        return alpha

    def generate_alpha_2of3(self):
        """
            This function should be called only on a local worker
            Generate 2 random numbers (alpha_i, alpha_i-1) using the two generators
            * generator cur - represents a generator initialized with this worker (i) seed
                        and it generates alpha_i
            * generator prev - represents a generator initialized with
                        the previous worker (i-1) seed and it generates alpha_i-1
        """

        assert self.generators, ERR_MSG

        cur_gen = self.generators["cur"]
        prev_gen = self.generators["prev"]

        alpha_cur, alpha_prev = (self.__get_next_elem(cur_gen), self.__get_next_elem(prev_gen))
        return alpha_cur, alpha_prev

    def __get_next_elem(self, generator, shape=(1,), ring_size=2 ** 32):
        tensor = torch.empty(shape, dtype=torch.long)
        worker = tensor.owner

        return tensor.random_(-ring_size // 2, (ring_size - 1) // 2, generator=generator)


def get_random(name_generator, shape, worker):
    func = None
    if worker == syft.local_worker:
        func = _get_random_tensor(name_generator, shape, worker.id)
    else:
        func = remote(_get_random_tensor, location=worker)

    return func(name_generator, shape, worker.id)


@allow_command
def _initialize_generators(cur_seed, prev_seed):
    worker = cur_seed.owner
    cur_generator = torch.Generator()
    prev_generator = torch.Generator()

    cur_generator.manual_seed(cur_seed.item())
    prev_generator.manual_seed(prev_seed.item())

    worker.crypto_store.przs.generators = {"cur": cur_generator, "prev": prev_generator}


@allow_command
def _get_random_tensor(name_generator, shape, worker_id):
    worker = syft.local_worker.get_worker(worker_id)

    assert worker.crypto_store.przs.generators, ERR_MSG

    generators = worker.crypto_store.przs.generators

    gen = generators[name_generator]
    rand_elem = torch.randint(
        -(RING_SIZE // 2), (RING_SIZE - 1) // 2, shape, dtype=torch.long, generator=gen
    )
    return rand_elem


PrimitiveStorage.register_component("przs", PRZS)
