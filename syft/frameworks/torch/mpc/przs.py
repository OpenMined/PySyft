from collections import defaultdict
from syft.frameworks.torch.mpc.primitives import PrimitiveStorage
from syft.generic.utils import remote, allow_command

import torch
import syft

RING_SIZE = 2 ** 32
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
            if cur_worker == syft.local_worker:
                ptr = cur_worker.torch.randint(-seed_max, seed_max - 1, size=(1,))
                ptr_next = ptr.send(next_worker)
            else:
                ptr = cur_worker.remote.torch.randint(-seed_max, seed_max - 1, size=(1,))
                ptr_next = ptr.copy().move(next_worker)

            workers_ptr[cur_worker]["cur_seed"] = ptr
            workers_ptr[next_worker]["prev_seed"] = ptr_next

        for worker, seeds in workers_ptr.items():
            cur_seed = seeds["cur_seed"]
            prev_seed = seeds["prev_seed"]
            if worker == syft.local_worker:
                func = _initialize_generators
            else:
                func = remote(_initialize_generators, location=worker)
            func(cur_seed, prev_seed)


def get_random(name_generator, shape, worker):
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
def _get_random_tensor(name_generator, shape, worker_id, ring_size=RING_SIZE):
    worker = syft.local_worker.get_worker(worker_id)
    assert worker.crypto_store.przs.generators, ERR_MSG

    generators = worker.crypto_store.przs.generators

    gen = generators[name_generator]
    rand_elem = torch.randint(0, ring_size, shape, dtype=torch.long, generator=gen)
    return rand_elem


def gen_alpha_3of3(worker, ring_size=RING_SIZE):
    if worker == syft.local_worker:
        func = _generate_alpha_3of3
    else:
        func = remote(_generate_alpha_3of3, location=worker)

    return func(worker.id, ring_size)


def gen_alpha_2of3(worker, ring_size=RING_SIZE):
    if worker == syft.local_worker:
        func = _generate_alpha_2of3
    else:
        func = remote(_generate_alpha_2of3, location=worker)

    return func(worker.id, ring_size)


@allow_command
def _generate_alpha_3of3(worker_id, ring_size=RING_SIZE):
    """
    Generate a random number (alpha) using the two generators
    * generator cur - represents a generator initialized with this worker (i) seed
    * generator prev - represents a generator initialized with
                the previous worker (i-1) seed
    """
    worker = syft.local_worker.get_worker(worker_id)
    assert worker.crypto_store.przs.generators, ERR_MSG

    generators = worker.crypto_store.przs.generators

    cur_gen = generators["cur"]
    prev_gen = generators["prev"]

    alpha = __get_next_elem(cur_gen, ring_size) - __get_next_elem(prev_gen, ring_size)
    return alpha


@allow_command
def _generate_alpha_2of3(worker_id, ring_size=RING_SIZE):
    """
    Generate 2 random numbers (alpha_i, alpha_i-1) using the two generators
    * generator cur - represents a generator initialized with this worker (i) seed
                and it generates alpha_i
    * generator prev - represents a generator initialized with
                the previous worker (i-1) seed and it generates alpha_i-1
    """
    worker = syft.local_worker.get_worker(worker_id)
    assert worker.crypto_store.przs.generators, ERR_MSG

    generators = worker.crypto_store.przs.generators

    cur_gen = generators["cur"]
    prev_gen = generators["prev"]

    alpha_cur, alpha_prev = (
        __get_next_elem(cur_gen, ring_size),
        __get_next_elem(prev_gen, ring_size),
    )
    return torch.tensor([alpha_cur.item(), alpha_prev.item()])


def __get_next_elem(generator, ring_size=RING_SIZE, shape=(1,)):
    tensor = torch.empty(shape, dtype=torch.long)
    return tensor.random_(0, ring_size, generator=generator)


PrimitiveStorage.register_component("przs", PRZS)
