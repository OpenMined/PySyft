from syft.frameworks.torch.mpc.primitives import PrimitiveStorage
from syft.generic.utils import remote, allow_command

import torch
import syft


class PRZS:
    def __init__(self):
        self.generators = {}

    ring_size = 2 ** 32

    @classmethod
    def setup(cls, players):
        seed_map = cls.generate_and_share_seeds(players)
        for worker, seeds in seed_map.items():
            if worker == syft.local_worker:
                initialize_generators = _initialize_generators
            else:
                initialize_generators = remote(_initialize_generators, location=worker)
            initialize_generators(*seeds)

    @classmethod
    def generate_and_share_seeds(cls, players):
        """
        Returns: dict {player i: seed i, seed i+1}
        """
        local_seeds = []
        remote_seeds = []
        number_of_players = len(players)
        for i in range(number_of_players):
            if players[i] == syft.local_worker:
                local_seed = players[i].torch.randint(high=cls.ring_size, size=[1])
                remote_seed = local_seed.send(players[(i + 1) % number_of_players])
            else:
                local_seed = players[i].remote.torch.randint(high=cls.ring_size, size=[1])
                remote_seed = local_seed.copy().move(players[(i + 1) % number_of_players])
            local_seeds.append(local_seed)
            remote_seeds.append(remote_seed)
        return {
            players[i]: (local_seeds[i], remote_seeds[(i - 1) % number_of_players])
            for i in range(number_of_players)
        }

    @property
    def generators(self):
        return self.__generators

    @generators.setter
    def generators(self, generators):
        self.__generators = generators


RING_SIZE = 2 ** 32
ERR_MSG = "You must call PRZS.setup because the seeds where not shared between workers"


@allow_command
def _initialize_generators(cur_seed, prev_seed):
    worker = cur_seed.owner
    cur_generator = torch.Generator()
    prev_generator = torch.Generator()

    cur_generator.manual_seed(cur_seed.item())
    prev_generator.manual_seed(prev_seed.item())

    worker.crypto_store.przs.generators = {"cur": cur_generator, "prev": prev_generator}


def get_random(name_generator, shape, worker):
    if worker == syft.local_worker:
        func = _get_random_tensor(name_generator, shape, worker.id)
    else:
        func = remote(_get_random_tensor, location=worker)

    return func(name_generator, shape, worker.id)


@allow_command
def _get_random_tensor(name_generator, shape, worker_id, ring_size=RING_SIZE):
    worker = syft.local_worker.get_worker(worker_id)
    if not worker.crypto_store.przs.generators:
        raise ValueError(ERR_MSG)

    generators = worker.crypto_store.przs.generators

    gen = generators[name_generator]
    rand_elem = torch.randint(high=ring_size, size=shape, generator=gen, dtype=torch.long)
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
    if not worker.crypto_store.przs.generators:
        raise ValueError(ERR_MSG)

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
    if not worker.crypto_store.przs.generators:
        raise ValueError(ERR_MSG)

    generators = worker.crypto_store.przs.generators

    cur_gen = generators["cur"]
    prev_gen = generators["prev"]

    alpha_cur, alpha_prev = (
        __get_next_elem(cur_gen, ring_size),
        __get_next_elem(prev_gen, ring_size),
    )
    return torch.tensor(alpha_cur.item()), torch.tensor(alpha_prev.item())


def __get_next_elem(generator, ring_size=RING_SIZE, shape=(1,)):
    tensor = torch.empty(shape, dtype=torch.long)
    return tensor.random_(0, ring_size, generator=generator)


PrimitiveStorage.register_component("przs", PRZS)
