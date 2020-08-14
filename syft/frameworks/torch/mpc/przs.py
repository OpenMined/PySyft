from collections import defaultdict

import torch

import syft
from syft.generic.utils import allow_command, remote

RING_SIZE = 2 ** 64


def przs_setup(workers):
    seed_max = 2 ** 32
    shifted_workers = [workers.pop()]+[workers]
    cycle_workers = zip(workers, shifted_workers)

    workers_ptr = defaultdict(dict)

    for cur_worker, next_worker in cycle_workers:
        ptr = cur_worker.remote.torch.randint(-seed_max, seed_max - 1, size=(1,))
        ptr_next = ptr.copy().move(next_worker)

        workers_ptr[cur_worker]["cur_seed"] = ptr
        workers_ptr[next_worker]["prev_seed"] = ptr_next

    for worker, seeds in workers_ptr.items():
        cur_seed = seeds["cur_seed"]
        prev_seed = seeds["prev_seed"]
        remote(_initialize_generators, location=worker)(cur_seed, prev_seed)


@allow_command
def _initialize_generators(cur_seed, prev_seed):
    worker = cur_seed.owner
    cur_generator = torch.Generator()
    prev_generator = torch.Generator()

    cur_generator.manual_seed(cur_seed.item())
    prev_generator.manual_seed(prev_seed.item())

    generators = {"przs": {"cur": cur_generator, "prev": prev_generator}}
    setattr(worker.crypto_store, "generators", generators)


@allow_command
def przs_get_random_number(name_generator, shape, id_worker):
    worker = syft.local_worker.get_worker(id_worker)
    generators = getattr(worker.crypto_store, "generators")
    gen = generators["przs"][name_generator]
    rand_elem = torch.randint(
        -(RING_SIZE // 2), (RING_SIZE - 1) // 2, shape, dtype=torch.long, generator=gen
    )
    return rand_elem


def przs_get_random(name_generator, shape, worker):
    return remote(przs_get_random_number, location=worker)(name_generator, shape, worker.id)
