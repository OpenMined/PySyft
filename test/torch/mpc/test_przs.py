from syft.frameworks.torch.mpc.przs import PRZS, get_random


def test_get_random_number(workers):
    alice, bob, james = (
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    workers = [alice, bob, james]

    PRZS.setup(workers)

    paired_workers = list(zip(workers, workers[1:]))
    paired_workers.append((workers[-1], workers[0]))

    shape = (3, 2)

    for worker_cur, worker_next in paired_workers:
        t1 = get_random("cur", shape, worker_cur)
        t2 = get_random("prev", shape, worker_next)

        assert (t1.get() == t2.get()).all()
