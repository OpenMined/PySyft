from syft.frameworks.torch.mpc.przs import przs_setup, przs_get_random


def test_get_random_number(workers):
    alice, bob, james = (
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    workers = [alice, bob, james]

    przs_setup(workers)
    cycle_workers = zip(workers, [*workers[1:], workers[0]])
    shape = (3, 2)

    for worker_cur, worker_next in cycle_workers:
        t1 = przs_get_random("cur", shape, worker_cur)
        t2 = przs_get_random("prev", shape, worker_next)

        assert (t1.get() == t2.get()).all()
