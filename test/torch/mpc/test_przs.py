from syft.frameworks.torch.mpc.przs import PRZS, get_random


def test_generate_seeds(workers):
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]
    players = [bob, alice, james]
    seed_map = PRZS.generate_and_share_seeds(players)
    assert type(seed_map is dict)
    assert list(seed_map.keys()) == [bob, alice, james]
    assert seed_map[bob][0].location is seed_map[bob][1].location is bob
    assert seed_map[bob][0].get() == seed_map[alice][1].get()


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
