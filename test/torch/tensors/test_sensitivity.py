import torch as th
import syft as sy


def test_init(workers):

    # Initialization Test A: making sure sensitivity, max_vals, min_vals, and entities
    # are calculated correctly

    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([1])
    assert x.max_vals == th.tensor([1])
    assert x.min_vals == th.tensor([0])
    assert (x.entities == th.tensor([[1, 0, 0, 0]])).all()

    # ensure it's calculated correctly even when sensitivity is greater than 1
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[2, 0, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([2])
    assert x.max_vals == th.tensor([2])
    assert x.min_vals == th.tensor([0])
    assert (x.entities == th.tensor([[1, 0, 0, 0]])).all()

    # test when multiple entities are contributing to a value
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([1])
    assert x.max_vals == th.tensor([2])
    assert x.min_vals == th.tensor([0])
    assert (x.entities == th.tensor([[1, 1, 0, 0]])).all()

    # test when min_ent_conts go negative
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, -1, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    assert x.sensitivity == th.tensor([2])
    assert x.max_vals == th.tensor([2])
    assert x.min_vals == th.tensor([-2])
    assert (x.entities == th.tensor([[1, 1, 0, 0]])).all()


def test_fail_init(workers):
    # Initialization Test B: test the initialization failure modes

    # test when min_ent_conts are greater than max_ent_conts
    try:
        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([[1, 1, 0, 0]]), max_ent_conts=th.tensor([[-1, -1, 0, 0]])
            )
        )
        assert False

    except AssertionError:
        assert True

    try:
        # test when min_ent_conts don't match max_ent_conts
        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([[-1, -1, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
            )
        )

        assert False

    except RuntimeError as e:
        print(str(e))
        assert "size of tensor" in str(e)

    # test when min_ent_conts and max_ent_conts are missing an outer dimension
    try:

        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([-1, -1, 0, 0]), max_ent_conts=th.tensor([1, 1, 0, 0])
            )
        )

        assert False
    except sy.frameworks.torch.tensors.decorators.sensitivity.MissingEntitiesDimensionException as e:
        assert True

    # test when a tensor's value is outside of the range specified by min_ent_conts and max_ent_conts
    try:

        # negative, non-positive, single entitiy, overlapping, symmetric add
        x = (
            th.tensor([1])
            .int()
            .track_sensitivity(
                min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
            )
        )
        assert False
    except sy.frameworks.torch.tensors.decorators.sensitivity.ValuesOutOfSpecifiedMinMaxRangeException as e:
        assert True


def test_add():
    # Test Add

    # positive, non-negative, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[0, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([0])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, non-positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[0, 0, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([2])
    assert y.max_vals == th.tensor([0])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, single entitiy, overlapping, symmetric add
    x = (
        th.tensor([-1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 0, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([2])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 0, 0, 0]])).all()

    # negative, positive, dual entitiy, overlapping, symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = x + x

    assert y.sensitivity == th.tensor([4])
    assert y.max_vals == th.tensor([4])
    assert y.min_vals == th.tensor([-2])
    assert (y.entities == th.tensor([[1, 1, 0, 0]])).all()

    # negative, positive, dual entitiy, non-overlapping, non-symmetric add
    x = (
        th.tensor([1])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[-1, 0, 0, 0]]), max_ent_conts=th.tensor([[1, 1, 0, 0]])
        )
    )

    y = (
        th.tensor([5])
        .int()
        .track_sensitivity(
            min_ent_conts=th.tensor([[4, 0, 0, 0]]), max_ent_conts=th.tensor([[5, 5, 0, 0]])
        )
    )

    z = x + y

    assert z.sensitivity == th.tensor([6])
    assert z.max_vals == th.tensor([12])
    assert z.min_vals == th.tensor([3])
    assert (z.entities == th.tensor([[1, 1, 0, 0]])).all()
