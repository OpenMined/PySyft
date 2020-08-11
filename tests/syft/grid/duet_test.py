import syft as sy
import torch as th


def test_duet():
    duet = sy.Duet(host="127.0.0.1", port=5000)
    obj_id = duet.id
    duet.stop()
    assert "SpecificLocation" in f"{obj_id}"


def test_duet_send():
    duet = sy.Duet(host="127.0.0.1", port=5001)

    x = th.tensor([1, 2, 3])
    xp = x.send(duet)

    assert xp.id_at_location == x.id

    duet.stop()
