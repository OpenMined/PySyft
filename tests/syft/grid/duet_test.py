import syft as sy


def test_duet():
    duet = sy.Duet(host="127.0.0.1", port=5001)
    obj_id = duet.id
    duet.stop()
    assert "SpecificLocation" in f"{obj_id}"
