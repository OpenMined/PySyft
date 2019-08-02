import syft as sy


def test_message_serde(hook):

    x = sy.Message(0, [1, 2, 3])
    x_bin = sy.serde.serialize(x)
    y = sy.serde.deserialize(x_bin, sy.local_worker)

    assert x.contents == y.contents
    assert x.msg_type == y.msg_type
