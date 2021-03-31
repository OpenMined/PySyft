# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address

def actual_test(msg, target):
    if msg == None:
        pytest.skip("Test Not Needed")

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2

def abstract_test(constructor, content, kwargs):
    target = Address(name="Alice")

    msg = constructor(content=content, address=target, **kwargs)
    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2
