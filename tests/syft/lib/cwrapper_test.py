# stdlib
import uuid

# third party
import torch as th

# syft absolute
from syft.core.common.uid import UID

# from syft.lib.cwrapper import CWrapperFactory


def test_wrapper_device() -> None:
    # we create a normal device
    d = th.device("cpu")
    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    d.id = uid
    comp = th.Tensor([[1.0]])

    # pass it into a torch function that takes a device
    out = th.eye(n=1, device=d.upcast())

    assert out == comp
    assert d.type == "cpu"
    assert d.index is None
    assert d.id == uid

    # our normal device is an instance and a subclass of torch.device
    # assert issubclass(type(d), th.device)
    assert isinstance(d, th.device)

    # Used while developing the MetaClass Wrapper
    # # we create a device with our CWrapper
    # FakeDevice = CWrapperFactory(shadow_type=th.device)
    # f = FakeDevice("cpu")

    # # set a UID
    # uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    # f.id = uid

    # # check the normal properties
    # assert f.type == "cpu"
    # assert f.index is None

    # # TODO: Fix, currently it is instance but not subclass
    # # assert issubclass(type(f), th.device)
    # assert isinstance(f, th.device)

    # # compare the two in lots of important ways
    # # TODO: Fix matching the types
    # # assert type(d) == type(f)
    # assert d == f
    # assert d.__class__ == f.__class__
    # assert type(d).mro() == type(f).mro()

    # # make sure we still have a UID
    # assert f.id == uid

    # # pass FakeDevice into a torch function
    # # out2 = th.eye(n=1, device=f)
    # # TODO: Fix, currently requires upcast
    # out2 = th.eye(n=1, device=f.upcast())

    # # the function works as expected and doesn't raise an exception
    # assert out2 == comp
