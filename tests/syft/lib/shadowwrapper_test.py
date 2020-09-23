# stdlib
import uuid

# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.shadowwrapper import ShadowWrapperMeta
from syft.lib.util import full_name_with_qualname


def test_wrapper_with_device() -> None:
    # we create a wrapped torch.device which is a special constructor class in torch
    # which Python will not let us subclass
    d = th.device("cpu")

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    d.id = uid

    # create something to compare with
    comp = th.Tensor([[1]])
    assert full_name_with_qualname(klass=type(comp)) == "syft.proxy.torch.Tensor"

    # confirm out wrapped device responds as expected
    assert d.type == "cpu"
    assert d.index is None

    # make sure have the UID id property we need
    assert d.id == uid

    # check if our device is an instance of the original constructor
    assert isinstance(d, th.device.unmodified_original_constructor)

    # the original issubclass of itself
    assert issubclass(
        th.device.unmodified_original_constructor,
        th.device.unmodified_original_constructor,
    )
    # however we cannot get issubclass to work yet, perhaps this will never be possible
    assert not issubclass(type(d), th.device.unmodified_original_constructor)

    # create an instance from the unmodified_original_constructor
    real_d = th.device.unmodified_original_constructor("cpu")
    assert issubclass(type(real_d), th.device.unmodified_original_constructor)

    # compare the two versions in lots of different ways
    assert full_name_with_qualname(klass=type(d)) == "syft.proxy.torch.device"
    assert full_name_with_qualname(klass=type(real_d)) == "torch.device"
    assert d == real_d
    assert d.__class__ == real_d.__class__
    assert set(type(real_d).mro()).issubset(set(type(d).mro()))

    # pass it into a torch function that takes a device, and use upcast
    out = th.ones(1, device=d.upcast())
    assert out == comp

    # the same as the real device
    out2 = th.ones(1, device=real_d)
    assert out2 == comp

    # check the _wrapped upcast version also matches
    assert d.upcast() == real_d
    assert issubclass(type(d.upcast()), th.device.unmodified_original_constructor)


def test_wrapper_with_device_remotely() -> None:
    # device does not have serde support since this probably makes no sense
    # however we will want to create devices on remote machines

    # create a VM and get a root client reference
    bob = sy.VirtualMachine(name="bob")
    root_bob_client = bob.get_root_client()

    # get a reference to bobs torch
    bob_th = root_bob_client.torch

    local_string = sy.lib.python.String("cpu")
    string_ptr = local_string.send(root_bob_client)

    # make a device on bobs system using a string which was sent over
    device_ptr = bob_th.device(string_ptr)
    assert (
        full_name_with_qualname(klass=type(device_ptr))
        == "syft.proxy.torch.devicePointer"
    )

    # make an int and send it over
    local_int = sy.lib.python.Int(1)
    int_ptr = local_int.send(root_bob_client)

    # make a tensor using torch.ones and the int that was sent over
    # pass in the remotely generated device to torch.ones
    out_ptr = bob_th.ones(int_ptr, device=device_ptr)
    assert (
        full_name_with_qualname(klass=type(out_ptr)) == "syft.proxy.torch.TensorPointer"
    )

    # get the result and make sure its as expected
    out = out_ptr.get()
    assert out == th.Tensor([1])


def test_wrapper_with_none() -> None:
    # this isn't used but demonstrates that ShadowWrapper can also wrap these types
    class MagicNone(metaclass=ShadowWrapperMeta, shadow_type=None):  # type: ignore

        # satisfies mypy
        def upcast(self) -> bool:
            return getattr(self, "upcast")()

        def id(self) -> bool:
            return getattr(self, "id")()

    NoneType = type(None)

    b = MagicNone(None)  # type: ignore

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    b.id = uid  # type: ignore

    # create something to compare with
    comp = None

    # confirm out wrapped device responds as expected
    assert b == None  # noqa: E711
    assert b == comp

    # make sure have the UID id property we need
    assert b.id == uid

    # check if our MagicBool is an instance of the original constructor
    assert isinstance(b, NoneType)  # type: ignore

    # the original issubclass of itself
    assert issubclass(type(comp), type(comp))

    # however we cannot get issubclass to work yet, perhaps this will never be possible
    assert not issubclass(type(b), NoneType)  # type: ignore

    # unlike the original which works as expected
    assert issubclass(type(comp), NoneType)  # type: ignore

    # compare the two versions in lots of different ways
    assert full_name_with_qualname(klass=type(b)) == "syft.proxy.builtins.NoneType"
    assert full_name_with_qualname(klass=type(comp)) == "builtins.NoneType"
    assert b == comp
    assert b.__class__ == comp.__class__  # type: ignore
    assert set(type(comp).mro()).issubset(set(type(b).mro()))

    # check the _wrapped upcast version also matches
    assert b.upcast() == comp
    assert issubclass(type(b.upcast()), NoneType)  # type: ignore


def test_wrapper_with_bool() -> None:
    # this isn't used but demonstrates that ShadowWrapper can also wrap these types
    class MagicBool(metaclass=ShadowWrapperMeta, shadow_type=bool):  # type: ignore

        # satisfies mypy
        def upcast(self) -> bool:
            return getattr(self, "upcast")()

        def id(self) -> bool:
            return getattr(self, "id")()

    b = MagicBool(True)  # type: ignore

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    b.id = uid  # type: ignore

    # create something to compare with
    comp = bool(True)

    # confirm out wrapped device responds as expected
    assert b == bool(True)  # type: ignore
    assert b == comp  # type: ignore

    # make sure have the UID id property we need
    assert b.id == uid

    # check if our MagicBool is an instance of the original constructor
    assert isinstance(b, bool)

    # the original issubclass of itself
    assert issubclass(type(comp), type(comp))

    # however we cannot get issubclass to work yet, perhaps this will never be possible
    assert not issubclass(type(b), bool)

    # unlike the original which works as expected
    assert issubclass(type(comp), bool)

    # compare the two versions in lots of different ways
    assert full_name_with_qualname(klass=type(b)) == "syft.proxy.builtins.bool"
    assert full_name_with_qualname(klass=type(comp)) == "builtins.bool"
    assert b == comp
    assert b.__class__ == comp.__class__
    assert set(type(comp).mro()).issubset(set(type(b).mro()))

    # check the _wrapped upcast version also matches
    assert b.upcast() == comp
    assert issubclass(type(b.upcast()), bool)
