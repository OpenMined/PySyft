# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.util import full_name_with_qualname


def test_subclassing_klass() -> None:
    # demonstrate a normal wrapped syft.proxy.torch.Tensor
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_client()

    x = th.Tensor([[-0.1, 0.1], [0.2, 0.3]])
    ptr_x = x.send(client)
    ptr_res = client.torch.zeros_like(ptr_x)
    res = ptr_res.get()

    assert (res == th.tensor([[0.0, 0.0], [0.0, 0.0]])).all()
    assert full_name_with_qualname(klass=type(x)) == "syft.proxy.torch.Tensor"

    torch = client.torch

    # now lets try subclassing
    class MyTensor(torch.Tensor):  # type: ignore
        pass

    y = MyTensor([[-0.1, 0.1], [0.2, 0.3]])
    y._id = UID()
    # TODO: we should change the name and mro I guess?
    assert full_name_with_qualname(klass=type(y)) == "syft.proxy.torch.Tensor"

    ptr_y = y.send(client)
    ptr_res2 = client.torch.zeros_like(ptr_y)
    res2 = ptr_res2.get()

    assert (res2 == th.tensor([[0.0, 0.0], [0.0, 0.0]])).all()
