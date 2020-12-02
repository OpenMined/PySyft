# stdlib
from typing import List as TypeList
from typing import Type as TypeType

# third party
import torch as th

# syft absolute
import syft as sy


def test_torch_function() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_client()

    x = th.tensor([[-0.1, 0.1], [0.2, 0.3]])
    ptr_x = x.send(client)
    ptr_res = client.torch.zeros_like(ptr_x)
    res = ptr_res.get()

    assert (res == th.tensor([[0.0, 0.0], [0.0, 0.0]])).all()


def test_path_cache() -> None:
    short_fqn = "torch.nn.Conv2d"
    conv2d_paths = [
        "torch.nn.modules.conv.Conv2d",
        "torch.nn.modules.Conv2d",
        "torch.nn.Conv2d",
    ]

    refs: TypeList[TypeType] = []
    for path in conv2d_paths:
        klass = sy.lib_ast(path, return_callable=True, obj_type=th.nn.Conv2d)
        assert klass == sy.lib_ast.torch.nn.Conv2d
        assert klass.name == "Conv2d"
        assert klass.path_and_name == short_fqn
        assert all(klass == ref for ref in refs)
        refs.append(klass)
