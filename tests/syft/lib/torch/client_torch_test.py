import syft as sy
import torch as th


def test_torch_function() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_client()

    x = th.tensor([[-0.1, 0.1], [0.2, 0.3]])
    ptr_x = x.send(client)
    ptr_res = client.torch.zeros_like(ptr_x)
    res = ptr_res.get()

    assert (res == th.tensor([[0.0, 0.0], [0.0, 0.0]])).all()
