# third party
import torch as th
from torch.utils.data import Dataset

# syft absolute
import syft as sy
from syft.core.remote_dataloader import RemoteDataset


class ExampleDataset(Dataset):
    def __init__(self, ten: th.Tensor):
        self.ten = ten

    def __len__(self) -> int:
        return self.ten.shape[0]

    def __getitem__(self, i: int) -> th.Tensor:
        return self.ten[i]


def test_remote_dataset() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    ten = th.rand((1000, 4))

    ds = ExampleDataset(ten)
    th.save(ds, "ds.pt")

    rds = RemoteDataset("ds.pt")
    rds_ptr = rds.send(alice_client)
    rds_ptr.create_dataset()

    assert rds_ptr.len().get() == 1000

    tensor_pointer_type = type(th.rand(1).send(alice_client))
    for tp in rds_ptr:
        assert isinstance(tp, tensor_pointer_type)

    # stdlib
    import os

    os.system("rm ds.pt")
