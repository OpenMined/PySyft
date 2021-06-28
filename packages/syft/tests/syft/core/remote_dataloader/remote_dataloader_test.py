# stdlib
import os

# third party
import pytest
import torch as th
from torch.utils.data import Dataset

# syft absolute
import syft as sy
from syft.core.remote_dataloader import RemoteDataLoader
from syft.core.remote_dataloader import RemoteDataset


class ExampleDataset(Dataset):
    def __init__(self, ten: th.Tensor):
        self.ten = ten

    def __len__(self) -> int:
        return self.ten.shape[0]

    def __getitem__(self, i: int) -> th.Tensor:
        return self.ten[i]


ten = th.rand((1000, 4))
ds = ExampleDataset(ten)


@pytest.mark.slow
def test_remote_dataset(client: sy.VirtualMachineClient) -> None:
    filename = "ds1.pt"
    th.save(ds, filename)

    rds = RemoteDataset(path=filename, data_type="torch_tensor")
    rds_ptr = rds.send(client)
    rds_ptr.load_dataset()

    assert rds_ptr.len().get() == 1000
    for tp in rds_ptr:
        assert isinstance(tp.get(), th.Tensor)
    os.unlink(filename)


@pytest.mark.slow
def test_remote_dataloader(root_client: sy.VirtualMachineClient) -> None:
    filename = "ds2.pt"
    th.save(ds, filename)

    rds = RemoteDataset(path=filename, data_type="torch_tensor")
    rdl = RemoteDataLoader(remote_dataset=rds, batch_size=4)
    rdl_ptr = rdl.send(root_client)

    rdl_ptr.load_dataset()
    rdl_ptr.create_dataloader()

    assert rdl_ptr.len().get() == 250
    for tp in rdl_ptr:
        assert isinstance(tp.get(), th.Tensor)
    os.unlink(filename)
