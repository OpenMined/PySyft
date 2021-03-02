# stdlib
from types import ModuleType
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
import pytest
import torch
from torch import nn
import torchvision
from torchvision import transforms

# syft absolute
import syft as sy
from syft.ast.module import Module


@pytest.mark.vendor(lib="pytorch_lightning")
def test_mnist() -> None:
    # third party
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer

    # TODO: Automatically synthesize these Union types
    SyModuleProxyType = Union[ModuleType, Module]
    SyModelProxyType = Union[nn.Module, sy.Module]

    # cant use lib_ast during test search time
    TorchTensorPointerType = Any  # sy.lib_ast.torch.Tensor.pointer_type
    TorchDataLoaderPointerType = Any  # sy.lib_ast.torch.utils.data.DataLoader
    SyTensorProxyType = Union[torch.Tensor, TorchTensorPointerType]  # type: ignore
    SyDataLoaderProxyType = Union[torch.utils.data.DataLoader, TorchDataLoaderPointerType]  # type: ignore

    sy.logger.remove()
    alice = sy.VirtualMachine(name="alice")
    duet = alice.get_root_client()
    sy.client_cache["duet"] = duet

    class SyNet(sy.Module):
        def __init__(self, torch_ref: SyModuleProxyType) -> None:
            super(SyNet, self).__init__(torch_ref=torch_ref)
            self.conv1 = self.torch_ref.nn.Conv2d(1, 32, 3, 1)
            self.conv2 = self.torch_ref.nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = self.torch_ref.nn.Dropout2d(0.25)
            self.dropout2 = self.torch_ref.nn.Dropout2d(0.5)
            self.fc1 = self.torch_ref.nn.Linear(9216, 128)
            self.fc2 = self.torch_ref.nn.Linear(128, 10)

        def forward(self, x: SyTensorProxyType) -> SyTensorProxyType:
            x = self.conv1(x)
            x = self.torch_ref.nn.functional.relu(x)
            x = self.conv2(x)
            x = self.torch_ref.nn.functional.relu(x)
            x = self.torch_ref.nn.functional.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = self.torch_ref.flatten(x, 1)
            x = self.fc1(x)
            x = self.torch_ref.nn.functional.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = self.torch_ref.nn.functional.log_softmax(x, dim=1)
            return output

    class Model(LightningModule):
        def __init__(
            self,
            local_torch: ModuleType,
            download_back: bool = False,
            set_to_local: bool = False,
        ) -> None:
            super().__init__()
            self.local_model = SyNet(local_torch)
            self.remote_torch = duet.torch
            self.local_torch = local_torch
            self.download_back = download_back
            self.set_to_local = set_to_local
            self.get = self.local_model.get
            self.send = self.local_model.send

        def is_remote(self) -> bool:
            # used to test out everything works locally
            if self.set_to_local:
                return False
            elif not self.trainer.testing:
                return True
            elif self.trainer.evaluation_loop.testing:
                return False
            else:
                return True

        @property
        def torchvision(self) -> SyModuleProxyType:
            return duet.torchvision if self.is_remote() else torchvision

        @property
        def torch(self) -> SyModuleProxyType:
            return self.remote_torch if self.is_remote() else self.local_torch

        @property
        def model(self) -> SyModelProxyType:
            if self.is_remote():
                return self.remote_model
            else:
                if self.download_back:
                    return self.get_model()
                else:
                    return self.local_model

        def send_model(self) -> None:
            self.remote_model = self.local_model.send(duet)

        def get_model(self) -> type(nn.Module):  # type: ignore
            return self.remote_model.get()

        def forward(self, x: SyTensorProxyType) -> SyTensorProxyType:
            return self.model(x)

        def loss(
            self, output: SyTensorProxyType, target: SyTensorProxyType
        ) -> SyTensorProxyType:
            return self.torch.nn.functional.nll_loss(output, target)  # type: ignore

        def training_step(
            self, batch: SyTensorProxyType, batch_idx: Optional[int]
        ) -> SyTensorProxyType:
            data_ptr, target_ptr = batch[0], batch[1]  # type: ignore
            output = self.forward(data_ptr)
            loss = self.loss(output, target_ptr)
            return loss

        def test_step(self, batch: SyTensorProxyType, batch_idx: Optional[int]) -> None:
            data, target = batch[0], batch[1]  # type: ignore
            output = self.forward(data)
            loss = self.loss(output, target)
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        def configure_optimizers(self) -> List:
            optimizer = self.torch.optim.SGD(self.model.parameters(), lr=0.1)  # type: ignore
            return [optimizer]

        def get_transforms(self) -> type(transforms.transforms.Compose):  # type: ignore
            current_list = duet.python.List if self.is_remote() else list
            transforms = current_list()
            transforms.append(self.torchvision.transforms.ToTensor())  # type: ignore
            transforms.append(self.torchvision.transforms.Normalize(0.1307, 0.3081))  # type: ignore
            return self.torchvision.transforms.Compose(transforms)  # type: ignore

        def train_dataloader(self) -> SyDataLoaderProxyType:
            transforms = self.get_transforms()
            train_data_ptr = self.torchvision.datasets.MNIST(  # type: ignore
                "../data", train=True, download=True, transform=transforms
            )
            train_loader_ptr = self.torch.utils.data.DataLoader(  # type: ignore
                train_data_ptr, batch_size=1
            )
            return train_loader_ptr

        def test_dataloader(self) -> SyDataLoaderProxyType:
            transforms = self.get_transforms()
            test_data = self.torchvision.datasets.MNIST(  # type: ignore
                "../data", train=False, download=True, transform=transforms
            )
            test_loader = self.torch.utils.data.DataLoader(test_data, batch_size=1)  # type: ignore
            return test_loader

    model = Model(torch, download_back=False)
    model.send_model()

    trainer = Trainer(
        default_root_dir="./tmp_data",
        max_epochs=15,
        accumulate_grad_batches=True,
        limit_train_batches=4,
        limit_test_batches=4,
    )

    trainer.fit(model)

    model.download_back = True
    trainer.test(model)

    assert True is True
