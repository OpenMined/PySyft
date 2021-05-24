# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
import pytest
import torch

# syft absolute
import syft as sy

SyTensorProxyType = Any  # Union[torch.Tensor, Any]
# cant use lib_ast during test search time
DataLoaderPointerType = Any  # sy.lib_ast.torch.utils.data.DataLoader.pointer_type
SyDataloaderProxyType = Any  # Union[torch.utils.data.DataLoader, DataLoaderPointerType]


@pytest.mark.vendor(lib="pytorch_lightning")
def test_lightning() -> None:
    # third party
    from pytorch_lightning import Trainer
    from pytorch_lightning.experimental.plugins.secure.pysyft import SyLightningModule

    tmpdir = "./"

    duet = sy.VirtualMachine().get_root_client()
    sy.client_cache["duet"] = duet

    class BoringSyNet(sy.Module):
        def __init__(self, torch_ref: Any) -> None:
            super(BoringSyNet, self).__init__(torch_ref=torch_ref)
            self.fc2 = self.torch_ref.nn.Linear(32, 2)

        def forward(self, x: SyTensorProxyType) -> SyTensorProxyType:
            return self.fc2(x)

    class LiftSyLightningModule(SyLightningModule):
        def __init__(self, module: sy.Module, duet: Any) -> None:
            super().__init__(module, duet)

        def training_step(
            self, batch: SyTensorProxyType, batch_idx: Optional[int]
        ) -> SyTensorProxyType:
            data_ptr = batch
            output = self.forward(data_ptr)
            return self.torch.nn.functional.mse_loss(
                output, self.torch.ones_like(output)
            )

        def test_step(self, batch: SyTensorProxyType, batch_idx: Optional[int]) -> None:
            output = self.forward(batch)
            loss = self.loss(output, self.torch.ones_like(output))
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        def configure_optimizers(self) -> List:
            optimizer = self.torch.optim.SGD(self.model.parameters(), lr=0.1)
            return [optimizer]

        def train_dataloader(self) -> SyDataloaderProxyType:
            return self.torch.utils.data.DataLoader(self.torch.randn(64, 32))

    module = BoringSyNet(torch)
    model = LiftSyLightningModule(module=module, duet=duet)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_test_batches=2,
    )

    trainer.fit(model)
    trainer.test()
    trainer.test(model)

    model = LiftSyLightningModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, module=module, duet=duet
    )
    trainer.fit(model)

    del sy.client_cache["duet"]
