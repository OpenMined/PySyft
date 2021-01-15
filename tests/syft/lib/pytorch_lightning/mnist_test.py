import syft as sy
from pytorch_lightning import Trainer, LightningModule
import torch
import torchvision


alice = sy.VirtualMachine(name="alice")
duet = alice.get_root_client()

class SyNet(sy.Module):
    def __init__(self, torch_ref):
        super(SyNet, self).__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = self.torch_ref.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = self.torch_ref.nn.Dropout2d(0.25)
        self.dropout2 = self.torch_ref.nn.Dropout2d(0.5)
        self.fc1 = self.torch_ref.nn.Linear(9216, 128)
        self.fc2 = self.torch_ref.nn.Linear(128, 10)

    def forward(self, x):
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
    def __init__(self, local_torch, download_back=False, set_to_local=False):
        super().__init__()
        self.local_model = SyNet(local_torch)
        self.remote_torch = duet.torch
        self.local_torch = local_torch
        self.download_back = download_back
        self.set_to_local = set_to_local
        self.get = self.local_model.get
        self.send = self.local_model.send

    def is_remote(self):
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
    def torchvision(self):
        return duet.torchvision if self.is_remote() else torchvision

    @property
    def torch(self):
        return self.remote_torch if self.is_remote() else self.local_torch

    @property
    def model(self):
        if self.is_remote():
            return self.remote_model
        else:
            if self.download_back:
                return self.get_model()
            else:
                return self.local_model

    def send_model(self):
        self.remote_model = self.local_model.send(duet)

    def get_model(self):
        return self.remote_model.get(
            request_block=True,
            name="model_download",
            reason="test evaluation",
            timeout_secs=5
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, output, target):
        return self.torch.nn.functional.nll_loss(output, target)

    def training_step(self, batch, batch_idx):
        data_ptr = batch[0]
        target_ptr = batch[1]
        data_ptr, target_ptr = batch[0], batch[1]
        output = self.forward(data_ptr)
        loss = self.loss(output, target_ptr)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch[0], batch[1]
        output = self.forward(data)
        loss = self.loss(output, target)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.torch.optim.SGD(self.model.parameters(), lr=0.1)
        return [optimizer]

    def get_transforms(self):
        current_list = duet.python.List if self.is_remote() else list
        transforms = current_list()
        transforms.append(self.torchvision.transforms.ToTensor())
        transforms.append(self.torchvision.transforms.Normalize(0.1307, 0.3081))
        return self.torchvision.transforms.Compose(transforms)

    def train_dataloader(self):
        transforms = self.get_transforms()
        train_data_ptr = self.torchvision.datasets.MNIST('../data', train=True, download=True,
                                                         transform=transforms)
        train_loader_ptr = self.torch.utils.data.DataLoader(train_data_ptr, batch_size=64)
        return train_loader_ptr

    def test_dataloader(self):
        transforms = self.get_transforms()
        test_data = self.torchvision.datasets.MNIST('../data', train=False, download=True,
                                                    transform=transforms)
        test_loader = self.torch.utils.data.DataLoader(test_data, batch_size=64)
        return test_loader

model = Model(torch, download_back=False)
model.send_model()

trainer = Trainer(
    default_root_dir='./tmp_data',
    max_epochs=2,
    limit_train_batches=2,
    limit_test_batches=2,
    accumulate_grad_batches=2
)

trainer.test(model)
trainer.fit(model)

model.download_back = True
trainer.test(model)