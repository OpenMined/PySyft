# stdlib
from typing import Any
from typing import List as TypeList

# third party
import numpy as np
import pytest
import torch as th
from torchvision import datasets
from torchvision import transforms

# syft absolute
import syft as sy
from syft import SyModule
from syft import SySequential
from syft import logger
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan
from syft.util import get_root_data_path

logger.remove()


class BasicBlock(SyModule):
    def __init__(
        self,
        f_in: int,
        f_out: int,
        stride1: int = 1,
        downsample: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.conv1 = th.nn.Conv2d(
            f_in, f_out, kernel_size=(3, 3), stride=stride1, padding=(1, 1), bias=False
        )
        self.bn1 = th.nn.BatchNorm2d(f_out)
        self.act1 = th.nn.ReLU()
        self.conv2 = th.nn.Conv2d(
            f_out, f_out, kernel_size=(3, 3), padding=(1, 1), bias=False
        )
        self.bn2 = th.nn.BatchNorm2d(f_out)
        self.act2 = th.nn.ReLU()
        if downsample is False:
            self.downsample = None
        else:
            self.downsample = SySequential(
                th.nn.Conv2d(f_in, f_out, kernel_size=(1, 1), stride=2, bias=False),
                th.nn.BatchNorm2d(f_out),
                input_size=self.input_size,
            )

    def forward(self, x: Any) -> Any:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(x=residual)[0]
        x += residual
        x = self.act2(x)
        return x


class ResNet18(SyModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # stem
        self.conv1 = th.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.bn1 = th.nn.BatchNorm2d(64)
        self.act1 = th.nn.ReLU()
        self.maxpool = th.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # blocks
        filters = [(64, 64), (64, 128), (128, 256), (256, 512)]
        input1_sizes = [(2, 64, 7, 7), (2, 64, 7, 7), (2, 128, 4, 4), (2, 256, 2, 2)]
        input2_sizes = [(2, 64, 7, 7), (2, 128, 7, 7), (2, 256, 4, 4), (2, 512, 2, 2)]

        for i in range(1, 5):
            downsample_first = i != 1
            f_in, f_out = filters[i - 1]
            f_in2 = f_out
            stride1 = 1 if i == 1 else 2
            input1_size = input1_sizes[i - 1]
            input2_size = input2_sizes[i - 1]

            layer = SySequential(
                BasicBlock(
                    f_in=f_in,
                    f_out=f_out,
                    downsample=downsample_first,
                    stride1=stride1,
                    input_size=input1_size,
                ),
                BasicBlock(f_in=f_in2, f_out=f_out, input_size=input2_size),
            )
            setattr(self, f"layer{i}", layer)

        # head
        self.global_pool = th.nn.AdaptiveAvgPool2d(1)
        self.fc = th.nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x: Any) -> Any:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # self.layern are user defined layers and therefore need the self.layern(x=x)[0] stuff
        x = self.layer1(x=x)[0]
        x = self.layer2(x=x)[0]
        x = self.layer3(x=x)[0]
        x = self.layer4(x=x)[0]
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x


# this currently takes 230 seconds, 60 of which is just making the model plans
# we need to heavily optimise this before we can include it in CI
@pytest.mark.skip
def test_resnet_18_custom_blocks(client: sy.VirtualMachineClient) -> None:
    cifar10_path = get_root_data_path()
    cifar10_path.mkdir(exist_ok=True, parents=True)
    norm = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

    cifar_train = datasets.CIFAR10(
        cifar10_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*norm),
            ]
        ),
    )

    cifar_test = datasets.CIFAR10(
        (cifar10_path),
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*norm)]
        ),
    )

    train_batch = 64
    test_batch = 1000

    dry_run = True

    if dry_run:
        train_batch = 2
        test_batch = 2

    train_loader = th.utils.data.DataLoader(
        cifar_train, batch_size=train_batch, shuffle=True, pin_memory=True
    )
    test_loader = th.utils.data.DataLoader(
        cifar_test, batch_size=test_batch, shuffle=True, pin_memory=True
    )

    model = ResNet18(input_size=(2, 3, 32, 32))

    remote_torch = ROOT_CLIENT.torch
    dummy_dl = sy.lib.python.List([next(iter(train_loader))])

    @make_plan
    def train(dl: sy.lib.python.List = dummy_dl, model: SyModule = model) -> TypeList:

        optimizer = remote_torch.optim.AdamW(model.parameters())

        for xy in dl:
            optimizer.zero_grad()
            x, y = xy[0], xy[1]
            out = model(x=x)[0]
            loss = remote_torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        return [model]

    def test(test_loader: th.utils.data.DataLoader, model: SyModule) -> float:
        correct = []
        model.eval()

        for data, target in test_loader:
            output = model(x=data)[0]
            _, pred = th.max(output, 1)
            correct.append(th.sum(np.squeeze(pred.eq(target.data))))
            if dry_run:
                break
        acc = sum(correct) / len(test_loader.dataset)
        return acc

    train_ptr = train.send(client)

    for i, (x, y) in enumerate(train_loader):
        dl = [[x, y]]

        res_ptr = train_ptr(dl=dl, model=model)
        (model,) = res_ptr.get()

        if (i % 10 == 0 and i != 0) or dry_run:
            acc = test(test_loader, model)
            print(f"Iter: {i} Test accuracy: {acc:.2F}", flush=True)

        if i > 50 or dry_run:
            break
