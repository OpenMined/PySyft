import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Network1(nn.Module):
    def __init__(self, out_features):
        super(Network1, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_features)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x


class Network2(nn.Module):
    def __init__(self, out_features):
        super(Network2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0, stride=1)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)  ## inverted!
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)  ## inverted!
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


def resnet18(out_features):
    model = models.resnet18()
    model.maxpool, model.relu = model.relu, model.maxpool
    model.fc = nn.Linear(in_features=512, out_features=out_features)
    model.eval()
    return model


def vgg16(out_features):
    model = models.vgg16()

    # Invert ReLU <-> Maxpool
    for i, module in enumerate(model.features[:-1]):
        next_module = model.features[i + 1]
        if isinstance(module, nn.ReLU) and isinstance(next_module, nn.MaxPool2d):
            model.features[i + 1] = module
            model.features[i] = next_module

    class Empty(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    model.avgpool = Empty()

    # Hackalert
    if out_features == 10:
        first_linear = nn.Linear(512, 4096)
    elif out_features == 200:
        first_linear = nn.Linear(512 * 2 * 2, 4096)
    else:
        raise ValueError("VGG16 can't be built for this dataset, maybe modify it?")

    model.classifier = nn.Sequential(
        first_linear,
        nn.ReLU(True),
        # nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        # nn.Dropout(),
        nn.Linear(4096, out_features),
    )

    return model


model_zoo = {"network1": Network1, "network2": Network2, "resnet18": resnet18, "vgg16": vgg16}


def get_model(model_name, out_features):
    return model_zoo[model_name](out_features)
