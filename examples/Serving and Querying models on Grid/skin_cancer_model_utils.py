"""Functions and variables shared accross the notebooks."""

from glob import glob
import os

import pandas as pd
from PIL import Image
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import syft as sy
import torch
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


DATASET_PATH = "./skin-cancer-mnist-ham10000"
test_generator = None


def read_skin_cancer_dataset():
    """Originally from https://www.kaggle.com/kmader/dermatology-mnist-loading-and-processing."""
    all_image_path = glob(os.path.join(DATASET_PATH, "*", "*.jpg"))

    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path
    }

    lesion_type_dict = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    df = pd.read_csv(os.path.join(DATASET_PATH, "HAM10000_metadata.csv"))
    df["path"] = df["image_id"].map(imageid_path_dict.get)
    df["cell_type"] = df["dx"].map(lesion_type_dict.get)
    # Binary classification
    df = df[df.cell_type.isin(["Melanoma", "Benign keratosis-like lesions"])]
    df["cell_type_idx"] = pd.Categorical(df["cell_type"]).codes
    df[["cell_type_idx", "cell_type"]].sort_values("cell_type_idx").drop_duplicates()

    return df


def split_data(df, test_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=221)
    validation_df, test_df = train_test_split(df, test_size=0.5, random_state=222)

    train_df = train_df.reset_index()
    validation_df = validation_df.reset_index()
    test_df = test_df.reset_index()
    return train_df, validation_df, test_df


def calculate_mean_and_std(loader):
    mean, std, nb_samples = 0.0, 0.0, 0.0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


def transform(input_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_model(
    num_classes: int = 2, is_plan=False, model_id="skin-cancer-model-encrypted"
):
    super_class = sy.Plan if is_plan else nn.Module

    class Net(super_class):
        """Similar to LeNet5 but without pooling."""

        def __init__(self):
            if is_plan:
                super(Net, self).__init__(id=model_id)
            else:
                super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5, 1)
            self.conv2 = nn.Conv2d(6, 16, 5, 1)
            self.fc1 = nn.Linear(9216, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

            if is_plan:
                self.add_to_state(["conv1", "conv2", "ffc1", "fc2", "fc3"])

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(-1, 9216)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()


def test(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    acc = 100.0 * correct / len(data_loader.dataset)

    print(
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
            test_loss, correct, len(data_loader.dataset), acc
        )
    )

    return test_loss, acc


def train(
    model, epochs, optimizer, training_generator, validation_generator, verbose=True
):
    train_metrics, valid_metrics = [], []

    step = 0
    for epoch in range(epochs):
        model.train()
        for data, target in training_generator:
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if verbose and step % 100 == 0:
                print("Training loss at step {} = {}".format(step, loss.item()))
            step += 1

        print("Epoch: {}".format(epoch))
        print("Train\n==============")
        train_metrics.append(test(model, training_generator))
        print("Test\n==============")
        valid_metrics.append(test(model, validation_generator))
    return train_metrics, valid_metrics


def plot_confusion_matrix(model, loader):
    # Predict the values from the validation dataset
    model.eval()

    model_output = torch.cat([model(x) for x, _ in loader])
    predictions = torch.argmax(model_output, dim=1)
    targets = torch.cat([y for _, y in loader])

    conf_matrix = confusion_matrix(targets, predictions)
    df_cm = pd.DataFrame(conf_matrix)
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})


def get_data_sample():
    global test_generator
    if not test_generator:
        df = read_skin_cancer_dataset()
        _, _, test_df = split_data(df)

        params = {"batch_size": 1, "shuffle": True, "num_workers": 6}

        # These values are from training
        input_size = 32
        train_mean, train_std = (
            torch.tensor([0.6979, 0.5445, 0.5735]),
            torch.tensor([0.0959, 0.1187, 0.1365]),
        )

        test_set = Dataset(
            test_df, transform=transform(input_size, train_mean, train_std)
        )
        test_generator = torch.utils.data.DataLoader(test_set, **params)

    return next(iter(test_generator))


class Dataset(torch.utils.data.Dataset):
    """Originally from https://towardsdatascience.com/skin-cancer-classification-with-machine-learning-c9d3445b2163."""

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df["path"][index])
        y = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            X = self.transform(X)

        return X, y
