# third party
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# syft absolute
import syft as sy

data_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def set_model_opt_loss(input_size, torch_ref):
    class SyNet(sy.Module):
        def __init__(self, input_shape, torch_ref):
            super(SyNet, self).__init__(torch_ref=torch_ref)
            self.fc1 = self.torch_ref.nn.Linear(input_shape, 32)
            self.fc2 = self.torch_ref.nn.Linear(32, 64)
            self.fc3 = self.torch_ref.nn.Linear(64, 1)

        def forward(self, x):
            x = self.torch_ref.relu(self.fc1(x))
            x = self.torch_ref.relu(self.fc2(x))
            x = self.torch_ref.sigmoid(self.fc3(x))
            return x

    model = SyNet(input_shape=input_size, torch_ref=torch_ref)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


def load_and_process_data(data_path):
    global data_columns
    data = pd.read_csv(data_path)
    y = data["Outcome"]
    x = data[data_columns]
    sc = StandardScaler()
    x = sc.fit_transform(x)

    class DatasetStruct(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.length = self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

        def __len__(self):
            return self.length

    trainset = DatasetStruct(x, y.values)  # DataLoader
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)
    return x, y, trainset, trainloader


def train(model, optimizer, loss_function, x, y, trainloader, epochs):
    losses = []
    acc = []
    epochs_list = []
    for i in range(epochs):

        for j, (x_train, y_train) in enumerate(trainloader):
            # calculate output
            output = model(x_train)

            # calculate loss
            loss = loss_function(output, y_train.reshape(-1, 1))

            # accuracy
            predicted = model(torch.tensor(x, dtype=torch.float32))
            accuracy = (
                predicted.reshape(-1).detach().numpy().round() == y
            ).mean()  # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 30 == 0 and j == 0:
                losses.append(loss.detach().numpy())
                acc.append(accuracy)
                epochs_list.append(i)

            if i % 50 == 0 and j == 0:
                print(f"epoch {i}\tloss : {loss}\t accuracy : {accuracy}")

    return model, losses, acc, epochs_list


def train_diabetes_model(torch_ref):
    x, y, trainset, trainloader = load_and_process_data("diabetes.csv")
    model, optimizer, loss_function = set_model_opt_loss(8, torch_ref)
    return train(model, optimizer, loss_function, x, y, trainloader, 300)


def plot_training_acc(acc, loss, epochs):
    # third party
    import matplotlib.pyplot as plt

    # Plot the accuracy
    plt.plot(epochs, acc, label="accuracy")

    # Plot the loss
    plt.plot(epochs, loss, label="loss")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
