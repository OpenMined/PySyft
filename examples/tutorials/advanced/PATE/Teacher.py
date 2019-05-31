import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.laplace import Laplace
from util import accuracy
from syft.frameworks.torch.differential_privacy import pate


class Teacher:
    """Implementation of teacher models.
       Teacher models are ensemble of models which learns directly disjoint splits of the sensitive data
       The ensemble of teachers are further used to label unlabelled public data on which the student is 
       trained. 
       Args:
           args[Arguments object]: An object of Arguments class with required hyperparameters
           n_teachers[int]: Number of teachers
           epochs[int]: Number of epochs to train each model
    """

    def __init__(self, args, model, n_teachers=1, epsilon=0.5):

        self.n_teachers = n_teachers
        self.model = model
        self.models = {}
        self.args = args
        self.init_models()
        self.epsilon = 0.5

    def init_models(self):
        """Initialize teacher models according to number of required teachers"""

        name = "model_"
        for index in range(0, self.n_teachers):

            model = self.model()
            self.models[name + str(index)] = model

    def addnoise(self, x):
        """Adds Laplacian noise to histogram of counts
           Args:
                counts[torch tensor]: Histogram counts
                epsilon[integer]:Amount of Noise
           Returns:
                counts[torch tensor]: Noisy histogram of counts
        """

        m = Laplace(torch.tensor([0.0]), torch.tensor([self.epsilon]))
        count = x + m.sample()

        return count

    def split(self, dataset):
        """Function to split the dataset into non-overlapping subsets of the data
           Args:
               dataset[torch tensor]: The dataset in the form of (image,label)
           Returns:
               split: Split of dataset
        """

        ratio = int(len(dataset) / self.n_teachers)
        iters = 0
        index = 0
        split = []
        last_batch = ratio * self.n_teachers

        for teacher in range(0, self.n_teachers):

            split.append([])

        for (data, target) in dataset:
            if (iters) % ratio == 0 and iters != 0:

                index += 1

            split[index].append([data, target])
            iters += 1

            if iters == last_batch:
                return split

        return split

    def train(self, dataset):
        """Function to train all teacher models.
           Args:
                dataset[torch tensor]: Dataset used to train teachers in format (image,label)
        """

        split = self.split(dataset)

        for epoch in range(1, self.args.epochs + 1):

            index = 0
            for model_name in self.models:

                print("TRAINING ", model_name)
                print("EPOCH: ", epoch)
                self.loop_body(split[index], model_name, 1)
                index += 1

    def loop_body(self, split, model_name, epoch):
        """Body of the training loop.
           Args:
               split: Split of the dataset which the model has to train.
               model_name: Name of the model.
               epoch: Epoch for which the model is being trained.
        """

        model = self.models[model_name]
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iters = 0
        loss = 0.0
        for (data, target) in split:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            iters += 1
        # Print loss by making using of log intervals
        print("Loss")
        print(loss.item())

    def aggregate(self, model_votes, batch_size):
        """Aggregate model output into a single tensor of votes of all models.
           Args:
                votes: Model output
                n_dataset: Number of datapoints
           Returns:
                counts: Torch tensor with counts across all models    
           """

        counts = torch.zeros([batch_size, 10])
        model_counts = torch.zeros([self.args.n_teachers, batch_size])
        model_index = 0

        for model in model_votes:

            index = 0

            for tensor in model_votes[model]:
                for val in tensor:

                    counts[index][val] += 1
                    model_counts[model_index][index] = val
                    index += 1

            model_index += 1

        return counts, model_counts

    def save_models(self):
        no = 0
        for model in self.models:

            torch.save(self.models[model].state_dict(), "models/" + model)
            no += 1

        print("\n")
        print("MODELS SAVED")
        print("\n")

    def load_models(self):

        path_name = "model_"

        for i in range(0, self.args.n_teachers):

            modelA = self.model()
            self.models[path_name + str(i)] = torch.load("models/" + path_name + str(i))
            self.models[path_name + str(i)] = modelA.load_state_dict()

    def analyze(self, preds, indices, moments=8):

        datadepeps, dataindeps = pate.perform_analysis_torch(
            preds, indices, noise_eps=0.1, delta=self.epsilon, moments=moments, beta=0.09
        )
        return datadepeps, dataindeps

    def predict(self, data):
        """Make predictions using Noisy-max using Laplace mechanism.
           Args:
                data: Data for which predictions are to be made
           Returns:
                predictions: Predictions for the data
        """

        model_predictions = {}

        for model in self.models:

            out = []
            output = self.models[model](data)
            output = output.max(dim=1)[1]
            out.append(output)

            model_predictions[model] = out

        counts, model_counts = self.aggregate(model_predictions, len(data))
        counts = counts.apply_(self.addnoise)

        predictions = []

        for batch in counts:

            predictions.append(torch.tensor(batch.max(dim=0)[1].long()).clone().detach())

        output = {"predictions": predictions, "counts": counts, "model_counts": model_counts}

        return output
