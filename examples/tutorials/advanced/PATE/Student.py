import torch
import torch.nn.functional as F
import torch.optim as optim


class Student:
    """Implementation of Student models
       The student model is trained from the public data labelled by teacher ensembles.
       The teacher ensembles were trained using sensitive data. The student model is further
       used to make predictions on public data.
       Args:
           args[Arguments obj]: Object of arguments class used to control hyperparameters
           model[torch model]: Model of Student 
    """

    def __init__(self, args, model):

        self.args = args
        self.model = model

    def predict(self, data):
        """Function which accepts unlabelled public data and labels it using 
           teacher's model.
           Args:
               model[torch model]: Teachers model
               data [torch tensor]: Public unlabelled data
           Returns:
               dataset[Torch tensor]: Labelled public dataset
        """

        return torch.max(self.model(data), 1)[1]

    def train(self, dataset):
        """Function to train the student model.
           Args:
               dataset[torch dataset]: Dataset using which model is trained.
        """

        for epoch in range(0, self.args.student_epochs):
            self.loop_body(dataset, epoch)

    def loop_body(self, dataset, epoch):
        """Body of the training loop.
           Args:
               dataset: dataset which is used to train the model.
               epoch: Epoch for which the model is being trained.
        """

        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iters = 0
        loss = 0.0
        for (data, target) in dataset:
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            iters += 1
        # Print loss by making using of log intervals
        print("\n")
        print("EPOCH")
        print(epoch)
        print("\n")
        print("Loss")
        print(loss.item())

    def save_model(self):
        torch.save(self.model.state_dict(), "Models/" + "student_model")
