def split(dataset, batch_size, split=0.2):
    """Splits the given dataset into training/validation.
       Args:
           dataset[torch dataloader]: Dataset which has to be split
           batch_size[int]: Batch size
           split[float]: Indicates ratio of validation samples
       Returns:
           train_set[list]: Training set
           val_set[list]: Validation set
    """

    index = 0
    length = len(dataset)

    train_set = []
    val_set = []

    for data, target in dataset:
        if index <= (length * split):
            train_set.append([data, target])
        else:
            val_set.append([data, target])

        index += 1

    return train_set, val_set


def accuracy(predictions, dataset):
    """Evaluates accuracy for given set of predictions and true labels.
       Args:
           predictions[torch tensor]: predictions made by classifier.
           labels[torch tensor]: true labels of the dataset.
       Returns:
           accuracy[float]: accuracy of classifier.
    """

    total = 0.0
    correct = 0.0

    for j in range(0, len(dataset)):
        correct += (predictions[j].long() == dataset[j].long()).sum().item()
        total += len(dataset[j])

    return (correct / total) * 100


def plot(x, y):
    """Plots a graph of given x and y.
       Args:
           
           x:
           y:
    """
    pass


def histogram(x, y):
    """Plots a histogram for corresponding x and y:
       Args:
           
           x:
           y:
    """
