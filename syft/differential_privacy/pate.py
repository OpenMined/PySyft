# The code is based on the following repo:
# https://github.com/tensorflow/models/tree/master/research/differential_privacy/multiple_teachers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import math
import numpy as np
import os
from scipy.io import loadmat as loadmat
from six.moves import urllib
from six.moves import xrange
import sys
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ckpt_path = "checkpoint/"


def prepare_mnist():
    kwargs = {"num_workers": 1}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=60000,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=10000,
        shuffle=False,
        **kwargs,
    )

    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))

    return train_data, train_labels, test_data, test_labels


def partition_dataset(data, labels, nb_teachers, teacher_id):
    """Simple partitioning algorithm that returns the right portion of the data
    needed by a given teacher out of a certain nb of teachers.

    :param data: input data to be partitioned
    :param labels: output data to be partitioned
    :param nb_teachers: number of teachers in the ensemble (affects size of each
                        partition)
    :param teacher_id: id of partition to retrieve
    :return:
    """

    # Sanity check
    assert len(data) == len(labels)
    assert int(teacher_id) < int(nb_teachers)

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    # Compute start, end indices of partition
    start = teacher_id * batch_len
    end = (teacher_id + 1) * batch_len

    # Slice partition off
    partition_data = data[start:end]
    partition_labels = labels[start:end]

    return partition_data, partition_labels


class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(model, train_loader, test_loader, ckpt_path, filename):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        model.train()  # set model to training mode

        # set up training metrics we want to track
        correct = 0
        train_num = len(train_loader.sampler)

        for ix, (img, label) in enumerate(
            train_loader
        ):  # iterate over training batches
            # img, label = img.to(device), label.to(device) # get data, send to gpu if needed
            img = img.type(torch.float32)
            # label = label.type(torch.float32)
            label = label.type(torch.LongTensor)
            optimizer.zero_grad()  # clear parameter gradients from previous training update
            output = model(img)  # forward pass
            # output = output.type(torch.float32)
            loss = F.cross_entropy(
                output, label, size_average=False
            )  # calculate network loss
            loss.backward()  # backward pass
            optimizer.step()  # take an optimization step to update model's parameters

            pred = output.max(1, keepdim=True)[1]  # get the index of the max logit
            correct += (
                pred.eq(label.view_as(pred)).sum().item()
            )  # add to running total of hits

        # print whole epoch's training accuracy; useful for monitoring overfitting
        print(
            "Train Accuracy: {}/{} ({:.0f}%)".format(
                correct, train_num, 100.0 * correct / train_num
            )
        )

    # set up training metrics we want to track
    test_correct = 0
    test_num = len(test_loader.sampler)
    with torch.no_grad():
        for ix, (img, label) in enumerate(test_loader):  # iterate over training batches
            # img, label = img.to(device), label.to(device) # get data, send to gpu if needed
            img = img.type(torch.float32)
            # label = label.type(torch.float32)
            label = label.type(torch.LongTensor)
            optimizer.zero_grad()  # clear parameter gradients from previous training update
            output = model(img)  # forward pass
            # output = output.type(torch.float32)
            loss = F.cross_entropy(
                output, label, size_average=False
            )  # calculate network loss

            pred = output.max(1, keepdim=True)[1]  # get the index of the max logit
            test_correct += (
                pred.eq(label.view_as(pred)).sum().item()
            )  # add to running total of hits

            # print whole epoch's training accuracy; useful for monitoring overfitting
        print(
            "Test Accuracy: {}/{} ({:.0f}%)".format(
                test_correct, test_num, 100.0 * test_correct / test_num
            )
        )

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    torch.save(model.state_dict(), ckpt_path + filename)


def train_teachers(
    model,
    train_data,
    train_labels,
    test_data,
    test_labels,
    nb_teachers,
    teacher_id,
    filename,
):

    data, labels = partition_dataset(train_data, train_labels, nb_teachers, teacher_id)

    train_prep = PrepareData(data, labels)
    train_loader = DataLoader(train_prep, batch_size=64, shuffle=True)

    test_prep = PrepareData(test_data, test_labels)
    test_loader = DataLoader(test_prep, batch_size=64, shuffle=False)

    print("\nTrain teacher ID: " + str(teacher_id))

    train(model, train_loader, test_loader, ckpt_path, filename)


def softmax_preds(model, nb_labels, images_loader, ckpt_path, return_logits=False):
    """Compute softmax activations (probabilities) with the model saved in the
    path specified as an argument.

    :param images: a np array of images
    :param ckpt_path: a TF model checkpoint
    :param logits: if set to True, return logits instead of probabilities
    :return: probabilities (or logits if logits is set to True)
    """
    # Compute nb samples and deduce nb of batches
    data_length = len(images_loader.dataset)
    preds = np.zeros((data_length, nb_labels), dtype=np.float32)
    start = 0

    check = torch.load(ckpt_path)
    model.load_state_dict(check)
    model.eval()  # set model to evaluate mode

    with torch.no_grad():
        for img, label in images_loader:
            output = model(img)
            output_softmax = F.softmax(output).data.numpy()

            end = start + len(img)

            preds[start:end, :] = output_softmax

            start += len(img)

    return preds


def ensemble_preds(model, dataset, nb_labels, nb_teachers, stdnt_data_loader):
    """Given a dataset, a number of teachers, and some input data, this helper
    function queries each teacher for predictions on the data and returns all
    predictions in a single array. (That can then be aggregated into one single
    prediction per input using aggregation.py (cf. function
    prepare_student_data() below)

    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :param stdnt_data: unlabeled student training data
    :return: 3d array (teacher id, sample id, probability per class)
    """

    # Compute shape of array that will hold probabilities produced by each
    # teacher, for each training point, and each output class
    result_shape = (nb_teachers, len(stdnt_data_loader.dataset), nb_labels)

    # Create array that will hold result
    result = np.zeros(result_shape, dtype=np.float32)

    # Get predictions from each teacher
    for teacher_id in xrange(nb_teachers):
        # Compute path of checkpoint file for teacher model with ID teacher_id
        filename = (
            str(dataset)
            + "_"
            + str(nb_teachers)
            + "_teachers_"
            + str(teacher_id)
            + ".pth"
        )
        # Get predictions on our training data and store in result array
        result[teacher_id] = softmax_preds(
            model, nb_labels, stdnt_data_loader, ckpt_path + filename
        )

        # This can take a while when there are a lot of teachers so output status
        print("Computed Teacher " + str(teacher_id) + " softmax predictions")

    return result


def prepare_student_data(
    model, dataset, nb_labels, nb_teachers, stdnt_share, lap_scale
):
    """Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model, according to parameters indicated in
    flags above.

    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: pairs of (data, labels) to be used for student training and testing
    """
    _, _, test_data, test_labels = prepare_mnist()

    # Transfor tensor to numpy
    test_labels = test_labels.data.numpy()

    # Make sure there is data leftover to be used as a test set
    assert stdnt_share < len(test_data)

    # Prepare [unlabeled] student training data (subset of test set)
    stdnt_data = test_data[:stdnt_share]
    stdnt_label = test_labels[:stdnt_share]

    stdnt_prep = PrepareData(stdnt_data, stdnt_label)

    stdnt_loader = DataLoader(stdnt_prep, batch_size=64, shuffle=False)

    # Compute teacher predictions for student training data
    teachers_preds = ensemble_preds(
        model, dataset, nb_labels, nb_teachers, stdnt_loader
    )

    # Aggregate teacher predictions to get student training labels
    stdnt_labels = noisy_max(teachers_preds, nb_labels, lap_scale)

    # Print accuracy of aggregated labels
    ac_ag_labels = accuracy(stdnt_labels, test_labels[:stdnt_share])
    print("\nAccuracy of the aggregated labels: " + str(ac_ag_labels) + "\n")

    # Store unused part of test set for use as a test set after student training
    stdnt_test_data = test_data[stdnt_share:]
    stdnt_test_labels = test_labels[stdnt_share:]

    return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(model, dataset, nb_labels, nb_teachers, stdnt_share, lap_scale):
    """This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same neural
    network architecture.

    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: True if student training went well
    """

    # Call helper function to prepare student data using teacher predictions
    stdnt_dataset = prepare_student_data(
        model, dataset, nb_labels, nb_teachers, stdnt_share, lap_scale
    )

    # Unpack the student dataset
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

    # Prepare checkpoint filename and path
    filename = str(dataset) + "_" + str(nb_teachers) + "_student.ckpt"

    stdnt_prep = PrepareData(stdnt_data, stdnt_labels)
    stdnt_loader = DataLoader(stdnt_prep, batch_size=64, shuffle=False)

    stdnt_test_prep = PrepareData(stdnt_test_data, stdnt_test_labels)
    stdnt_test_loader = DataLoader(stdnt_test_prep, batch_size=64, shuffle=False)

    # Start student training
    train(model, stdnt_loader, stdnt_test_loader, ckpt_path, filename)

    # Compute final checkpoint name for student
    student_preds = softmax_preds(
        model, nb_labels, stdnt_test_loader, ckpt_path + filename
    )

    # Compute teacher accuracy
    precision = accuracy(student_preds, stdnt_test_labels)
    print("\nPrecision of student after training: " + str(precision))

    return True


def labels_from_probs(probs):
    """Helper function: computes argmax along last dimension of array to obtain
    labels (max prob or max logit value)

    :param probs: numpy array where probabilities or logits are on last dimension
    :return: array with same shape as input besides last dimension with shape 1
            now containing the labels
    """
    # Compute last axis index
    last_axis = len(np.shape(probs)) - 1

    # Label is argmax over last dimension
    labels = np.argmax(probs, axis=last_axis)

    # Return as np.int32
    return np.asarray(labels, dtype=np.int32)


def noisy_max(logits, nb_labels, lap_scale, return_clean_votes=False):
    """This aggregation mechanism takes the softmax/logit output of several
    models resulting from inference on identical inputs and computes the noisy-
    max of the votes for candidate classes to select a label for each sample:
    it adds Laplacian noise to label counts and returns the most frequent
    label.

    :param logits: logits or probabilities for each sample
    :param lap_scale: scale of the Laplacian noise to be added to counts
    :param return_clean_votes: if set to True, also returns clean votes (without
                        Laplacian noise). This can be used to perform the
                        privacy analysis of this aggregation mechanism.
    :return: pair of result and (if clean_votes is set to True) the clean counts
             for each class per sample and the the original labels produced by
             the teachers.
    """

    # Compute labels from logits/probs and reshape array properly
    labels = labels_from_probs(logits)
    labels_shape = np.shape(labels)
    labels = labels.reshape((labels_shape[0], labels_shape[1]))

    # Initialize array to hold final labels
    result = np.zeros(int(labels_shape[1]))

    if return_clean_votes:
        # Initialize array to hold clean votes for each sample
        clean_votes = np.zeros((int(labels_shape[1]), nb_labels))

    # Parse each sample
    for i in xrange(int(labels_shape[1])):
        # Count number of votes assigned to each class
        label_counts = np.bincount(labels[:, i], minlength=10)

        if return_clean_votes:
            # Store vote counts for export
            clean_votes[i] = label_counts

        # Cast in float32 to prepare before addition of Laplacian noise
        label_counts = np.asarray(label_counts, dtype=np.float32)

        # Sample independent Laplacian noise for each class
        for item in xrange(10):
            label_counts[item] += np.random.laplace(loc=0.0, scale=float(lap_scale))

        # Result is the most frequent label
        result[i] = np.argmax(label_counts)

    # Cast labels to np.int32 for compatibility with deep_cnn.py feed dictionaries
    result = np.asarray(result, dtype=np.int32)

    if return_clean_votes:
        # Returns several array, which are later saved:
        # result: labels obtained from the noisy aggregation
        # clean_votes: the number of teacher votes assigned to each sample and class
        # labels: the labels assigned by teachers (before the noisy aggregation)
        return result, clean_votes, labels
    else:
        # Only return labels resulting from noisy aggregation
        return result


def aggregation_most_frequent(logits):
    """This aggregation mechanism takes the softmax/logit output of several
    models resulting from inference on identical inputs and computes the most
    frequent label. It is deterministic (no noise injection like noisy_max()
    above.

    :param logits: logits or probabilities for each sample
    :return:
    """
    # Compute labels from logits/probs and reshape array properly
    labels = labels_from_probs(logits)
    labels_shape = np.shape(labels)
    labels = labels.reshape((labels_shape[0], labels_shape[1]))

    # Initialize array to hold final labels
    result = np.zeros(int(labels_shape[1]))

    # Parse each sample
    for i in xrange(int(labels_shape[1])):
        # Count number of votes assigned to each class
        label_counts = np.bincount(labels[:, i], minlength=10)

        label_counts = np.asarray(label_counts, dtype=np.int32)

        # Result is the most frequent label
        result[i] = np.argmax(label_counts)

    return np.asarray(result, dtype=np.int32)


def accuracy(logits, labels):
    """Return accuracy of the array of logits (or label predictions) wrt the
    labels.

    :param logits: this can either be logits, probabilities, or a single label
    :param labels: the correct labels to match against
    :return: the accuracy as a float
    """
    assert len(logits) == len(labels)

    if len(np.shape(logits)) > 1:
        # Predicted labels are the argmax over axis 1
        predicted_labels = np.argmax(logits, axis=1)
    else:
        # Input was already labels
        assert len(np.shape(logits)) == 1
        predicted_labels = logits

    # Check against correct labels to compute correct guesses
    correct = np.sum(predicted_labels == labels.reshape(len(labels)))

    # Divide by number of labels to obtain accuracy
    accuracy = float(correct) / len(labels)

    # Return float value
    return accuracy
