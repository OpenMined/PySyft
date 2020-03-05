#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
from torchvision import datasets, transforms


def _get_norm_mnist(dir, reduced=None, binary=False):
    """Downloads and normalizes mnist"""
    mnist_train = datasets.MNIST(dir, download=True, train=True)
    mnist_test = datasets.MNIST(dir, download=True, train=False)

    # compute normalization factors
    data_all = torch.cat([mnist_train.data, mnist_test.data]).float()
    data_mean, data_std = data_all.mean(), data_all.std()
    tensor_mean, tensor_std = data_mean.unsqueeze(0), data_std.unsqueeze(0)

    # normalize
    mnist_train_norm = transforms.functional.normalize(
        mnist_train.data.float(), tensor_mean, tensor_std
    )
    mnist_test_norm = transforms.functional.normalize(
        mnist_test.data.float(), tensor_mean, tensor_std
    )

    # change all nonzero labels to 1 if binary classification required
    if binary:
        mnist_train.targets[mnist_train.targets != 0] = 1
        mnist_test.targets[mnist_test.targets != 0] = 1

    # create a reduced dataset if required
    if reduced is not None:
        mnist_norm = (mnist_train_norm[:reduced], mnist_test_norm[:reduced])
        mnist_labels = (mnist_train.targets[:reduced], mnist_test.targets[:reduced])
    else:
        mnist_norm = (mnist_train_norm, mnist_test_norm)
        mnist_labels = (mnist_train.targets, mnist_test.targets)
    return mnist_norm, mnist_labels


def split_features(
    split=0.5, dir="/tmp", party1="alice", party2="bob", reduced=None, binary=False
):
    """Splits features between Party 1 and Party 2"""
    mnist_norm, mnist_labels = _get_norm_mnist(dir, reduced, binary)
    mnist_train_norm, mnist_test_norm = mnist_norm
    mnist_train_labels, mnist_test_labels = mnist_labels

    num_features = mnist_train_norm.shape[1]
    split_point = int(split * num_features)

    party1_train = mnist_train_norm[:, :, :split_point]
    party2_train = mnist_train_norm[:, :, split_point:]
    party1_test = mnist_test_norm[:, :, :split_point]
    party2_test = mnist_test_norm[:, :, split_point:]

    torch.save(party1_train, os.path.join(dir, party1 + "_train.pth"))
    torch.save(party2_train, os.path.join(dir, party2 + "_train.pth"))
    torch.save(party1_test, os.path.join(dir, party1 + "_test.pth"))
    torch.save(party2_test, os.path.join(dir, party2 + "_test.pth"))
    torch.save(mnist_train_labels, os.path.join(dir, "train_labels.pth"))
    torch.save(mnist_test_labels, os.path.join(dir, "test_labels.pth"))


def split_observations(
    split=0.5, dir="/tmp", party1="alice", party2="bob", reduced=None, binary=False
):
    """Splits observations between Party 1 and Party 2"""
    mnist_norm, mnist_labels = _get_norm_mnist(dir, reduced, binary)
    mnist_train_norm, mnist_test_norm = mnist_norm
    mnist_train_labels, mnist_test_labels = mnist_labels

    num_train_obs = mnist_train_norm.shape[0]
    obs_train_split = int(split * num_train_obs)
    num_test_obs = mnist_test_norm.shape[0]
    obs_test_split = int(split * num_test_obs)

    party1_train = mnist_train_norm[:obs_train_split, :, :]
    party2_train = mnist_train_norm[obs_train_split:, :, :]
    party1_test = mnist_test_norm[:obs_test_split, :, :]
    party2_test = mnist_test_norm[obs_test_split:, :, :]
    torch.save(party1_train, os.path.join(dir, party1 + "_train.pth"))
    torch.save(party2_train, os.path.join(dir, party2 + "_train.pth"))
    torch.save(party1_test, os.path.join(dir, party1 + "_test.pth"))
    torch.save(party2_test, os.path.join(dir, party2 + "_test.pth"))

    party1_train_labels = mnist_train_labels[:obs_train_split]
    party1_test_labels = mnist_test_labels[:obs_test_split]
    party2_train_labels = mnist_train_labels[obs_train_split:]
    party2_test_labels = mnist_test_labels[obs_test_split:]

    torch.save(party1_train_labels, os.path.join(dir, party1 + "_train_labels.pth"))
    torch.save(party1_test_labels, os.path.join(dir, party1 + "_test_labels.pth"))
    torch.save(party2_train_labels, os.path.join(dir, party2 + "_train_labels.pth"))
    torch.save(party2_test_labels, os.path.join(dir, party2 + "_test_labels.pth"))


def split_features_v_labels(
    dir="/tmp", party1="alice", party2="bob", reduced=None, binary=False
):
    """Gives Party 1 features and Party 2 labels"""
    mnist_norm, mnist_labels = _get_norm_mnist(dir, reduced, binary)
    mnist_train_norm, mnist_test_norm = mnist_norm
    mnist_train_labels, mnist_test_labels = mnist_labels

    torch.save(mnist_train_norm, os.path.join(dir, party1 + "_train.pth"))
    torch.save(mnist_test_norm, os.path.join(dir, party1 + "_test.pth"))
    torch.save(mnist_train_labels, os.path.join(dir, party2 + "_train_labels.pth"))
    torch.save(mnist_test_labels, os.path.join(dir, party2 + "_test_labels.pth"))


def split_train_v_test(
    dir="/tmp", party1="alice", party2="bob", reduced=None, binary=False
):
    """Gives Party 1 training data and Party 2 the test data """
    mnist_norm, mnist_labels = _get_norm_mnist(dir, reduced, binary)
    mnist_train_norm, mnist_test_norm = mnist_norm
    mnist_train_labels, mnist_test_labels = mnist_labels

    torch.save(mnist_train_norm, os.path.join(dir, party1 + "_train.pth"))
    torch.save(mnist_test_norm, os.path.join(dir, party2 + "_test.pth"))
    torch.save(mnist_train_labels, os.path.join(dir, party1 + "_train_labels.pth"))
    torch.save(mnist_test_labels, os.path.join(dir, party2 + "_test_labels.pth"))


def main():
    parser = argparse.ArgumentParser("Split data for use in Tutorials")
    parser.add_argument(
        "--option",
        type=str,
        choices={"features", "data", "features_v_labels", "train_v_test"},
    )
    parser.add_argument("--ratio", type=float, default=0.72)
    parser.add_argument("--name_party1", type=str, default="alice")
    parser.add_argument("--name_party2", type=str, default="bob")
    parser.add_argument("--dest", type=str, default="/tmp")
    parser.add_argument("--reduced", type=int, default=None)
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()

    if args.option == "features":
        split_features(
            split=args.ratio,
            dir=args.dest,
            party1=args.name_party1,
            party2=args.name_party2,
            reduced=args.reduced,
            binary=args.binary,
        )
    elif args.option == "data":
        split_observations(
            split=args.ratio,
            dir=args.dest,
            party1=args.name_party1,
            party2=args.name_party2,
            reduced=args.reduced,
            binary=args.binary,
        )
    elif args.option == "features_v_labels":
        split_features_v_labels(
            dir=args.dest,
            party1=args.name_party1,
            party2=args.name_party2,
            reduced=args.reduced,
            binary=args.binary,
        )
    elif args.option == "train_v_test":
        split_train_v_test(
            dir=args.dest,
            party1=args.name_party1,
            party2=args.name_party2,
            reduced=args.reduced,
            binary=args.binary,
        )
    else:
        raise ValueError("Invalid split option")


if __name__ == "__main__":
    main()
