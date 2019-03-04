import numpy as np

import torch

from syft.frameworks.torch.differential_privacy import pate


def test_base_dataset():

    num_teachers, num_examples, num_labels = (100, 50, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds
    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5
    )

    print("Original")
    print("\n")
    print(data_ind_eps)
    print(data_dep_eps)

    assert data_dep_eps < data_ind_eps


def test_base_dataset_torch():

    num_teachers, num_examples, num_labels = (100, 50, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds
    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis_torch(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5
    )

    print("\n")
    print("TORCH")
    print("\n")
    print(data_ind_eps)
    print(data_dep_eps)

    assert data_dep_eps < data_ind_eps


test_base_dataset()
test_base_dataset_torch()
