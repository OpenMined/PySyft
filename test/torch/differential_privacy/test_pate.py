import numpy as np

import torch

from syft.frameworks.torch.dp import pate

np.random.seed(0)


def test_base_dataset():

    num_teachers, num_examples, num_labels = (100, 50, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds

    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5
    )

    assert data_dep_eps < data_ind_eps


def test_base_dataset_torch():

    num_teachers, num_examples, num_labels = (100, 50, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds

    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis_torch(
        preds, indices, noise_eps=0.1, delta=1e-5
    )

    assert data_dep_eps < data_ind_eps


def test_torch_ref_match():

    # Verify if the torch implementation values match the original Numpy implementation.

    num_teachers, num_examples, num_labels = (100, 50, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds

    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis_torch(
        preds, indices, noise_eps=0.1, delta=1e-5
    )

    data_dep_eps_ref, data_ind_eps_ref = pate.perform_analysis(
        preds, indices, noise_eps=0.1, delta=1e-5
    )

    assert torch.isclose(data_dep_eps, torch.tensor(data_dep_eps_ref, dtype=torch.float32))
    assert torch.isclose(data_ind_eps, torch.tensor(data_ind_eps_ref, dtype=torch.float32))
