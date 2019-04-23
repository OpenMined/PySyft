import numpy as np
from syft.frameworks.torch.differential_privacy import pate


def test_base_dataset():

    num_teachers, num_examples, num_labels = (100, 50, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds
    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5
    )

    assert data_dep_eps < data_ind_eps
