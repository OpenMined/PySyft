# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Modifications copyright (C) 2020 OpenMined
#
# Added type hints to functions
# Added moment values to print statements when calculating sensitivity
# ==============================================================================


"""
This script computes bounds on the privacy cost of training the
student model from noisy aggregation of labels predicted by teachers.
It should be used only after training the student (and therefore the
teachers as well). We however include the label files required to
reproduce key results from our paper (https://arxiv.org/abs/1610.05755):
the epsilon bounds for MNIST and SVHN students.
"""
import math
from typing import List, Tuple, Union

import numpy as np
import torch


def compute_q_noisy_max(counts: Union[np.ndarray, List[float]], noise_eps: float) -> float:
    """
    Returns ~ Pr[outcome != winner].

    Args:
        counts: a list of scores
        noise_eps: privacy parameter for noisy_max
    Returns:
        q: the probability that outcome is different from true winner.
    """
    # For noisy max, we only get an upper bound.
    # Pr[ j beats i*] \leq (2+gap(j,i*))/ 4 exp(gap(j,i*)
    # proof at http://mathoverflow.net/questions/66763/
    # tight-bounds-on-probability-of-sum-of-laplace-random-variables

    winner = np.argmax(counts)
    counts_normalized = noise_eps * (counts - counts[winner])

    counts_rest = np.array([counts_normalized[i] for i in range(len(counts)) if i != winner])
    q = 0.0

    for c in counts_rest:
        gap = -c
        q += (gap + 2.0) / (4.0 * math.exp(gap))

    return min(q, 1.0 - (1.0 / len(counts)))


def compute_q_noisy_max_approx(counts: List[float], noise_eps: float) -> float:
    """
    Returns ~ Pr[outcome != winner].

    Args:
        counts: a list of scores
        noise_eps: privacy parameter for noisy_max
    Returns:
        q: the probability that outcome is different from true winner.
    """
    # For noisy max, we only get an upper bound.
    # Pr[ j beats i*] \leq (2+gap(j,i*))/ 4 exp(gap(j,i*)
    # proof at http://mathoverflow.net/questions/66763/
    # tight-bounds-on-probability-of-sum-of-laplace-random-variables
    # This code uses an approximation that is faster and easier
    # to get local sensitivity bound on.

    winner = np.argmax(counts)
    counts_normalized = noise_eps * (counts - counts[winner])
    counts_rest = np.array([counts_normalized[i] for i in range(len(counts)) if i != winner])
    gap = -max(counts_rest)
    q = (len(counts) - 1) * (gap + 2.0) / (4.0 * math.exp(gap))
    return min(q, 1.0 - (1.0 / len(counts)))


def logmgf_exact(q: float, priv_eps: float, l: int) -> float:
    """
    Computes the logmgf value given q and privacy eps.

    The bound used is the min of three terms. The first term is from
    https://arxiv.org/pdf/1605.02065.pdf.
    The second term is based on the fact that when event has probability (1-q) for
    q close to zero, q can only change by exp(eps), which corresponds to a
    much smaller multiplicative change in (1-q)
    The third term comes directly from the privacy guarantee.

    Args:
        q: pr of non-optimal outcome
        priv_eps: eps parameter for DP
        l: moment to compute.
    Returns:
        Upper bound on logmgf
    """
    if q < 0.5:
        t_one = (1 - q) * math.pow((1 - q) / (1 - math.exp(priv_eps) * q), l)
        t_two = q * math.exp(priv_eps * l)
        t = t_one + t_two
        try:
            log_t = math.log(t)
        except ValueError:
            print("Got ValueError in math.log for values :" + str((q, priv_eps, l, t)))
            log_t = priv_eps * l
    else:
        log_t = priv_eps * l

    return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)


def logmgf_from_counts(counts: Union[np.ndarray, List[float]], noise_eps: float, l: int) -> float:
    """
    ReportNoisyMax mechanism with noise_eps with 2*noise_eps-DP
    in our setting where one count can go up by one and another
    can go down by 1.

    Args:
        counts: an array of scores
        noise_eps: noise epsilon used
        l: moment to compute
    Returns:
        q: Upper bound on logmgf
    """
    q = compute_q_noisy_max(counts, noise_eps)
    return logmgf_exact(q, 2.0 * noise_eps, l)


def sens_at_k(counts: np.ndarray, noise_eps: float, l: int, k: float) -> float:
    """
    Return sensitivity at distance k.

    Args:
        counts: an array of scores
        noise_eps: noise parameter used
        l: moment whose sensitivity is being computed
        k: distance
    Returns:
        sensitivity: at distance k
    """
    counts_sorted = sorted(counts, reverse=True)

    if 0.5 * noise_eps * l > 1:
        print(f"l of {l} too large to compute sensitivity with noise epsilon {noise_eps}")
        return 0

    # Now we can assume that at k, gap remains positive
    # or we have reached the point where logmgf_exact is
    # determined by the first term and ind of q.
    if counts[0] < counts[1] + k:
        return 0

    counts_sorted[0] -= k
    counts_sorted[1] += k
    val = logmgf_from_counts(counts_sorted, noise_eps, l)

    counts_sorted[0] -= 1
    counts_sorted[1] += 1
    val_changed = logmgf_from_counts(counts_sorted, noise_eps, l)

    return val_changed - val


def smoothed_sens(counts: np.ndarray, noise_eps: float, l: int, beta: float) -> float:
    """
    Compute beta-smooth sensitivity.

    Args:
        counts: array of scores
        noise_eps: noise parameter
        l: moment of interest
        beta: smoothness parameter
    Returns:
        smooth_sensitivity: a beta smooth upper bound
    """
    k = 0
    smoothed_sensitivity = sens_at_k(counts, noise_eps, l, k)

    while k < max(counts):
        k += 1
        sensitivity_at_k = sens_at_k(counts, noise_eps, l, k)
        smoothed_sensitivity = max(smoothed_sensitivity, math.exp(-beta * k) * sensitivity_at_k)

        if sensitivity_at_k == 0.0:
            break

    return smoothed_sensitivity


def perform_analysis(
    teacher_preds: np.ndarray,
    indices: np.ndarray,
    noise_eps: float,
    delta: float = 1e-5,
    moments: int = 8,
    beta: float = 0.09,
) -> Tuple[float, float]:
    """
    Performs PATE analysis on predictions from teachers and combined predictions for student.

    Args:
        teacher_preds: a numpy array of dim (num_teachers x num_examples). Each value corresponds
            to the index of the label which a teacher gave for a specific example
        indices: a numpy array of dim (num_examples) of aggregated examples which were aggregated
            using the noisy max mechanism.
        noise_eps: the epsilon level used to create the indices
        delta: the desired level of delta
        moments: the number of moments to track (see the paper)
        beta: a smoothing parameter (see the paper)
    Returns:
        tuple: first value is the data dependent epsilon, then the data independent epsilon
    """
    num_teachers, num_examples = teacher_preds.shape
    _num_examples = indices.shape[0]
    labels = set(teacher_preds.flatten())
    num_labels = len(labels)

    assert num_examples == _num_examples

    counts_mat = np.zeros((num_examples, num_labels))

    for i in range(num_examples):
        for j in range(num_teachers):
            counts_mat[i, int(teacher_preds[j, i])] += 1

    l_list = 1.0 + np.array(range(moments))

    total_log_mgf_nm = np.array([0.0 for _ in l_list])
    total_ss_nm = np.array([0.0 for _ in l_list])

    for i in indices:
        total_log_mgf_nm += np.array(
            [logmgf_from_counts(counts_mat[i], noise_eps, l) for l in l_list]
        )

        total_ss_nm += np.array([smoothed_sens(counts_mat[i], noise_eps, l, beta) for l in l_list])

    # We want delta = exp(alpha - eps l).
    # Solving gives eps = (alpha - ln (delta))/l

    eps_list_nm = (total_log_mgf_nm - math.log(delta)) / l_list

    # If beta < eps / 2 ln (1/delta), then adding noise Lap(1) * 2 SS/eps
    # is eps,delta DP
    # Also if beta < eps / 2(gamma +1), then adding noise 2(gamma+1) SS eta / eps
    # where eta has density proportional to 1 / (1+|z|^gamma) is eps-DP
    # Both from Corolloary 2.4 in
    # http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    # Print the first one's scale

    ss_eps = 2.0 * beta * math.log(1 / delta)

    if min(eps_list_nm) == eps_list_nm[-1]:
        print(
            "Warning: May not have used enough values of l. Increase 'moments' variable and "
            "run again."
        )

    # Data independent bound, as mechanism is
    # 2*noise_eps DP.
    data_ind_log_mgf = np.array([0.0 for _ in l_list])
    data_ind_log_mgf += num_examples * np.array(
        [logmgf_exact(1.0, 2.0 * noise_eps, l) for l in l_list]
    )

    data_ind_eps_list = (data_ind_log_mgf - math.log(delta)) / l_list

    return min(eps_list_nm), min(data_ind_eps_list)


def tensors_to_literals(tensor_list: List[torch.Tensor]) -> List[Union[float, int]]:
    """
    Converts list of torch tensors to list of integers/floats. Fix for not having the functionality
    which converts list of tensors to tensors

    Args:
        tensor_list: List of torch tensors
    Returns:
        literal_list: List of floats/integers
    """
    literal_list = []

    for tensor in tensor_list:
        literal_list.append(tensor.item())

    return literal_list


def logmgf_exact_torch(q: float, priv_eps: float, l: int) -> float:
    """
    Computes the logmgf value given q and privacy eps.

    The bound used is the min of three terms. The first term is from
    https://arxiv.org/pdf/1605.02065.pdf.
    The second term is based on the fact that when event has probability (1-q) for
    q close to zero, q can only change by exp(eps), which corresponds to a
    much smaller multiplicative change in (1-q)
    The third term comes directly from the privacy guarantee.

    Args:
        q: pr of non-optimal outcome
        priv_eps: eps parameter for DP
        l: moment to compute.
    Returns:
        Upper bound on logmgf
    """
    if q < 0.5:
        t_one = (1 - q) * math.pow((1 - q) / (1 - math.exp(priv_eps) * q), l)
        t_two = q * math.exp(priv_eps * l)
        t = t_one + t_two

        try:
            log_t = math.log(t)
        except ValueError:
            print("Got ValueError in math.log for values :" + str((q, priv_eps, l, t)))
            log_t = priv_eps * l
    else:
        log_t = priv_eps * l

    return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)


def compute_q_noisy_max_torch(
    counts: Union[List[torch.Tensor], torch.Tensor], noise_eps: float
) -> float:
    """
    Returns ~ Pr[outcome != winner].

    Args:
        counts: a list of scores
        noise_eps: privacy parameter for noisy_max
    Returns:
        q: the probability that outcome is different from true winner.
    """
    if type(counts) != torch.tensor:
        counts = torch.tensor(tensors_to_literals(counts), dtype=torch.float)

    _, winner = counts.max(0)
    counts_normalized = noise_eps * (counts.clone().detach().type(torch.float) - counts[winner])

    counts_normalized = tensors_to_literals(counts_normalized)
    counts_rest = torch.tensor(
        [counts_normalized[i] for i in range(len(counts)) if i != winner], dtype=torch.float
    )
    q = 0.0

    index = 0
    for c in counts_rest:
        gap = -c
        q += (gap + 2.0) / (4.0 * math.exp(gap))

        index += 1

    return min(q, 1.0 - (1.0 / len(counts)))


def logmgf_from_counts_torch(
    counts: Union[List[torch.Tensor], torch.Tensor], noise_eps: float, l: int
) -> float:
    """
    ReportNoisyMax mechanism with noise_eps with 2*noise_eps-DP
    in our setting where one count can go up by one and another
    can go down by 1.

    Args:
        counts: a list of scores
        noise_eps: noise parameter used
        l: moment whose sensitivity is being computed
    Returns:
        q: the probability that outcome is different from true winner
    """
    q = compute_q_noisy_max_torch(counts, noise_eps)

    return logmgf_exact_torch(q, 2.0 * noise_eps, l)


def sens_at_k_torch(counts: torch.Tensor, noise_eps: float, l: int, k: int) -> float:
    """
    Return sensitivity at distane k.

    Args:
        counts: tensor of scores
        noise_eps: noise parameter used
        l: moment whose sensitivity is being computed
        k: distance
    Returns:
        sensitivity: at distance k
    """

    counts_sorted = sorted(counts, reverse=True)

    if 0.5 * noise_eps * l > 1:
        print(f"l of {l} is too large to compute sensitivity with noise epsilon {noise_eps}")
        return 0

    if counts[0] < counts[1] + k:
        return 0

    counts_sorted[0] -= k
    counts_sorted[1] += k
    val = logmgf_from_counts_torch(counts_sorted, noise_eps, l)

    counts_sorted[0] -= 1
    counts_sorted[1] += 1
    val_changed = logmgf_from_counts_torch(counts_sorted, noise_eps, l)

    return val_changed - val


def smooth_sens_torch(counts: torch.Tensor, noise_eps: float, l: int, beta: float) -> float:
    """Compute beta-smooth sensitivity.

    Args:
        counts: tensor of scores
        noise_eps: noise parameter
        l: moment of interest
        beta: smoothness parameter
    Returns:
        smooth_sensitivity: a beta smooth upper bound
    """
    k = 0
    smoothed_sensitivity = sens_at_k_torch(counts, noise_eps, l, k)

    while k < max(counts):
        k += 1
        sensitivity_at_k = sens_at_k_torch(counts, noise_eps, l, k)
        smoothed_sensitivity = max(smoothed_sensitivity, math.exp(-beta * k) * sensitivity_at_k)

        if sensitivity_at_k == 0.0:
            break

    return smoothed_sensitivity


def perform_analysis_torch(
    preds: torch.Tensor,
    indices: torch.Tensor,
    noise_eps: float = 0.1,
    delta: float = 1e-5,
    moments: int = 8,
    beta: float = 0.09,
) -> Tuple[float, float]:
    """
    Performs PATE analysis on predictions from teachers and combined predictions for student.

    Args:
        preds: a torch tensor of dim (num_teachers x num_examples). Each value corresponds to the
            index of the label which a teacher gave for a specific example
        indices: a torch tensor of dim (num_examples) of aggregated examples which were aggregated
            using the noisy max mechanism.
        noise_eps: the epsilon level used to create the indices
        delta: the desired level of delta
        moments: the number of moments to track (see the paper)
        beta: a smoothing parameter (see the paper)
    Returns:
        tuple: first value is the data dependent epsilon, then the data independent epsilon
    """
    num_teachers, num_examples = preds.shape
    _num_examples = indices.shape[0]

    # Check that preds is shape (teachers x examples)
    assert num_examples == _num_examples

    labels = list(preds.flatten())
    labels = {tensor.item() for tensor in labels}
    num_labels = len(labels)

    counts_mat = torch.zeros(num_examples, num_labels, dtype=torch.float32)

    # Count number of teacher predictions of each label for each example
    for i in range(num_examples):
        for j in range(num_teachers):
            counts_mat[i, int(preds[j, i])] += 1

    l_list = 1 + torch.tensor(range(moments), dtype=torch.float)

    total_log_mgf_nm = torch.tensor([0.0 for _ in l_list], dtype=torch.float)
    total_ss_nm = torch.tensor([0.0 for _ in l_list], dtype=torch.float)

    for i in indices:
        total_log_mgf_nm += torch.tensor(
            [logmgf_from_counts_torch(counts_mat[i].clone(), noise_eps, l) for l in l_list]
        )

        total_ss_nm += torch.tensor(
            [smooth_sens_torch(counts_mat[i].clone(), noise_eps, l, beta) for l in l_list],
            dtype=torch.float,
        )

    eps_list_nm = (total_log_mgf_nm - math.log(delta)) / l_list
    ss_eps = 2.0 * beta * math.log(1 / delta)

    if min(eps_list_nm) == eps_list_nm[-1]:
        print(
            "Warning: May not have used enough values of l. Increase 'moments' variable "
            "and run again."
        )

    # Computer epsilon when not taking teacher quorum into account
    data_ind_log_mgf = torch.tensor([0.0 for _ in l_list])
    data_ind_log_mgf += num_examples * torch.tensor(
        tensors_to_literals([logmgf_exact_torch(1.0, 2.0 * noise_eps, l) for l in l_list])
    )

    data_ind_eps_list = (data_ind_log_mgf - math.log(delta)) / l_list

    return min(eps_list_nm), min(data_ind_eps_list)
