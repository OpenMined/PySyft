# syft.frameworks.torch.differential_privacy package

## Submodules

## syft.frameworks.torch.differential_privacy.pate module

This script computes bounds on the privacy cost of training the
student model from noisy aggregation of labels predicted by teachers.
It should be used only after training the student (and therefore the
teachers as well). We however include the label files required to
reproduce key results from our paper ([https://arxiv.org/abs/1610.05755](https://arxiv.org/abs/1610.05755)):
the epsilon bounds for MNIST and SVHN students.


#### syft.frameworks.torch.differential_privacy.pate.compute_q_noisy_max(counts, noise_eps)
returns ~ Pr[outcome != winner].


* **Parameters**

    * **counts** – a list of scores

    * **noise_eps** – privacy parameter for noisy_max



* **Returns**

    the probability that outcome is different from true winner.



* **Return type**

    q



#### syft.frameworks.torch.differential_privacy.pate.compute_q_noisy_max_approx(counts, noise_eps)
returns ~ Pr[outcome != winner].


* **Parameters**

    * **counts** – a list of scores

    * **noise_eps** – privacy parameter for noisy_max



* **Returns**

    the probability that outcome is different from true winner.



* **Return type**

    q



#### syft.frameworks.torch.differential_privacy.pate.logmgf_exact(q, priv_eps, l)
Computes the logmgf value given q and privacy eps.

The bound used is the min of three terms. The first term is from
[https://arxiv.org/pdf/1605.02065.pdf](https://arxiv.org/pdf/1605.02065.pdf).
The second term is based on the fact that when event has probability (1-q) for
q close to zero, q can only change by exp(eps), which corresponds to a
much smaller multiplicative change in (1-q)
The third term comes directly from the privacy guarantee.
:param q: pr of non-optimal outcome
:param priv_eps: eps parameter for DP
:param l: moment to compute.


* **Returns**

    Upper bound on logmgf



#### syft.frameworks.torch.differential_privacy.pate.logmgf_from_counts(counts, noise_eps, l)
ReportNoisyMax mechanism with noise_eps with 2\*noise_eps-DP
in our setting where one count can go up by one and another
can go down by 1.


#### syft.frameworks.torch.differential_privacy.pate.perform_analysis(teacher_preds, indices, noise_eps, delta=1e-05, moments=8, beta=0.09)
“Performs PATE analysis on predictions from teachers and combined predictions for student.


* **Parameters**

    * **teacher_preds** – a numpy array of dim (num_teachers x num_examples). Each value corresponds to the
      index of the label which a teacher gave for a specific example

    * **indices** – a numpy array of dim (num_examples) of aggregated examples which were aggregated using
      the noisy max mechanism.

    * **noise_eps** – the epsilon level used to create the indices

    * **delta** – the desired level of delta

    * **moments** – the number of moments to track (see the paper)

    * **beta** – a smoothing parameter (see the paper)



* **Returns**

    first value is the data dependent epsilon, then the data independent epsilon



* **Return type**

    tuple



#### syft.frameworks.torch.differential_privacy.pate.sens_at_k(counts, noise_eps, l, k)
Return sensitivity at distane k.


* **Parameters**

    * **counts** – an array of scores

    * **noise_eps** – noise parameter used

    * **l** – moment whose sensitivity is being computed

    * **k** – distance



* **Returns**

    at distance k



* **Return type**

    sensitivity



#### syft.frameworks.torch.differential_privacy.pate.smoothed_sens(counts, noise_eps, l, beta)
Compute beta-smooth sensitivity.


* **Parameters**

    * **counts** – array of scors

    * **noise_eps** – noise parameter

    * **l** – moment of interest

    * **beta** – smoothness parameter



* **Returns**

    a beta smooth upper bound



* **Return type**

    smooth_sensitivity


## Module contents
