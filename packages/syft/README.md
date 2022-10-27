<h1 align="center">

  <br>
  <a href="https://openmined.github.io/PySyft/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/logo_big.png" alt="PySyft" width="200"></a>
  <br>
  A library for computing on data<br /> you do not own and cannot see
  <br>

</h1>

<div align="left"> <a href="https://pypi.org/project/syft/"><img src="https://pepy.tech/badge/syft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://hub.docker.com/u/openmined"><img src="https://img.shields.io/badge/docker-images-blue?logo=docker" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/nightlies-run.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/nightlies-run.yml/badge.svg?branch=dev" /></a> <a href="https://gitpod.io/#https://github.com/OpenMined/PySyft"><img src="https://img.shields.io/badge/gitpod-908a85?logo=gitpod" /></a>
<a href="https://slack.openmined.org/"><img src="https://img.shields.io/badge/chat-on%20slack-purple?logo=slack" /></a> <a href="https://openmined.github.io/PySyft/"><img src="https://img.shields.io/badge/read-docs-yellow?logo=mdbook" /></a>
<br /><br /></div>
<br /><br />

<div align="center"><a href="#"><img src="https://stars.medv.io/openmined/pysyft.svg" /></a></div>

</div>

# PySyft is a Python library for secure and private Deep Learning.

PySyft decouples private data from model training, using
[Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
[Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy),
and Encrypted Computation (like
[Multi-Party Computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
and [Homomorphic Encryption (HE)](https://en.wikipedia.org/wiki/Homomorphic_encryption))
within the main Deep Learning frameworks like PyTorch and TensorFlow. Join the movement on
[Slack](http://slack.openmined.org/).

---

Most software libraries let you compute over the information you own and see inside of machines you control. However, this means that you cannot compute on information without first obtaining (at least partial) ownership of that information. It also means that you cannot compute using machines without first obtaining control over those machines. This is very limiting to human collaboration and systematically drives the centralization of data, because you cannot work with a bunch of data without first putting it all in one (central) place.

The Syft ecosystem seeks to change this system, allowing you to write software which can compute over information you do not own on machines you do not have (total) control over. This not only includes servers in the cloud, but also personal desktops, laptops, mobile phones, websites, and edge devices. Wherever your data wants to live in your ownership, the Syft ecosystem exists to help keep it there while allowing it to be used privately for computation.

## 0.6.0 Release

The current stable release is `0.6.0`


## Installation

```
$ pip install -U syft --pre
```


## Documentation
For full documentation, including installation and tutorials, please see https://openmined.github.io/PySyft/



## Contributing

The guide for contributors can be found [here](https://github.com/OpenMined/PySyft/blob/main/packages/syft/CONTRIBUTING.md).
It covers all that you need to know to start contributing code to PySyft today.

Also, join the rapidly growing community of 12,000+ on [Slack](http://slack.openmined.org).
The Slack community is very friendly and great about quickly answering questions about the use and development of PySyft!




# Supporters

<table border="0">
<tr>
<th align="center">
<a href="https://sloan.org/"><img src="packages/syft/docs/img/logo_sloan.png" /></a>
</th>
<th align="center">
<a href="https://opensource.fb.com/"><img src="packages/syft/docs/img/logo_meta.png" /></a>
</th>
<th align="center">
<a href="https://pytorch.org/"><img src="packages/syft/docs/img/logo_torch.png" /></a>
</th>
<th align="center">
<a href="https://www.udacity.com/"><img src="packages/syft/docs/img/logo_udacity.png" /></a>
</th>
<th align="center">
<a href="https://summerofcode.withgoogle.com/"><img src="packages/syft/docs/img/logo_gsoc.png" /></a>
</th>
<th align="center">
<a href="https://developers.google.com/season-of-docs"><img src="packages/syft/docs/img/logo_gsod.png" /></a>
</th>
<th align="center">
<img src="packages/syft/docs/img/logo_arkhn_light.png#gh-light-mode-only" />
<img src="packages/syft/docs/img/logo_arkhn.png#gh-dark-mode-only" />
</th>
<th align="center">
<img src="packages/syft/docs/img/logo_cape_light.png#gh-light-mode-only" />
<img src="packages/syft/docs/img/logo_cape.png#gh-dark-mode-only" />
</th>
<th align="center">
<a href="https://begin.ai/"><img src="packages/syft/docs/img/logo_begin.png" /></a>
</th>
</tr>
</table>

# Open Collective

`OpenMined` is a registered `501(c)(3)` in the USA. We are funded by our gracious supporters on <a href="https://opencollective.com/openmined">Open Collective</a>. <br /><br />
<img src="packages/syft/docs/img/opencollective_light.png#gh-light-mode-only" alt="Contributors" width="100%" />
<img src="packages/syft/docs/img/opencollective_dark.png#gh-dark-mode-only" alt="Contributors" width="100%" />


# Disclaimer

Syft is under active development and is not yet ready for pilots on private data without our assistance. As early access participants, please contact us via [Slack](https://slack.openmined.org/) or email if you would like to ask a question or have a use case that you would like to discuss.

# License

[Apache License 2.0](LICENSE)<br />
<a href="https://www.flaticon.com/free-icons/person" title="person icons">Person icons created by Freepik - Flaticon</a>
