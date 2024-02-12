<div align="left"> <a href="https://pypi.org/project/syft/"><img src="https://static.pepy.tech/badge/pysyft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://hub.docker.com/u/openmined"><img src="https://img.shields.io/badge/docker-images-blue?logo=docker" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/nightlies.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/nightlies.yml/badge.svg?branch=dev" /></a> <a href="https://slack.openmined.org/"><img src="https://img.shields.io/badge/chat-on%20slack-purple?logo=slack" /></a> <a href="https://openmined.github.io/PySyft/"><img src="https://img.shields.io/badge/read-docs-yellow?logo=mdbook" /></a>
<br /><br /></div>

<img alt="Syft Logo" src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/title_syft_light.png" width="200px" />

Perform data science on `data` that remains in `someone else's` server

# Quickstart

✅ `Linux` ✅ `macOS` ✅ `Windows` ✅ `Docker` ✅ `Podman` ✅ `Kubernetes`

## Install Client

```bash
$ pip install -U syft[data_science]
```

## Launch Server

```python
# from Jupyter / Python
import syft as sy
sy.requires(">=0.8.3,<0.8.4")
node = sy.orchestra.launch(name="my-domain", port=8080, dev_mode=True, reset=True)
```

```bash
# or from the command line
$ syft launch --name=my-domain --port=8080 --reset=True

Starting syft-node server on 0.0.0.0:8080
```

## Launch Client

```python
import syft as sy
sy.requires(">=0.8.3,<0.8.4")
domain_client = sy.login(port=8080, email="info@openmined.org", password="changethis")
```

## PySyft in 10 minutes

📝 <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api">API Example Notebooks</a>

- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/00-load-data.ipynb">00-load-data.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/01-submit-code.ipynb">01-submit-code.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/02-review-code-and-approve.ipynb">02-review-code-and-approve.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/03-data-scientist-download-result.ipynb">03-data-scientist-download-result.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/04-jax-example.ipynb">04-jax-example.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/05-custom-policy.ipynb">05-custom-policy.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/06-multiple-code-requests.ipynb">06-multiple-code-requests.ipynb</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8/07-domain-register-control-flow.ipynb">07-domain-register-control-flow.ipynb</a>

## Deploy Kubernetes Helm Chart

**Note**: Assuming we have a Kubernetes cluster already setup.

#### 1. Add and update Helm repo for Syft

```sh
helm repo add openmined https://openmined.github.io/PySyft/helm
helm repo update openmined
```

#### 2. Search for available Syft versions

```sh
helm search repo openmined/syft --versions --devel
```

#### 3. Set your preferred Syft Chart version

```sh
SYFT_VERSION="<paste the chart version number>"
```

#### 4. Provisioning Helm Charts

```sh
helm install my-domain openmined/syft --version $SYFT_VERSION --namespace syft --create-namespace --set ingress.className="traefik"
```

### Ingress Controllers

For Azure AKS

```sh
helm install ... --set ingress.className="azure-application-gateway"
```

For AWS EKS

```sh
helm install ... --set ingress.className="alb"
```

For Google GKE we need the [`gce` annotation](https://cloud.google.com/kubernetes-engine/docs/how-to/load-balance-ingress#create-ingress) annotation.

```sh
helm install ... --set ingress.class="gce"
```

## Deploy to a Container Engine or Cloud

1. Install our handy 🛵 cli tool which makes deploying a Domain or Gateway server to Docker or VM a one-liner:  
   `pip install -U hagrid`

2. Then run our interactive jupyter Install 🧙🏽‍♂️ Wizard<sup>BETA</sup>:  
   `hagrid quickstart`

3. In the tutorial you will learn how to install and deploy:  
   `PySyft` = our `numpy`-like 🐍 Python library for computing on `private data` in someone else's `Domain`

   `PyGrid` = our 🐳 `docker` / 🐧 `vm` `Domain` & `Gateway` Servers where `private data` lives

## Docs and Support

- 📚 <a href="https://openmined.github.io/PySyft/">Docs</a>
- `#support` on <a href="https://slack.openmined.org/">Slack</a>

# Install Notes

- HAGrid 0.3 Requires: 🐍 `python` 🐙 `git` - Run: `pip install -U hagrid`
- Interactive Install 🧙🏽‍♂️ Wizard<sup>BETA</sup> Requires 🛵 `hagrid`: - Run: `hagrid quickstart`
- PySyft 0.8.1 Requires: 🐍 `python 3.9 - 3.11` - Run: `pip install -U syft`
- PyGrid Requires: 🐳 `docker`, 🦦 `podman` or ☸️ `kubernetes` - Run: `hagrid launch ...`

# Versions

`0.9.0` - Coming soon...  
`0.8.4` (Beta) - `dev` branch 👈🏽 <a href="https://github.com/OpenMined/PySyft/tree/dev/notebooks/api/0.8">API</a> - Coming soon...  
`0.8.3` (Stable) - <a href="https://github.com/OpenMined/PySyft/tree/0.8.3/notebooks/api/0.8">API</a>

Deprecated:

- `0.8.2` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.2/notebooks/api/0.8">API</a>
- `0.8.1` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.1/notebooks/api/0.8">API</a>
- `0.8.0` - <a href="https://github.com/OpenMined/PySyft/tree/0.8/notebooks/api/0.8">API</a>
- `0.7.0` - <a href="https://github.com/OpenMined/courses/tree/introduction-to-remote-data-science-dev">Course 3 Updated</a>
- `0.6.0` - <a href="https://github.com/OpenMined/courses/tree/introduction-to-remote-data-science">Course 3</a>
- `0.5.1` - <a href="https://github.com/OpenMined/courses/tree/foundations-of-private-computation">Course 2</a> + M1 Hotfix
- `0.2.0` - `0.5.0`

PySyft and PyGrid use the same `version` and its best to match them up where possible. We release weekly betas which can be used in each context:

PySyft (Stable): `pip install -U syft`  
PyGrid (Stable) `hagrid launch ... tag=latest`

PySyft (Beta): `pip install -U syft --pre`  
PyGrid (Beta): `hagrid launch ... tag=beta`

HAGrid is a cli / deployment tool so the latest version of `hagrid` is usually the best.

# What is Syft?

<img align="right" src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_big.png" alt="Syft" height="250" style="padding-left:30px;">

`Syft` is OpenMined's `open source` stack that provides `secure` and `private` Data Science in Python. Syft decouples `private data` from model training, using techniques like [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html), [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy), and [Encrypted Computation](https://en.wikipedia.org/wiki/Homomorphic_encryption). This is done with a `numpy`-like interface and integration with `Deep Learning` frameworks, so that you as a `Data Scientist` can maintain your current workflow while using these new `privacy-enhancing techniques`.

### Why should I use Syft?

`Syft` allows a `Data Scientist` to ask `questions` about a `dataset` and, within `privacy limits` set by the `data owner`, get `answers` to those `questions`, all without obtaining a `copy` of the data itself. We call this process `Remote Data Science`. It means in a wide variety of `domains` across society, the current `risks` of sharing information (`copying` data) with someone such as, privacy invasion, IP theft and blackmail will no longer prevent the vast `benefits` such as innovation, insights and scientific discovery which secure access will provide.

No more cold calls to get `access` to a dataset. No more weeks of `wait times` to get a `result` on your `query`. It also means `1000x more data` in every domain. PySyft opens the doors to a streamlined Data Scientist `workflow`, all with the individual's `privacy` at its heart.

<!--
# Tutorials

<table border="5" bordercolor="grey">
<tr>
<th align="center">
<img width="441" height="1">
<div align="center">
<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/personas_image/dataowner.png" alt="" width="100" height="100" align="center">
<p>Data Owner</p></div>
</th>
<th align="center">
<img width="441" height="1">
<div align="center"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/personas_image/datascientist.png" alt="" width="100" height="100" align="center">
<p>Data Scientist</p></div>

</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/personas_image/dataengineer.png" alt="" width="100" height="100" align="center">
<p>Data Engineer</p>
</div>
</th>
</tr>
<tr>
<td valign="top">

- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/quickstart/data-owner/00-deploy-domain.ipynb">Deploy a Domain Server</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/quickstart/data-owner/01-upload-data.ipynb">Upload Private Data</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/quickstart/data-owner/02-create-account-configure-pb.ipynb">Create Accounts</a>
- Manage Privacy Budget</a>
- <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/quickstart/data-owner/03-join-network.ipynb">Join a Network</a>
- Learn how PETs streamline Data Policies

</td>
<td valign="top">

- Install Syft</a>
- Connect to a Domain</a>
- Search for Datasets</a>
- Train Models
- Retrieve Secure Results
- Learn Differential Privacy

</td>
<td valign="top">

- Setup Dev Mode</a>
- Deploy to Azure
- Deploy to GCP
- Deploy to Kubernetes
- Customize Networking
- Modify PyGrid UI
</td>
</tr>
</table>
-->

# Terminology

<table border="5" bordercolor="grey">
<tr>
<th align="center">
<img width="441" height="1">
<p>👨🏻‍💼 Data Owners</p>
</th>
<th align="center">
<img width="441" height="1">
<p>👩🏽‍🔬 Data Scientists</p>
</th>
</tr>
<tr>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

Provide `datasets` which they would like to make available for `study` by an `outside party` they may or may not `fully trust` has good intentions.

</td>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

Are end `users` who desire to perform `computations` or `answer` a specific `question` using one or more data owners' `datasets`.

</td>
</tr>
<tr>
<th align="center">
<img width="441" height="1">
<p>🏰 Domain Server</p>
</th>
<th align="center">
<img width="441" height="1">
<p>🔗 Gateway Server</p>
</th>
</tr>
<tr>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

Manages the `remote study` of the data by a `Data Scientist` and allows the `Data Owner` to manage the `data` and control the `privacy guarantees` of the subjects under study. It also acts as a `gatekeeper` for the `Data Scientist's` access to the data to compute and experiment with the results.

</td>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

Provides services to a group of `Data Owners` and `Data Scientists`, such as dataset `search` and bulk `project approval` (legal / technical) to participate in a project. A gateway server acts as a bridge between it's members (`Domains`) and their subscribers (`Data Scientists`) and can provide access to a collection of `domains` at once.</td>

</tr>
<tr>
</table>

# Community

<table border="5" bordercolor="grey">
<tr>
<th align="center" valign="top">
<img width="441" height="1">
<div align="center">

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/panel_slack_title_light.png" alt="" width="100%" align="center" />

<a href="https://slack.openmined.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/panel_slack.png" alt="" width="100%" align="center" /></a>

</div>
</th>
<th align="center" valign="top">
<img width="441" height="1">
<div align="center">

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/panel_title_videos_papers_light.png" alt="" width="100%" align="center" />

<p align="left"><sub><sup>
🎥 <a href="https://www.youtube.com/watch?v=qVf0tPBzr2k">PETs: Remote Data Science Unleashed - R gov 2021</a><br />
🎥 <a href="https://youtu.be/sCoDWKTbh3s?list=PL_lsbAsL_o2BQKXG7mkGFA8LSApCnhljL">Introduction to Remote Data Science - PyTorch 2021</a><br />
🎥 <a href="https://youtu.be/kzLeTz_vIeQ?list=PL_lsbAsL_o2BtOz6KUfUI_Zla6Rg5dmyc">The Future of AI Tools - PyTorch 2020</a><br />
🎥 <a href="https://www.youtube.com/watch?v=4zrU54VIK6k&t=1s">Privacy Preserving AI - MIT Deep Learning Series</a><br />
🎥 <a href="https://www.youtube.com/watch?v=Pr4erdusiW0">Privacy-Preserving Data Science - TWiML Talk #241</a><br />
🎥 <a href="https://www.youtube.com/watch?v=NJBBE_SN90A">Privacy Preserving AI - PyTorch Devcon 2019</a><br />
📖 <a href="https://arxiv.org/pdf/2110.01315.pdf">Towards general-purpose infrastructure for protect...</a><br />
📖 <a href="https://arxiv.org/pdf/2104.12385.pdf">Syft 0.5: A platform for universally deployable ...</a><br />
📖 <a href="https://arxiv.org/pdf/1811.04017.pdf">A generic framework for privacy preserving deep ...</a>
</sup></sup></p>
</div>
</th>
<th align="center" valign="top">
<img width="441" height="1">
<div align="center">

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/panel_padawan_title_light.png" alt="" width="100%" align="center" />

<a href="https://blog.openmined.org/work-on-ais-most-exciting-frontier-no-phd-required/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/panel_padawan.png" alt="" width="100%" align="center"></a>

</div>
</th>
</tr>
</table>

# Courses

<table border="5" bordercolor="grey">
<tr>
<th align="center">
<img width="441" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/our-privacy-opportunity"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/course_privacy.png" alt="" width="100%" align="center" /></a>
</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/foundations-of-private-computation"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/course_foundations.png" alt="" width="100%" align="center" /></a>
</div>
</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/introduction-to-remote-data-science"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/course_introduction.png" alt="" width="100%" align="center"></a>
</div>
</th>
</tr>
</table>

# Contributors

OpenMined and Syft appreciates all contributors, if you would like to fix a bug or suggest a new feature, please see our [guidelines](https://openmined.github.io/PySyft/developer_guide/index.html).<br />

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/contributors_light.jpg" alt="Contributors" width="100%" />

# Supporters

<table border="0">
<tr>
<th align="center">
<a href="https://sloan.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_sloan.png" /></a>
</th>
<th align="center">
<a href="https://opensource.fb.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_meta.png" /></a>
</th>
<th align="center">
<a href="https://pytorch.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_torch.png" /></a>
</th>
<th align="center">
<a href="https://www.dpmc.govt.nz/">
<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_nz_light.png" />
</a>
</th>
<th align="center">
<a href="https://twitter.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_twitter.png" /></a>
</th>
<th align="center">
<a href="https://google.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_google.png" /></a>
</th>
<th align="center">
<a href="https://microsoft.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_microsoft.png" /></a>
</th>
<th align="center">
<a href="https://omidyar.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_on.png" /></a>
</th>
<th align="center">
<a href="https://www.udacity.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_udacity.png" /></a>
</th>
<th align="center">
<a href="https://www.centerfordigitalhealthinnovation.org/">

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_cdhi_light.png" />

</a>
</th>
<th align="center">
<a href="https://arkhn.org/">
<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/logo_arkhn_light.png" />
</a>
</th>
</tr>
</table>

# Open Collective

`OpenMined` is a fiscally sponsored `501(c)(3)` in the USA. We are funded by our generous supporters on <a href="https://opencollective.com/openmined">Open Collective</a>. <br /><br />

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.8.4/docs/img/opencollective_light.png" alt="Contributors" width="100%" />

# Disclaimer

Syft is under active development and is not yet ready for pilots on private data without our assistance. As early access participants, please contact us via [Slack](https://slack.openmined.org/) or email if you would like to ask a question or have a use case that you would like to discuss.

# License

[Apache License 2.0](LICENSE)<br />
<a href="https://www.flaticon.com/free-icons/person" title="person icons">Person icons created by Freepik - Flaticon</a>

<!-- 🥇 -->
