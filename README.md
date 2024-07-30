<div align="left"> <a href="https://pypi.org/project/syft/"><img src="https://static.pepy.tech/badge/pysyft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://hub.docker.com/u/openmined"><img src="https://img.shields.io/badge/docker-images-blue?logo=docker" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/nightlies.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/nightlies.yml/badge.svg?branch=dev" /></a> <a href="https://slack.openmined.org/"><img src="https://img.shields.io/badge/chat-on%20slack-purple?logo=slack" /></a> <a href="https://docs.openmined.org/en/latest/index.html"><img src="https://img.shields.io/badge/read-docs-yellow?logo=mdbook" /></a>
<br /><br /></div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/title_syft_dark.png">
  <img alt="Syft Logo" src="docs/img/title_syft_light.png" width="200px" />
</picture>

<h3> Data Science on data you are not allowed to see</h3>

PySyft enables a new way to do data science, where you can use non-public information, without seeing nor obtaining a copy of the data itself. All you need is to connect to a <a href="https://docs.openmined.org/en/latest/components/datasite_server.html">Datasite</a>! 

Datasites are like websites, but for data. Designed with the principles of <a href="https://arxiv.org/abs/2012.08347">structured transparency</a>, it enables data owners to control how their data is protected and data scientists to use data responsibly.

PySyft supports any statistical analysis or machine learning, offering support for running directly Python code - even using third-party Python libraries.

<h4> Supported on:</h4>

✅ Linux
✅ macOS 
✅ Windows
✅ Docker
✅ Kubernetes


# Quickstart

Try out your <a href="https://docs.openmined.org/en/latest/index.html">first query against a live demo Datasite! </a>


## Install Client

```bash
$ pip install -U syft[data_science]
```

More instructions are available <a href="https://docs.openmined.org/en/latest/quick_install.ipynb">here</a>.

## Launch Server

Launch <a href="https://docs.openmined.org/en/latest/deployment/deployment-doc-1-2-intro-req.html">a development server </a> directly in your Jupyter Notebook:

```python
import syft as sy

sy.requires(">=0.9,<0.9.1")

server = sy.orchestra.launch(
    name="my-datasite",
    port=8080,
    create_producer=True,
    n_consumers=1,
    dev_mode=False,
    reset=True, # resets database
)
```

or from the command line:
```bash
$ syft launch --name=my-datasite --port=8080 --reset=True

Starting syft-datasite server on 0.0.0.0:8080
```

Datasite servers can be deployed as a single container using Docker or directly in Kubernetes. Check out our <a href="https://docs.openmined.org/en/latest/deployment/deployment-doc-1-2-intro-req.ipynb">deployment guide.</a>

## Launch Client

Main way to use a Datasite is via our Syft client, in a Jupyter Notebook. Check out our <a href="https://docs.openmined.org/en/latest/components/syft-client.ipynb"> PySyft client guide</a>:

```python
import syft as sy

sy.requires(">=0.9,<0.9.1")

datasite_client = sy.login(
    port=8080,
    email="info@openmined.org",
    password="changethis"
)
```

## PySyft - Getting started 📝 

Learn about PySyft via our getting started guide:

- <a href="https://docs.openmined.org/en/latest/getting_started/introduction.html">PySyft from the ground up</a>
- <a href="https://docs.openmined.org/en/latest/getting_started/part1_dataset_and_assets.html"> Part 1: Datasets & Assets</a>
- <a href="https://docs.openmined.org/en/latest/getting_started/part2_datasite_access.html"> Part 2: Client and Datasite Access</a>
- <a href="https://docs.openmined.org/en/latest/getting_started/part3_research_study.html"> Part 3: Propose the research study</a>
- <a href="https://docs.openmined.org/en/latest/getting_started/part4_review_code_request.html"> Part 4: Review Code Requests</a>
- <a href="https://docs.openmined.org/en/latest/getting_started/part5_retrieving_results.html"> Part 5: Retrieving Results</a>


# PySyft In-depth

📚 Check out <a href="https://docs.openmined.org/en/latest/index.html">our docs website</a>.

Quick PySyft components links:
- <a href="https://docs.openmined.org/en/latest/components/datasite_server.html">DataSite Server</a>

- <a href="https://docs.openmined.org/en/latest//components/syft_client.html">Syft Client</a>

- <a href="https://docs.openmined.org/en/latest/components/datasets.html">Datasets API (`.datasets`)</a>

- <a href="https://docs.openmined.org/en/latest/components/users-api.html">Users API (`.users`)</a>

- <a href="https://docs.openmined.org/en/latest/components/projects_api.html">Projects API (`.projects`)</a>

- <a href="https://docs.openmined.org/en/latest/components/requests_api.html">Request API (`.requests`)</a>

- <a href="https://docs.openmined.org/en/latest/components/code-api.html">Code API (`.code`)</a>

- <a href="https://docs.openmined.org/en/latest/components/syft_policies.html">Syft Policies API (`.policy`)</a>

- <a href="https://docs.openmined.org/en/latest/components/settings_api.html">Settings API (`.settings`)</a>

- <a href="https://docs.openmined.org/en/latest/components/notifications.html">Notifications API (`.notifications`)</a>

- <a href="https://docs.openmined.org/en/latest/components/syncing_api.html">Sync API (`.sync`)</a>


## Why use PySyft?

In a variety of domains across society, data owners have **valid concerns about the risks associated with sharing their data**, such as legal risks, privacy invasion (*misuing the data*), or intellectual property (*copying and redistributing it*).

Datasites enable data scientists to **answer questions** without even seeing or acquiring a copy of the data, **within the data owners's definition of acceptable use**. We call this process <b> Remote Data Science</b>.

This means that the **current risks** of sharing information with someone will **no longer prevent** the vast benefits such as innovation, insights and scientific discovery. With each Datasite, data owners are able to enable `1000x more accesible data` in each scientific field and lead, together with data scientists, breakthrough innovation.

Learn more about our work on <a href="https://openmined.org/">our website</a>.


## Support

For questions about PySyft, reach out via `#support` on <a href="https://slack.openmined.org/">Slack</a>.

<!-- 
## Deploy Kubernetes Helm Chart

#### 0. Deploy Kubernetes

Required resources: 1 CPU and 4GB RAM. However, you will need some special instructions to deploy, please consult [these instructions](https://github.com/OpenMined/PySyft/blob/dev/notebooks/tutorials/deployments/03-deploy-k8s-k3d.ipynb) or look at the resource constraint testing [here](https://github.com/OpenMined/PySyft/pull/8828#issue-2300774645).  
Recommended resources: 8+ Cores and 16GB RAM

If you're using Docker Desktop to deploy your Kubernetes, you may need to go into Settings > Resources and increase CPUs and Memory.

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
helm install my-datasite openmined/syft --version $SYFT_VERSION --namespace syft --create-namespace --set ingress.className="traefik"
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

For Google GKE we need the [`gce` annotation](https://cloud.google.com/kubernetes-engine/docs/how-to/load-balance-ingress#create-ingress).

```sh
helm install ... --set ingress.class="gce"
```

## Note:

🚨 Our old deployment tool `HAGrid` has been `deprecated`. For the updated deployment options kindly refer to:

- 📚 <a href="https://github.com/OpenMined/PySyft/tree/dev/notebooks/tutorials/deployments">Deployments</a> -->

<!-- 
# Install Notes

- PySyft 0.8.6 Requires: 🐍 `python 3.10 - 3.12` - Run: `pip install -U syft`
- Syft Server Requires: 🐳 `docker` or ☸️ `kubernetes` -->

## Syft Versions

:exclamation: PySyft and Syft Server must use the same `version`. 

### Latest Stable
`0.8.8` (Stable) - <a href="https://docs.openmined.org/en/latest/index.html">Docs</a>
- Install PySyft (Stable): `pip install -U syft`


### Latest Beta
`0.9.0` (Beta) - `dev` branch 👈🏽 
- Install PySyft (Beta): `pip install -U syft --pre`

**Deprecated**:

- `0.8.7` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.7/notebooks/api/0.8">API</a>
- `0.8.6` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.6/notebooks/api/0.8">API</a>
- `0.8.5-post.2` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.5-post.2/notebooks/api/0.8">API</a>
- `0.8.4` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.4/notebooks/api/0.8">API</a>
- `0.8.3` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.3/notebooks/api/0.8">API</a>
- `0.8.2` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.2/notebooks/api/0.8">API</a>
- `0.8.1` - <a href="https://github.com/OpenMined/PySyft/tree/0.8.1/notebooks/api/0.8">API</a>
- `0.8.0` - <a href="https://github.com/OpenMined/PySyft/tree/0.8/notebooks/api/0.8">API</a>
- `0.7.0` - <a href="https://github.com/OpenMined/courses/tree/introduction-to-remote-data-science-dev">Course 3 Updated</a>
- `0.6.0` - <a href="https://github.com/OpenMined/courses/tree/introduction-to-remote-data-science">Course 3</a>
- `0.5.1` - <a href="https://github.com/OpenMined/courses/tree/foundations-of-private-computation">Course 2</a> + M1 Hotfix
- `0.2.0` - `0.5.0`

<!-- 
# What is Syft?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo_big_dark.png">
  <img align="right" src="docs/img/logo_big.png" alt="Syft" height="250" style="padding-left:30px;">
</picture>

`Syft` is OpenMined's `open source` stack that provides `secure` and `private` Data Science in Python. Syft decouples `private data` from model training, using techniques like [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html), [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy), and [Encrypted Computation](https://en.wikipedia.org/wiki/Homomorphic_encryption). This is done with a `numpy`-like interface and integration with `Deep Learning` frameworks, so that you as a `Data Scientist` can maintain your current workflow while using these new `privacy-enhancing techniques`. -->

<!--
# Tutorials

<table border="5" bordercolor="grey">
<tr>
<th align="center">
<img width="441" height="1">
<div align="center">
<img src="docs/img/personas_image/dataowner.png" alt="" width="100" height="100" align="center">
<p>Data Owner</p></div>
</th>
<th align="center">
<img width="441" height="1">
<div align="center"><img src="docs/img/personas_image/datascientist.png" alt="" width="100" height="100" align="center">
<p>Data Scientist</p></div>

</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<img src="docs/img/personas_image/dataengineer.png" alt="" width="100" height="100" align="center">
<p>Data Engineer</p>
</div>
</th>
</tr>
<tr>
<td valign="top">

- <a href="notebooks/quickstart/data-owner/00-deploy-datasite.ipynb">Deploy a Datasite Server</a>
- <a href="notebooks/quickstart/data-owner/01-upload-data.ipynb">Upload Private Data</a>
- <a href="notebooks/quickstart/data-owner/02-create-account-configure-pb.ipynb">Create Accounts</a>
- Manage Privacy Budget</a>
- <a href="notebooks/quickstart/data-owner/03-join-network.ipynb">Join a Network</a>
- Learn how PETs streamline Data Policies

</td>
<td valign="top">

- Install Syft</a>
- Connect to a Datasite</a>
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
- Modify Syft UI
</td>
</tr>
</table>
-->
<!-- 
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

Provide `datasets` which they would like to make available for `study` by an `outside party` they may or may not `fully trust` has good intentions.

</td>
<td valign="top">

Are end `users` who desire to perform `computations` or `answer` a specific `question` using one or more data owners' `datasets`.

</td>
</tr>
<tr>
<th align="center">
<img width="441" height="1">
<p>🏰 Datasite Server</p>
</th>
<th align="center">
<img width="441" height="1">
<p>🔗 Gateway Server</p>
</th>
</tr>
<tr>
<td valign="top">

Manages the `remote study` of the data by a `Data Scientist` and allows the `Data Owner` to manage the `data` and control the `privacy guarantees` of the subjects under study. It also acts as a `gatekeeper` for the `Data Scientist's` access to the data to compute and experiment with the results.

</td>
<td valign="top">

Provides services to a group of `Data Owners` and `Data Scientists`, such as dataset `search` and bulk `project approval` (legal / technical) to participate in a project. A gateway server acts as a bridge between it's members (`Datasites`) and their subscribers (`Data Scientists`) and can provide access to a collection of `datasites` at once.</td>

</tr>
<tr>
</table> -->

# Community

OpenMined is a growing community of over 15000 engineers, educators, scientists, product managers and privacy-aware individuals. 

<table border="5" bordercolor="grey">
<tr>
<th align="center" valign="top">
<img width="441" height="1">
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/panel_slack_title_dark.png">
  <img src="docs/img/panel_slack_title_light.png" alt="" width="100%" align="center" />
</picture>

<a href="https://slack.openmined.org/"><img src="docs/img/panel_slack.png" alt="" width="100%" align="center" /></a>

</div>
</th>
<th align="center" valign="top">
<img width="441" height="1">
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/panel_title_videos_papers.png">
  <img src="docs/img/panel_title_videos_papers_light.png" alt="" width="100%" align="center" />
</picture>

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/panel_padawan_title_dark.png">
  <img src="docs/img/panel_padawan_title_light.png" alt="" width="100%" align="center" />
</picture>

<a href="https://blog.openmined.org/work-on-ais-most-exciting-frontier-no-phd-required/"><img src="docs/img/panel_padawan.png" alt="" width="100%" align="center"></a>

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
<a href="https://courses.openmined.org/courses/our-privacy-opportunity"><img src="docs/img/course_privacy.png" alt="" width="100%" align="center" /></a>
</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/foundations-of-private-computation"><img src="docs/img/course_foundations.png" alt="" width="100%" align="center" /></a>
</div>
</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/introduction-to-remote-data-science"><img src="docs/img/course_introduction.png" alt="" width="100%" align="center"></a>
</div>
</th>
</tr>
</table>

# Contributors

OpenMined and Syft appreciates all contributors, if you would like to fix a bug or suggest a new feature, please reach out via Github or Slack!

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/contributors_dark.jpg">
  <img src="docs/img/contributors_light.jpg" alt="Contributors" width="100%" />
</picture>

# Supporters

<table border="0">
<tr>
<th align="center">
<a href="https://sloan.org/"><img src="docs/img/logo_sloan.png" /></a>
</th>
<th align="center">
<a href="https://opensource.fb.com/"><img src="docs/img/logo_meta.png" /></a>
</th>
<th align="center">
<a href="https://pytorch.org/"><img src="docs/img/logo_torch.png" /></a>
</th>
<th align="center">
<a href="https://www.dpmc.govt.nz/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo_nz_dark.png">
  <img src="docs/img/logo_nz_light.png" />
</picture>
</a>
</th>
<th align="center">
<a href="https://twitter.com/"><img src="docs/img/logo_twitter.png" /></a>
</th>
<th align="center">
<a href="https://google.com/"><img src="docs/img/logo_google.png" /></a>
</th>
<th align="center">
<a href="https://microsoft.com/"><img src="docs/img/logo_microsoft.png" /></a>
</th>
<th align="center">
<a href="https://omidyar.com/"><img src="docs/img/logo_on.png" /></a>
</th>
<th align="center">
<a href="https://www.udacity.com/"><img src="docs/img/logo_udacity.png" /></a>
</th>
<th align="center">
<a href="https://www.centerfordigitalhealthinnovation.org/">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo_cdhi_dark.png">
  <img src="docs/img/logo_cdhi_light.png" />
</picture>

</a>
</th>
<th align="center">
<a href="https://arkhn.org/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo_arkhn.png">
  <img src="docs/img/logo_arkhn_light.png" />
</picture>
</a>
</th>
</tr>
</table>


# License

[Apache License 2.0](LICENSE)<br />
<a href="https://www.flaticon.com/free-icons/person" title="person icons">Person icons created by Freepik - Flaticon</a>

<!-- 🥇 -->
