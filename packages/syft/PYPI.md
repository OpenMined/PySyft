<div align="left"> <a href="https://pypi.org/project/syft/"><img src="https://static.pepy.tech/badge/pysyft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://hub.docker.com/u/openmined"><img src="https://img.shields.io/badge/docker-images-blue?logo=docker" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/nightlies.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/nightlies.yml/badge.svg?branch=dev" /></a> <a href="https://join.slack.com/t/openmined/shared_invite/zt-2hxwk07i9-HO7u5C7XOgou4Z62VU78zA/"><img src="https://img.shields.io/badge/chat-on%20slack-purple?logo=slack" /></a> <a href="https://docs.openmined.org/en/latest/index.html"><img src="https://img.shields.io/badge/read-docs-yellow?logo=mdbook" /></a>
<br /><br /></div>

<img alt="Syft Logo" src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/Syft-Logo.svg" width="200px" />

<h3> Data Science on data you are not allowed to see</h3>

PySyft enables a new way to do data science, where you can use non-public information, without seeing nor obtaining a copy of the data itself. All you need is to connect to a <a href="https://docs.openmined.org/en/latest/components/datasite-server.html">Datasite</a>!

Datasites are like websites, but for data. Designed with the principles of <a href="https://arxiv.org/abs/2012.08347">structured transparency</a>, they enable data owners to control how their data is protected and data scientists to use data without obtaining a copy.

PySyft supports any statistical analysis or machine learning, offering support for directly running Python code - even using third-party Python libraries.

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
pip install -U "syft[data_science]"
```

More instructions are available <a href="https://docs.openmined.org/en/latest/quick-install.html">here</a>.

## Launch Server

Launch <a href="https://docs.openmined.org/en/latest/deployment/deployment-doc-1-2-intro-req.html">a development server </a> directly in your Jupyter Notebook:

```python
import syft as sy

sy.requires(">=0.9.1,<0.9.2")

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

Datasite servers can be deployed as a single container using Docker or directly in Kubernetes. Check out our <a href="https://docs.openmined.org/en/latest/deployment/deployment-doc-1-2-intro-req.html">deployment guide.</a>

## Launch Client

Main way to use a Datasite is via our Syft client, in a Jupyter Notebook. Check out our <a href="https://docs.openmined.org/en/latest/components/syft-client.html"> PySyft client guide</a>:

```python
import syft as sy

sy.requires(">=0.9.1,<0.9.2")

datasite_client = sy.login(
    port=8080,
    email="info@openmined.org",
    password="changethis"
)
```

## PySyft - Getting started 📝

Learn about PySyft via our getting started guide:

- <a href="https://docs.openmined.org/en/latest/getting-started/introduction.html">PySyft from the ground up</a>
- <a href="https://docs.openmined.org/en/latest/getting-started/part1-dataset-and-assets.html"> Part 1: Datasets & Assets</a>
- <a href="https://docs.openmined.org/en/latest/getting-started/part2-datasite-access.html"> Part 2: Client and Datasite Access</a>
- <a href="https://docs.openmined.org/en/latest/getting-started/part3-research-study.html"> Part 3: Propose the research study</a>
- <a href="https://docs.openmined.org/en/latest/getting-started/part4-review-code-request.html"> Part 4: Review Code Requests</a>
- <a href="https://docs.openmined.org/en/latest/getting-started/part5-retrieving-results.html"> Part 5: Retrieving Results</a>

# PySyft In-depth

📚 Check out <a href="https://docs.openmined.org/en/latest/index.html">our docs website</a>.

Quick PySyft components links:

- <a href="https://docs.openmined.org/en/latest/components/datasite-server.html">DataSite Server</a>

- <a href="https://docs.openmined.org/en/latest//components/syft-client.html">Syft Client</a>

- <a href="https://docs.openmined.org/en/latest/components/datasets.html">Datasets API (`.datasets`)</a>

- <a href="https://docs.openmined.org/en/latest/components/users-api.html">Users API (`.users`)</a>

<!-- - <a href="https://docs.openmined.org/en/latest/components/projects-api.html">Projects API (`.projects`)</a> -->

- <a href="https://docs.openmined.org/en/latest/components/requests-api.html">Request API (`.requests`)</a>

- <a href="https://docs.openmined.org/en/latest/components/code-api.html">Code API (`.code`)</a>

- <a href="https://docs.openmined.org/en/latest/components/syft-policies.html">Syft Policies API (`.policy`)</a>

- <a href="https://docs.openmined.org/en/latest/components/settings-api.html">Settings API (`.settings`)</a>

- <a href="https://docs.openmined.org/en/latest/components/notifications.html">Notifications API (`.notifications`)</a>

- <a href="https://docs.openmined.org/en/latest/components/syncing-api.html">Sync API (`.sync`)</a>

## Why use PySyft?

In a variety of domains across society, data owners have **valid concerns about the risks associated with sharing their data**, such as legal risks, privacy invasion (_misuing the data_), or intellectual property (_copying and redistributing it_).

Datasites enable data scientists to **answer questions** without even seeing or acquiring a copy of the data, **within the data owners's definition of acceptable use**. We call this process <b> Remote Data Science</b>.

This means that the **current risks** of sharing information with someone will **no longer prevent** the vast benefits such as innovation, insights and scientific discovery. With each Datasite, data owners are able to enable `1000x more accesible data` in each scientific field and lead, together with data scientists, breakthrough innovation.

Learn more about our work on <a href="https://openmined.org/">our website</a>.

## Support

For questions about PySyft, reach out via `#support` on <a href="https://slack.openmined.org/">Slack</a>.

## Syft Versions

:exclamation: PySyft and Syft Server must use the same `version`.

**Latest Stable**

- `0.9.1` (Stable) - <a href="https://docs.openmined.org/en/latest/index.html">Docs</a>
- Install PySyft (Stable): `pip install -U syft`

**Latest Beta**

- `0.9.2` (Beta) - `dev` branch 👈🏽
- Install PySyft (Beta): `pip install -U syft --pre`

Find more about previous <a href="https://github.com/OpenMined/PySyft/tree/0.9.2/./releases.md">releases here</a>.

# Community

Supported by the OpenMined Foundation, the OpenMined Community is an online network of over 17,000 technologists, researchers, and industry professionals keen to _unlock 1000x more data in every scientific field and industry_.

<a href="https://join.slack.com/t/openmined/shared_invite/zt-2hxwk07i9-HO7u5C7XOgou4Z62VU78zA"><img width=150px src="https://img.shields.io/badge/Join_us-%20slack-purple?logo=slack" /></a>

# Courses

<table border="5" bordercolor="grey">
<tr>
<th align="center">
<img width="200" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/our-privacy-opportunity"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/course_privacy.png" alt="" width="100%" align="center" /></a>
</th>
<th align="center">
<img width="200" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/foundations-of-private-computation"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/course_foundations.png" alt="" width="100%" align="center" /></a>
</div>
</th>
<th align="center">
<img width="200" height="1">
<div align="center">
<a href="https://courses.openmined.org/courses/introduction-to-remote-data-science"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/course_introduction.png" alt="" width="100%" align="center"></a>
</div>
</th>
</tr>
</table>

# Contributors

OpenMined and Syft appreciates all contributors, if you would like to fix a bug or suggest a new feature, please reach out via <a href="https://github.com/OpenMined/PySyft/issues">Github</a> or <a href="https://join.slack.com/t/openmined/shared_invite/zt-2hxwk07i9-HO7u5C7XOgou4Z62VU78zA/">Slack</a>!

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/contributors_light.jpg" alt="Contributors" width="100%" />

# About OpenMined

OpenMined is a non-profit foundation creating technology infrastructure that helps researchers get answers from data without needing a copy or direct access. Our community of technologists is building Syft.

<a href="https://donate.stripe.com/fZe03H0aLdAO59e9AA
"><img width=200px src="https://img.shields.io/badge/Donate_to-OpenMined-yellow?logo=stripe" /></a>

# Supporters

<table border="0">
<tr>
<th align="center">
<a href="https://sloan.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_sloan.png" /></a>
</th>
<th align="center">
<a href="https://opensource.fb.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_meta.png" /></a>
</th>
<th align="center">
<a href="https://pytorch.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_torch.png" /></a>
</th>
<th align="center">
<a href="https://www.dpmc.govt.nz/">
<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_nz_light.png" />
</a>
</th>
<th align="center">
<a href="https://twitter.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_twitter.png" /></a>
</th>
<th align="center">
<a href="https://google.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_google.png" /></a>
</th>
<th align="center">
<a href="https://microsoft.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_microsoft.png" /></a>
</th>
<th align="center">
<a href="https://omidyar.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_on.png" /></a>
</th>
<th align="center">
<a href="https://www.udacity.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_udacity.png" /></a>
</th>
<th align="center">
<a href="https://www.centerfordigitalhealthinnovation.org/">

<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_cdhi_light.png" />

</a>
</th>
<th align="center">
<a href="https://arkhn.org/">
<img src="https://raw.githubusercontent.com/OpenMined/PySyft/0.9.2/docs/img/logo_arkhn_light.png" />
</a>
</th>
</tr>
</table>

# License

[Apache License 2.0](LICENSE)<br />
<a href="https://www.flaticon.com/free-icons/person" title="person icons">Person icons created by Freepik - Flaticon</a>

<!-- 🥇 -->
