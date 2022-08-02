<div align="left"> <a href="https://pypi.org/project/syft/"><img src="https://pepy.tech/badge/syft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://hub.docker.com/u/openmined"><img src="https://img.shields.io/badge/docker-images-blue?logo=docker" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/syft-version_tests.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/syft-version_tests.yml/badge.svg?branch=dev" /></a>
 <a href="https://github.com/OpenMined/PySyft/actions/workflows/nightlies-run.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/nightlies-run.yml/badge.svg?branch=dev" /></a>
<a href="https://slack.openmined.org/"><img src="https://img.shields.io/badge/chat-on%20slack-purple?logo=slack" /></a> <a href="https://openmined.github.io/PySyft/"><img src="https://img.shields.io/badge/read-docs-yellow?logo=mdbook" /></a>
<br /><br /></div>

<img src="packages/syft/docs/img/title_syft_light.png#gh-light-mode-only" alt="Syft Logo" width="200px" />
<img src="packages/syft/docs/img/title_syft_dark.png#gh-dark-mode-only" alt="Syft Logo" width="200px" />

Remote Data Science - Code for `computing on data`, you `do not own` and `cannot see`

<div align="left">
<img src="packages/syft/docs/img/header.png#gh-light-mode-only" alt="Syft Overview" width="100%" />
<img src="packages/syft/docs/img/header.png#gh-dark-mode-only" alt="Syft Overview" width="100%" />
</div>

<br />

# Quickstart

✅ `Linux` ✅ `macOS` ✅ `Windows`†‡
<img src="packages/syft/docs/img/terminalizer.gif" height="400" align="right" />

1. Install our handy python cli tool:  
   🛵 `pip install hagrid`
2. Then run our interactive jupyter quickstart tutorial:  
   💻 `hagrid quickstart`

- In the tutorial you will learn how to install and deploy:  
  `PySyft` = our `torch`-like 🐍 Python Library  
  `PyGrid` = our 🐳 `docker` / `k8s` Data Platform

- During quickstart we will deploy `PyGrid` to localhost with 🐳 `docker`, however 🛵 HAGrid can deploy to `k8s` or a 🐧 `ubuntu` VM on `azure` / `gcp` / `ANY_IP_ADDRESS` by using 🔨 `ansible`†

3. Read our 📚 <a href="https://openmined.github.io/PySyft/">Docs</a>
4. Ask Questions ❔ in `#support` on <a href="https://slack.openmined.org/">Slack</a>

# Install Notes

- HAGrid Requires: 🐍 `python` 🐙 `git` - Run: `pip install hagrid`  
  †`ansible` is not supported on `Windows` preventing some remote deployment targets
- PySyft Requires: 🐍 `python 3.7+` - Run: `pip install syft`  
  ‡`Windows` users must run this first: `pip install jaxlib===0.3.7 -f https://whls.blob.core.windows.net/unstable/index.html`
- PyGrid Requires: 🐳 `docker` / `k8s` or 🐧 `ubuntu` VM - Run: `hagrid launch ...`

# Versions

`0.8.0 beta` - `dev` branch 👈🏽  
`0.7.0` - Stable  
`0.6.0` - <a href="https://github.com/OpenMined/courses/tree/introduction-to-remote-data-science">Course 3</a>  
`0.5.1` - <a href="https://github.com/OpenMined/courses/tree/foundations-of-private-computation">Course 2</a> + M1 Hotfix  
`0.2.0` - `0.5.0` Deprecated

PySyft and PyGrid use the same `version` and its best to match them up where possible. We release weekly betas which can be used in each context:
PySyft: `pip install syft --pre`
PyGrid: `hagrid launch ... tag=latest`
Quickstart: `hagrid quickstart --pre`

HAGrid is a cli / deployment tool so the latest version of `hagrid` is usually the best.

# What is Syft?

<img align="right" src="packages/syft/docs/img/logo_big.png#gh-light-mode-only" alt="Syft" height="250" style="padding-left:30px;">

<img align="right" src="packages/syft/docs/img/logo_big_dark.png#gh-dark-mode-only" alt="Syft" height="250" style="padding-left:30px;">

`Syft` is OpenMined's `open source` stack that provides `secure` and `private` Data Science in Python. Syft decouples `private data` from model training, using techniques like [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html), [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy), and [Encrypted Computation](https://en.wikipedia.org/wiki/Homomorphic_encryption). This is done with a `torch`-like interface and integration with `Deep Learning` frameworks so that you as a `Data Scientist` can maintain your current workflow while using these new `privacy-enhancing techniques`.

### Why should I use Syft?

`Syft` allows a `Data Scientist` to ask `questions` about a `dataset` and, within `privacy limits` set by the `data owner`, get `answers` to those `questions`, all without obtaining a `copy` of the data itself. We call this process `Remote Data Science`. It means in a wide variety of `domains` across society, the current `risks` of sharing information (`copying` data) with someone such as, privacy invasion, IP theft and blackmail will no longer prevent the ability to utilize the vast `benefits` such as innovation, insights and scientific discovery.

No more cold calls to get `access` to a dataset. No more weeks of `wait times` to get a `result` on your `query`. It also means `1000x more data` in every domain. PySyft opens the doors to a streamlined Data Scientist `workflow`, all with the individual's `privacy` at its heart.

# Tutorials

<table border="5" bordercolor="grey">
<tr>
<th align="center">
<img width="441" height="1">
<div align="center">
<img src="packages/syft/docs/img/personas_image/dataowner.png" alt="" width="100" height="100" align="center">
<p>Data Owner</p></div>
</th>
<th align="center">
<img width="441" height="1">
<div align="center"><img src="packages/syft/docs/img/personas_image/datascientist.png" alt="" width="100" height="100" align="center">
<p>Data Scientist</p></div>

</th>
<th align="center">
<img width="441" height="1">
<div align="center">
<img src="packages/syft/docs/img/personas_image/dataengineer.png" alt="" width="100" height="100" align="center">
<p>Data Engineer</p>
</div>
</th>
</tr>
<tr>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

- <a href="#">Deploy a Domain Server</a>
- <a href="#">Upload Private Data</a>
- <a href="#">Create Accounts</a>
- <a href="#">Manage Privacy Budget</a>
- <a href="#">Join a Network</a>
- Learn how PETs streamline Data Policies

</td>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

- <a href="#">Install Syft</a>
- <a href="#">Connect to a Domain</a>
- <a href="#">Search for Datasets</a>
- Train Models
- Retrieve Secure Results
- Learn Differential Privacy

</td>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

- <a href="#">Setup Dev Mode</a>
- Deploy to Azure
- Deploy to GCP
- Deploy to Kubernetes
- Customize Networking
- Modify PyGrid UI
</td>
</tr>
</table>

# Important Terms

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

End `users` who desire to perform `computations` or `answer` a specific `question` using one or more data owners' `datasets`.

</td>
</tr>
<tr>
<th align="center">
<img width="441" height="1">
<p>🏰 Domain Server</p>
</th>
<th align="center">
<img width="441" height="1">
<p>🔗 Network Server</p>
</th>
</tr>
<tr>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

Manages the `remote study` of the data by a `Data Scientist` and allows the `Data Owner` to manage the `data` and control the `privacy guarantees` of the subjects under study. It also acts as a `gatekeeper` for the `Data Scientist's` access to the data to compute and experiment with the results.

</td>
<td valign="top">
<!-- REMOVE THE BACKSLASHES -->

Provides services to a group of `Data Owners` and `Data Scientists`, such as dataset `search` and bulk `project approval` (legal / technical) to participate in a project. A network server acts as a bridge between it's members (`Domains`) and their subscribers (`Data Scientists`) and can provide access to a collection of `domains` at once.</td>

</tr>
<tr>
</table>

The steps performed by the respective personas are shown below:

<div>
    <img src="packages/syft/docs/img/big-picture.png#gh-light-mode-only" alt="big-picture-overview" width="100%">
    <img src="packages/syft/docs/img/big-picture-dark.png#gh-dark-mode-only" alt="big-picture-overview" width="100%">
</div>

# Community

`Openmined` is a vibrant group of `developers`, `data scientists`, `researchers`, and `decision-makers`. If you want to be a part of OpenMined's `thriving community` and would like to dive deep into the `concepts` of `Remote Data Science`, we offer you to share your `thoughts` with us on our [slack](https://communityinviter.com/apps/openmined/openmined/) channel and follow the below `study materials` to get familiar with the `PySyft` library.

<table border="5" bordercolor="grey">
  <thead>
    <tr>
      <th>Join Slack (14,500+)</th>
      <th>OpenMined Courses</th>
      <th>Padawan Program</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href='https://communityinviter.com/apps/openmined/openmined/'><img src="packages/syft/docs/img/personas_image/slack.jpeg" height=150 width=220 /></a></td>
      <td><a href='https://courses.openmined.org/'><img src="packages/syft/docs/img/personas_image/course.png" height=150 width=220 /></a></td>
      </td>
      <td><a href='https://blog.openmined.org/work-on-ais-most-exciting-frontier-no-phd-required/'><img src="http://img.youtube.com/vi/SWekBc0wnxY/maxresdefault.jpg" title="Padawan" alt="YouTube Video" height=150 width=220 /></a></td>
    </tr>
  </tbody>
</table>

# Call for Contributors

OpenMined and Syft appreciates all contributors, and if you would like to fix a bug or suggest a new feature, please see our [Contribution guidelines](https://openmined.github.io/PySyft/developer_guide/index.html).

If you are still looking for some help in understanding Syft, learn more about the Syft library using this [resource](https://openmined.github.io/PySyft/resources/index.html).

# Disclaimer

Syft is under active development and is not yet ready for total pilots on private data without our assistance. As early access participants, please contact us via [Slack](https://communityinviter.com/apps/openmined/openmined/) or email if you would like to ask a question or have a use case that you would like to propose.

# Organisational Contributors

Syft exists because of all the great people who contributed to this project. We are very grateful for contributions to Syft and Grid from the following organizations!

  <br>
  <img src="packages/syft/docs/img/Organizational_Contributions.gif" alt="Syft" width="400">
  </br>

# License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/main/packages/syft/LICENSE)<br />
<a href="https://www.flaticon.com/free-icons/person" title="person icons">Person icons created by Freepik - Flaticon</a>
