:notoc:

.. PySyft documentation master file, created by
   sphinx-quickstart on Sun Oct  3 23:51:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: syft

PySyft's documentation
======================

:mod:`PySyft` is an open source library that provides secure and private Deep
Learning in Python.

Latest release
--------------

Latest stable release can be found on:

- `PyPI <https://pypi.org/project/syft/>`_
- `Docker Hub <https://hub.docker.com/u/openmined>`_

Join our slack
--------------
Our rapidly growing community of 12,000+ can be found on `Slack <http://slack.openmined.org/>`_. The Slack
community is very friendly and great about quickly answering questions about the
use and development of PySyft!

.. toctree::
  :maxdepth: 1
  :hidden:

  getting_started/index

.. toctree::
  :maxdepth: 1
  :hidden:

  api_reference/index

.. toctree::
  :maxdepth: 1
  :hidden:

  developer_guide/index

.. toctree::
  :maxdepth: 1
  :hidden:

  deployment/glossary

.. toctree::
  :maxdepth: 1
  :hidden:

  resources/index

.. toctree::
  :maxdepth: 1
  :hidden:

  guides/index

.. toctree:
   :caption: API Docs


 .. rubric:: Modules

 .. autosummary::
   :toctree: api_reference
   :recursive:

   syft.client
   syft.external
   syft.server
   syft.serde
   syft.service
   syft.store
   syft.types
   syft.util


.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/main_panels/index_getting_started.svg

    Getting Started
    ^^^^^^^^^^^^^^^

    New to *PySyft*? Here you can find a guide into installing and first steps into
    using PySyft, as a data owner or data scientist.

    +++

    .. link-button:: getting_started/index
            :type: ref
            :text: To the getting started guides
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/main_panels/index_user_guide.svg

    User guide
    ^^^^^^^^^^

    The user guide provides in-depth explanation on the key concepts used by PySyft
    and a glossary of terms you may encounter.

    +++

    .. link-button:: user_guide/index
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/main_panels/index_api.svg

    API reference
    ^^^^^^^^^^^^^

    The reference guide contains description of the PySyft API, covering how the
    methods work and which parameters can be used.
    It assumes that you have an understanding of the key concepts.

    +++

    .. link-button:: api_reference/index
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/main_panels/index_contribute.svg

    Developer guide
    ^^^^^^^^^^^^^^^

    Want to contribute to Openmined tech stack, fix a typo or file a bug? Here
    you can find extensive guidelines on how you can help improve PySyft.

    +++

    .. link-button:: developer_guide/index
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link
