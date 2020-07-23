======
PySyft
======

A library for computing on data you do not own and cannot see.


Contents
========

.. toctree::
   :maxdepth: 2

   User Guide <user_guide>
   Contributor Guide <contributor_guide>
   License <license>
   Authors <authors>
   Changelog <changelog>
   Module Reference <api/modules>

Description
===========

Most software libraries let you compute over information you own and see inside of machines you control. However, this means that you cannot compute on information without first obtaining (at least partial) ownership of that information. It also means that you cannot compute using machines without first obtaining control over those machines. This is very limiting to human collaboration in all areas of life and systematically drives the centralization of data, because you cannot work with a bunch of data without first putting it all in one (central) place.

The Syft ecosystem seeks to change this system, allowing you to write software which can compute over information you do not own on machines you do not have (general) control over. This not only includes servers in the cloud, but also personal desktops, laptops, mobile phones, websites, and edge devices. Wherever your data wants to live in your ownership, the Syft ecosystem exists to help keep it there while allowing it to be used for computation.

This library is the centerpiece of the Syft ecosystem. It has two primary purposes. You can either use PySyft to:

1) **Dynamic:** Directly compute over data you cannot see.
2) **Static:** Create static graphs of computation which can be deployed/scaled at a later date on different compute.

The Syft ecosystem includes libraries which allow for communication with and computation over a variety of runtimes:

- KotlinSyft_ (Android) - run PySyft computations in Kotlin
- SwiftSyft_ (iOS) - run PySyft computations in Swift
- syft.js_ (Web & Mobile) - run PySyft computations in Javascript
- PySyft_ (Python) - the core PySyft library itself, capable of creating and running computations

.. _KotlinSyft: https://github.com/OpenMined/KotlinSyft
.. _SwiftSyft: https://github.com/OpenMined/SwiftSyft
.. _syft.js: https://github.com/OpenMined/syft.js
.. _PySyft: https://github.com/OpenMined/PySyft

However, the Syft ecosystem only focuses on consistent object serialization/deserialization, core abstractions, and algorithm design/execution across these languages. These libraries alone will not connect you with data in the real world. The Syft ecosystem is supported by the Grid ecosystem, which focuses on deployment, scalability, and other additional concerns around running real-world systems to compute over and process data (such as data compliance web applications). Syft is the library that defines objects, abstractions, and algorithms. Grid is the platform which lets you deploy them within a real institution (or on the open internet, but we don't yet recommend this). The Grid ecosystem includes:

- GridNetwork_ - think of this like DNS for private data. It helps you find remote data assets so that you can compute with them.
- PyGrid_ - This is the gateway to an organization's data, responsible for permissions, load balancing, and governance.
- GridNode_ - This is an individual node within an organization's datacenter, running executions requested by external parties.
- GridMonitor_ - This is a UI which allows an institution to introspect and control their PyGrid node and the GridNodes it manages.

.. _GridNetwork: https://github.com/OpenMined/GridNetwork
.. _PyGrid: https://github.com/OpenMined/PyGrid
.. _GridNode: https://github.com/OpenMined/GridNode
.. _GridMonitor: https://github.com/OpenMined/GridMonitor

Want to Use PySyft?
===================

If you would like to become a user of PySyft, please progress to our :ref:`contributor_guide`.

Want to Develop PySyft?
=======================

If you would like to become a developer of PySyft, please see our :ref:`contributor_guide`. This documentation will help you setup your development environment, give you a roadmap for learning the codebase, and help you find your first project to contribute.



.. note::

    This is the main page of your project's `Sphinx`_ documentation.
    It is formatted in `reStructuredText`_. Add additional pages
    by creating rst-files in ``docs`` and adding them to the `toctree`_ below.
    Use then `references`_ in order to link them from this page, e.g.
    :ref:`authors` and :ref:`changes`.

    It is also possible to refer to the documentation of other Python packages
    with the `Python domain syntax`_. By default you can reference the
    documentation of `Sphinx`_, `Python`_, `NumPy`_, `SciPy`_, `matplotlib`_,
    `Pandas`_, `Scikit-Learn`_. You can add more by extending the
    ``intersphinx_mapping`` in your Sphinx's ``conf.py``.

    The pretty useful extension `autodoc`_ is activated by default and lets
    you include documentation from docstrings. Docstrings can be written in
    `Google style`_ (recommended!), `NumPy style`_ and `classical style`_.


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
