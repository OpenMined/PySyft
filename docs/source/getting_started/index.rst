.. _getting_started:

===============
Getting started
===============

.. toctree::
  :maxdepth: 1

Why PySyft and PyGrid?
######################

Syft decouples private data from model training, using
`Federated Learning <https://ai.googleblog.com/2017/04/federated-learning-collaborative.html>`_,
`Differential Privacy <https://en.wikipedia.org/wiki/Differential_privacy>`_,
and Encrypted Computation (like
`Multi-Party Computation (MPC) <https://en.wikipedia.org/wiki/Secure_multi-party_computation>`_
and `Homomorphic Encryption (HE) <https://en.wikipedia.org/wiki/Homomorphic_encryption>`_
within the main Deep Learning frameworks like PyTorch and TensorFlow.

Most software libraries let you compute over the information you own and see inside of machines you control. However, this means that you cannot compute on information without first obtaining (at least partial) ownership of that information. It also means that you cannot compute using machines without first obtaining control over those machines. This is very limiting to human collaboration and systematically drives the centralization of data, because you cannot work with a bunch of data without first putting it all in one (central) place.

The Syft ecosystem seeks to change this system, allowing you to write software which can compute over information you do not own on machines you do not have (total) control over. This not only includes servers in the cloud, but also personal desktops, laptops, mobile phones, websites, and edge devices. Wherever your data wants to live in your ownership, the Syft ecosystem exists to help keep it there while allowing it to be used privately for computation.

Beginner-level PySyft & PyGrid Installation
###########################################

.. toctree::
   :maxdepth: 3

Welcome to the domain deployment installation tutorials!
This section of our documentation is designed to be the
simplest way to get you started deploying a PyGrid Domain
to an OSX, Linux, or Windows machine and interacting with it
as a data scientist using PySyft. If you're looking
for cloud deployment, or more advanced tutorials such as
ansible, vagrant, kubernetes, or virtualbox deployment, please see the
`advanced deployment documentation <https://openmined.github.io/PySyft/deployment/index.html#>`__.

The purpose of these tutorials is to help you install everything
you need to run a Domain node from your personal machine (such
as if you're running through OpenMined
`courses <https://courses.openmined.org/#>`__
or
`tutorials <https://github.com/OpenMined/PySyft/tree/dev/notebooks#>`__).
To that end, we will also be installing everything you might need to run Jupyter
notebooks with PySyft installed, such as if you're pretending to be
both Data Owner and Data Scientist as a part of a tutorial or course.

Step 1: Are you on OSX, Windows, or Linux?
==========================================

Installation differs greatly depending on whether your personal machine is
running OSX, Linux, or Windows. PySyft and PyGrid are relatively new pieces
of software so not all versions of these are supported. However, the first
step of your journey is to figure out which operating system you are running
and choose the right tutorial for installation. Then within the dropdowns below,
choose which version is right for you. Once you've found the right version,
and completed the tutorial for that version, you'll be all done!!! Good luck!

There are 3 types of operating systems for you to choose from: OSX, Linux, and Windows.

OSX Tutorials
~~~~~~~~~~~~~

If you know you're running OSX but you're not sure what version you're running,
click the Apple logo at the top left corner, then click "About this Mac" and you'll
see something like:

|find_osx_version|

See where this image says "11.5.1"? Figure out what number yours says in that place
and use that number to determine which of these installation tutorials you should
follow to complete your installation. If you don't see your number, choose the
closest that you can.

#. `Big Sur (11.5.1) <https://openmined.github.io/PySyft/install_tutorials/osx_11_5_1.html#>`__.

Linux Tutorials
~~~~~~~~~~~~~~~

If you know that you're running Linux but you're not sure what version you're running,
open up a command line and type:

.. code-block:: bash

  $ lsb_release -a

Which should print something like the following:

|find_ubuntu_version|

See where this image says "20.04.3"? Figure out what number yours says in that place

#. `Ubuntu (20.04.3 - Focal Fossa) <https://openmined.github.io/PySyft/install_tutorials/linux.html##>`__.

Windows Tutorials
~~~~~~~~~~~~~~~~~

If you know that you're running Windows but you're not sure what version you're running,
press (Windows Key + R) and then in the text box that appears type:

.. code-block:: bash

  $ winver

and hit (Enter)! This should print something like the following:

|find_windows_version|

See where this image says "Windows 10" and "20H2"? Figure out what numbers yours says in those place
and use those number to determine which of these installation tutorials you should
follow to complete your installation. If you don't see one of your numbers, choose the
closest that you can.

#. `Windows 10 (20H2) <https://openmined.github.io/PySyft/install_tutorials/windows.html>`__.

Best of luck on your journey!

.. |find_osx_version| image:: ../_static/install_tutorials/find_osx_version.png
   :width: 50%

.. |find_ubuntu_version| image:: ../_static/install_tutorials/find_ubuntu_version.png
   :width: 50%

.. |find_windows_version| image:: ../_static/install_tutorials/find_windows_version.png
   :width: 50%



F.A.Q
#####
.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                          What is a privacy budget?
                    </div>
                    <span class="badge gs-badge-link">

.. raw:: html

                      </span>
                  </div>
              </div>
              <div id="collapseOne" class="collapse" data-parent="#accordion">
                  <div class="card-body">

  The privacy budget is a collection of quantitative measures through which a
  data owner can pre-determine the degree of information access they grant to a
  data scientist so that that limit is automatically enforceable through
  automated systems. In our specific setup the privacy budget is measured against
   data subjects, not datasets. Therefore, the epsilon value indicates how much
   can be learned from any one data subject.
