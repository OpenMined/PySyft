.. _developer_guide:

======================
Contributor Guidelines
======================

Thank you for your interest in contributing to PySyft! This document
contains rules and a step-by-step approach to contributing to any PySyft
repository.

Getting Started
***************

The PySyft community admires and welcomes your expertise and enthusiasm!
We are always excited to work with new contributors and engage with them
to build exciting new features or improve existing bugs.

If you are unsure where to start or how your skills fit in, take a look
at open `Good first issue <https://github.com/OpenMined/PySyft/labels/Good%20first%20issue%20%3Amortar_board%3A>`__ or introduce yourself in the #introductions
channel on
`Slack <https://communityinviter.com/apps/openmined/openmined/>`__, and
we will match you to the issue that fits your expertise or skills.

   Note: PySyft is a community-driven open-source project. We thrive on
   treating everyone equally and valuing our diverse group of
   contributors. To foster a strong commitment to creating an open,
   inclusive and positive environment, we have a
   `Code-of-Conduct <https://github.com/OpenMined/.github/blob/master/CODE_OF_CONDUCT.md>`__
   to make our community thrive.

If you are new to the open-source ecosystem, this
`guide <https://opensource.guide/how-to-contribute/>`__ will help you
explain what, why and how to contribute to open-source for first-timers
and for beginners.

Setup
*****

1. Fork
~~~~~~~

To avoid merge conflicts with the main development branch, you need to
`fork <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`__
PySyft repository.

After that, you need to
`sync <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork>`__
your fork with the main repository. Call ``git fetch upstream`` and
``git pull 'branch-name'`` before you do your local changes to get the
remote changes and be up-to-date.

Depending on which repository you’re trying to contribute to, you will
first need to install various code dependencies. Be sure to read the
main `README <https://github.com/OpenMined/PySyft/blob/dev/README.md>`__
file to see what you need to do to get started with the PySyft library.

2. Issues
~~~~~~~~~

The PySyft `issue <https://github.com/OpenMined/PySyft/issues>`__
tracker has a lot of open-issues. Some things to keep in mind before
picking an issue are:

-  Use labels to filter down your ideal issue.
-  Find issues related to a bug or want some feature update.
-  Find duplicate issues and link related ones.

We recommend picking an open issue with the label `Good first issue <https://github.com/OpenMined/PySyft/labels/Good%20first%20issue%20%3Amortar_board%3A>`__
as a starter. These issues should only require general programming
knowledge and little to no insights into technical aspects of the
project.

After you have decided upon the right issue, please post a comment on
that issue, and we will assign you as the assignee. Some things to keep
in mind are:

-  Each issue someone is currently working on should have an assignee.
   We try our best to keep the assignee up-to-date.
-  If you want to contribute to an issue someone else is already working
   on, please contact that person via Slack or Github and organize
   yourself.
-  Make sure to check the comments once before you start working on an
   issue, even when no one is assigned to it.

3. Testing
~~~~~~~~~~

The PySyft library is in the development stage, and we always recommend
creating the necessary tests and keeping the test coverage as high or
higher than before.

You can always ask for help in Slack or Github if you don’t feel
confident about your tests. We’re happy to help!

4. Documentation
~~~~~~~~~~~~~~~~

It is vital to make sure other people can understand your changes. When
adding a new module or feature to the PySyft library, we encourage you
to ensure code quality by documenting your code.

We ask you to follow the appropriate style guide to its best practices
for the language you’re working in for documentation and general code
cleanliness.

OpenMined enforces the following style guides by language:

-  `Python <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__
-  `Javascript <https://prettier.io/>`__
-  `C++ <https://google.github.io/styleguide/cppguide.html>`__
-  `Java <https://google.github.io/styleguide/javaguide.html>`__
-  `Kotlin <https://kotlinlang.org/docs/coding-conventions.html>`__
-  `Swift <https://google.github.io/swift/>`__
-  `Go <https://go.dev/doc/effective_go>`__
-  `Rust <https://doc.rust-lang.org/1.0.0/style/README.html>`__
-  `Julia <https://docs.julialang.org/en/v1/manual/style-guide/>`__
-  `R <https://google.github.io/styleguide/Rguide.html>`__

..

   Note: As with any software project, keeping the amount of code to a
   minimum is essential, so keep code duplication to a minimum!

5. Pull Requests
~~~~~~~~~~~~~~~~

At any point in time, you can send a `GitHub Pull
Request <https://github.com/OpenMined/PySyft/pulls>`__ to PySyft (read
more about `pull
requests <https://docs.github.com/en/pull-requests>`__), so others can
see your changes and give you feedback.

Before sending a PR, it is crucial to make sure you comply with the
below instructions:

-  No commits should be made to the ``master`` or ``dev`` branch directly.
-  Always write a clear log message for your commits describing what
   changes you have done and their impacts.
-  If your PR is still a work in progress and not ready to be merged,
   please add an [WIP] at the start of the title.

   -  Example: ``[WIP] Serialization of PointerTensor``

-  Name the branch as either the feature you are implementing or the
   issue you are trying to fix.

   -  Example: ``Network_monitor``

6. Review
~~~~~~~~~

After each commit, GitHub Actions will check your new code against the
formatting guidelines, test acceptance, and code coverage. We will only
merge PRs that pass the GitHub Actions checks.

If your check fails, don’t worry, you will still be able to make changes
and make your code pass the checks.
