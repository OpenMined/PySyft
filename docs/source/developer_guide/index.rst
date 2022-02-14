.. _developer_guide:

======================
Contributor Guidelines
======================

Getting Started
###############

Git and Github
^^^^^^^^^^^^^^

All our development is done using Git and Github. If you're not too familiar with Git and Github, `start by reviewing this guide <https://guides.github.com/activities/hello-world>`_.

Slack
^^^^^

`Join our Slack community <http://slack.openmined.org>`_.

Setup
#####

Start by "forking"
^^^^^^^^^^^^^^^^^^

To contribute to any OpenMined repository you will need to `fork a repository <https://guides.github.com/activities/forking/>`_. Then you can work risk-free on your fork without affecting development on the main repository.

You will just need to fork once. After that you can call ``git fetch upstream`` and ``git pull 'branch-name'``` before you do your local changes to get the remote changes and be up-to-date.

Syncing with the latest changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To sync your fork with the main repository `please see this guide <https://help.github.com/articles/syncing-a-fork/>`_.

Installation
^^^^^^^^^^^^

Depending on which repository you're trying to contribute to, you will likely need to install various code dependencies first. Be sure to read the main README file to see what you need to do to get started.

Contributing
------------

Beginner Issues
^^^^^^^^^^^^^^^

If you are new to the project and want to get into the code, we recommend picking an issue with the label ``Good first issue``. These issues should only require general programming knowledge and little to no insights into technical aspects the project.

Issue Allocation
^^^^^^^^^^^^^^^^

Each issue someone is currently working on should have an assignee. If you want to contribute to an issue someone else is already working on please make sure to get in contact with that person via Slack or Github and organize yourself.

If you want to work on an open issue, please post a comment on that issue and we will assign you as the assignee.

.. note:: We try our best to keep the assignee up-to-date, but as we are all humans with our own scheduling issues, make sure to check the comments once before you start working on an issue even when no one is assigned to it.

Writing Test Cases
^^^^^^^^^^^^^^^^^^

Always make sure to create the necessary tests and keep test coverage as high or higher than it was before you. You can always ask for help in Slack or Github if you don't feel confident about your tests. We're happy to help!

Documentation and Codestyle
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure code quality and make sure other people can understand your changes, you have to document your code. For documentation and general code cleanliness, we ask that you `follow the appropriate styleguide <https://github.com/OpenMined/.github/blob/master/STYLEGUIDE.md>`_ for the language you're working in.

Keep it DRY (Don't repeat yourself)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with any software project, it's important to keep the amount of code to a minimum, so keep code duplication to a minimum!

Creating a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^
At any point in time you can create a pull request, so others can see your changes and give you feedback.

.. note:: If your PR is still work in progress and not ready to be merged please add a `[WIP]` at the start of the title.
Example:`[WIP] Serialization of PointerTensor`

Check CI and Wait for Reviews
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After each commit GitHub Actions will check your new code against the formatting guidelines, test acceptance, and code coverage. **We will only merge PR's that pass the GitHub Actions checks.**

If your check fails, don't worry, you will still be able to make changes and make your code pass the checks.
