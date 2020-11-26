# Contribution Guidelines

This document contains a set of guidelines to help you during the contribution process.
This project is open source and we welcome contributions from everyone in the form of bug fixes, new features, documentation and suggestions.


## Slack
If you have questions or want to get involved with any of the other exciting projects at OpenMined [Join our Slack community](http://slack.openmined.org).

## Issues / PRs
- Development is done on the `dev` branch so if you want to add a PR please point it at this branch and not `master`.
- If you are working on a existing issue posted by someone else, please ask to be added as Assignee so that effort is not duplicated.
- If you want to contribute to an issue someone else is already working on please get in contact with that person via slack or github and discuss your collaboration.
- If you wish to create your own issue or PR please explain your reasoning within the Issue template and make sure your code passes all the CI checks.

**Caution**: We try our best to keep the assignee up-to-date, but as we are all humans with our own schedules mistakes happen. If you are unsure, please check the comments of the issue to see if someone else has already started work before you begin.

### Beginner Issues
If you are new to the project and want to get into the code, we recommend picking an issue with the label "good first issue". These issues should only require general programming knowledge and little to none insights into the project.

## Requirements
Before you get started you will need a few things installed depending on your operating system.

- OS Package Manager
- Python 3.6+
- git
- protobuf (protoc)


### Linux
If you are using Ubuntu this is `apt-get` and should already be available on your machine.
### MacOS
On MacOS the main package manager is called [Brew](https://brew.sh/).

Install Brew with:
```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```
Afterwards you can now use the `brew` package manager for installing additionally required packages below.

### Windows
For windows the recommended package manager is [chocolatey](https://chocolatey.org/).

## Git
You will need git to clone, commit and push code to GitHub.

### MacOS
```
$ brew install git
```

## protoc
We utilise protobuf and `protoc` protobuf compiler to automatically code generate python protobuf interfaces for much of our serialization / deserialization functionality. To generate protobufs you need the protoc tool installed and available on your path. Protoc / protobuf are available on many OS Package Managers and there are pre-compiled binaries for many systems available on their repo: https://github.com/protocolbuffers/protobuf

### MacOS
Install protobuf and protoc like this:
```
$ brew install protobuf
```


## Python Versions
This project supports Python 3.6+, however if you are contributing it can help to be able to switch between python versions to fix issues or bugs that relate to a specific python version. Depending on your operating system there are a number of ways to install different versions of python however one of the easiest is with the `pyenv` tool. Additionally as we will be frequently be installing and changing python packages for this project we should isolate it from your system python and other projects you have using a virtualenv.

### MacOS

Install the `pyenv` tool with `brew`:
```
$ brew install pyenv
```

### Using pyenv
Running the command will give you help:
```
$ pyenv
```

Lets say you wanted to install python 3.6.9 because its the version that Google Colab uses and you want to debug a Colab issue.

First, search for available python versions:
```
$  pyenv install --list | grep 3.6
...
3.6.7
3.6.8
3.6.9
3.6.10
3.6.11
3.6.12
```

Wow, there are lots of options, lets install 3.6.
```
$ pyenv install 3.6.9
```

Now, lets see what versions are installed:
```
$ pyenv versions
3.5.9
3.6.9
3.7.8
3.9.0
```

Thats all we need for now. You generally should not change which python version your system is using by default and instead we will use virtualenv manager to pick from these compiled and installed Python versions later.

## Virtual Environments
If you do not fully understand what a Virtual Environment is and why you need it, I would urge you to read this section because its actually a very simple concept but misunderstanding Python, site-packages and virtual environments leads to many common problems when working with projects and packages.

### What is a Virtual Environment
Ever wonder how python finds your packages that you have installed? The simple answer is, it recursively searches up a few folders from where ever the binary `python` or `python.exe` looking for a folder called site-packages.

When you open a shell try typing:
```
$ which python
/usr/local/bin/python3
```

Lets take a closer look at that symlink:
```
$ ls -l /usr/local/bin/python3
/usr/local/bin/python3 -> ../Cellar/python@3.9/3.9.0_1/bin/python3
```

Okay so that means if I run this python3 interpreter im going to get python 3.9.0 and it will look for packages where ever that folder is in my Brew Cellar.

So what if I wanted to isolate a project from that and even use a different version of python you ask?
Quite simply a virtual environment is a folder where you store a copy of the python binary you want to use, and then you change the PATH of your shell to use that binary first so all future package resolution commands including installing packages with `pip` etc will go in that subfolder. This explains why with most virtualenv tools you have to activate them often by running `source` on a shell file to change your shells PATH.

This is so common there are a multitude of tools to help with this, and the process is now officially supported within python3 itself.

**Bonus Points**
Watch: Reverse-engineering Ian Bicking's brain: inside pip and virtualenv
https://www.youtube.com/watch?v=DlTasnqVldc

##### What about Python Package Management
Okay so virtualenvs are only part of the process, they give you isolated folder structures in which you can install, update and delete packages without worrying about messing up other projects. But how do i install a package? Is that only pip, what about conda or pipenv or poetry?

Most of these tools aim to provide the same functionality which is to create virtualenvs, and handle the installation of packages as well as making the experience of activating and managing virtualenvs as seamless as possible. Some, as in the case of conda even provide their own package repositories and additional non python package support.

For the example below I will be using `pipenv` purely because it is extremely simple to use, and is itself simply a pip package which means as long as you have any version of python3 on your system you can use this to bootstrap everything else.

name | packages | virtualenvs
--- | --- | --- | ---
pip + venv | ✅ | ✅
pipenv | ✅ | ✅
conda | ✅ | ✅
poetry | ✅ | ✅


## Pipenv
As you will be running pipenv to create virtualenvs you will want to install pipenv into your normal system python site-packages.
This can be achieved by simply `pip` installing it from your shell.

```
$ pip install pipenv
```

### Common Issues
- what is the difference between pip and pip3?
pip3 was introduced as an alias to use the pip package manager from python3 on systems where python 2.x is still used by the operating system.
When in doubt use pip3 or check the path and version that your python or pip binary is using.
- I don't have pip?
On some systems like Ubuntu you need to install pip first with `apt-get install python3-pip` or you you can use the new official way to install pip from python:
```
$ python3 -m ensurepip
```

## Git Repo

### Forking PySyft
As you will be making contributions you will need somewhere to push your code. The way you do this is by forking the repository so that your own github user profile has a copy of the source code.

Navigate to the page and click the fork button:
https://github.com/OpenMined/pysyft

You will now have a url like this with your copy:
https://github.com/<your-username>/pysyft

### Clone GitHub Repo
```
$ git clone https://github.com/<your-username>/pysyft
$ cd pysyft
```

### Switch to Dev Branch
The majority of our work will fork off dev.
```
$ git checkout dev
```

### Branching
Do not forget to create a branch from `dev` that describes the issue or feature you are working on.
```
$ git checkout -b "feature_1234"
```

### Syncing your Fork

To sync your fork (remote) with the OpenMined/PySyft (upstream) repository please see this [Guide](https://help.github.com/articles/syncing-a-fork/) on how to sync your fork or follow the given commands.

```
$ git remote update
$ git checkout <branch-name>
$ git rebase upstream/<branch-name>
```

### Learn More Git
If you want to learn more about git or github then checkout [this guide](https://guides.github.com/activities/hello-world).

## Setting up the VirtualEnv
Lets create a virtualenv and install the required packages so that we can start developing on Syft.

### Pipenv
Using pipenv you would do the following:
```
$ pipenv --python=3.6
```
We installed python 3.6 earlier so here we can just specify the version and we will get a virtualenv with that version. If you want to use a different version make sure to install it to your system with your system package manager or `pyenv` first.

We have created the virtualenv but it is not active yet.
If you type the following:
```
$ which python
/usr/bin/python
```

You can see that we still have a python path that is in our system binary folder.

Lets activate the virtualenv with:
```
$ pipenv shell
```

You should now see that the prompt has changed and if you run the following:
```
$ which python
/Users/madhavajay/.local/share/virtualenvs/PySyft-lHlz_cKe/bin/python
```

Okay, any time we are inside the virtualenv every python and pip command we run will use this isolated version that we defined and will not effect the rest of the system or other projects.

### Install Python Dependencies
Once you are inside the virtualenv you can do this with pip or pipenv.

**NOTE** this is required for several `dev` packages like pytest-xdist etc.
```
$ pip install -r requirements.txt
```
or
```
$ pipenv install --dev --skip-lock
```

Now you can verify we have installed a lot of stuff by running:
```
$ pip freeze
```

### Linking the PySyft src
Now we need to link the src directory of the pysyft code base into our site-packages
so that it acts like its installed but we can change any file we like and `import` again
to see the changes.

```
$ pip install -e .
```

The best way to know everything is working is to run the tests.

Run the quick tests with all your CPU cores by running:
```
$ pytest -m fast -n auto
```

If they pass then you know everything is setup correctly.


## Jupyter
Jupyter is not in requirements.txt as its technically not needed however you will likely use it extensively in Duet.
Its worth installing this within the Virtual Environment and making sure its a recent version as there are some issues with Jupyter 5.x so its important that you install Jupyter 6+

```
$ cd pysyft
$ pipenv shell
$ pip install jupyter
```

## Duet Network
If you wish to run your own Duet Network instead of the AWS one, simply run the script in a shell:
```
$ syft-network
```

This will start a flask application on port 5000 which you can then pass in to the sy.duet() commands like so:
```python
import syft as sy
duet = sy.duet(network_url="http://127.0.0.1:5000/")
```

## Code Quality

### Formatting, Linting and Type Checking
We use several tools to keep our code base high quality. They are automatically run when you use the `pre_commit.sh` script.
- black
- flake8
- isort
- mypy

### Tests and CI
When you push your code it will run through a series of GitHub Actions which will ensure that the code meets our minimum standards of code quality before a Pull Request can be reviewed and approved.

To make sure your code will pass these CI checks before you push you should use the pre-commit hooks and run tests locally.

We aim to have a 100% test coverage, and the GitHub Actions CI will fail if the coverage is below a certain value. You can evaluate your coverage using the following commands.

```
$ pytest -m fast -n auto
```

### Writing Test Cases

Always make sure to create the necessary tests and keep test coverage at 100%. You can always ask for help in slack or via github if you don't feel confident about your tests.

### Documentation and Code Style Guide
To ensure code quality and make sure other people can understand your changes, you have to document your code. For documentation we are using the Google Python Style Rules which can be found [here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). A well written example can we viewed [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

You documentation should not describe the obvious, but explain what's the intention behind the code and how you tried to realize your intention.

You should also document non self-explanatory code fragments e.g. complicated for-loops. Again please do not just describe what each line is doing but also explain the idea behind the code fragment and why you decided to use that exact solution.

### Imports Formatting
We use isort to automatically format the python imports. Make sure to run it either manually or as part of the `pre_commit.sh` script.

Run isort manually like this:
```
$ isort .
```

### Generating Documentation
```
$ sphinx-apidoc -f -o docs/modules/ syft/
```

## Type Checking
The codebase uses [Mypy](http://mypy-lang.org/) for type hinting the code, providing clarity and catching errors prior to runtime. The pre-commit checks include a very thorough Mypy check so make sure your code passes these checks before you start your PR.

Due to issue [#2323](https://github.com/OpenMined/PySyft/issues/2323) you can ignore existing type issues found by mypy.

## Pre-Commit
We are using a tool called [pre-commit](https://pre-commit.com/) which is a plugin system that allows easy configuration of popular code quality tools such as linting, formatting, testing and security checks.

### MacOS

First install the pre-commit tool:
```
$ brew install pre-commit
```

Now make sure to install the pre-commit hooks for this repo:
```
$ cd pysyft
$ pre-commit install
```

To make sure its working run the pre-commit checks with:
```
$ pre-commit run --all-files
```

Now every time you try to commit code these checks will run and warn you if there was an issue.
This same check is run on CI so if it fails on your machine it will probably fail on GitHub.

## Useful Scripts
We have a number of useful utility bash scripts for Linux and MacOS (or WSL) which we
regularly use during development to perform pre-flight checks before committing and pushing.

- pre_commit.sh
This attempts to replicate what happens on GitHub CI and runs the following checks:
    - pytest -m fast
    - bandit
    - nb_test.sh
    - build_proto.sh
    - isort
    - black
    - pre-commit

If this passes then your code will probably pass CI unless you have an issue in the slow tests.
You can always check that manually with:
```
$ pytest -m slow -n auto
```

- build_proto.sh
This script will re-generate all of the protobuf files using the `protoc` protobuf compiler.
- nb_test.sh
This converts notebooks that have asserts into tests so they can be run with pytest
- colab.sh
This fixes some issues in Colab with python 3.6.9 and our code and helps cloning the
repo if you want to test code which is not on PyPI yet.

### Creating a Pull Request

At any point in time you can create a pull request, so others can see your changes and give you feedback. Please create all pull requests to the `dev` branch.

If your PR is still work in progress and not ready to be merged please add a `[WIP]` at the start of the title and choose the Draft option on GitHub.

Example:`[WIP] Serialization of PointerTensor`

### Check CI and Wait for Reviews
After each commit GitHub Actions will check your new code against the formatting guidelines (should not cause any problems when you setup your pre-commit hook) and execute the tests to check if the test coverage is high enough.

We will only merge PRs that pass the GitHub Actions checks.

If your check fails, don't worry, you will still be able to make changes and make your code pass the checks. Try to replicate the issue on your local machine by running the same check or test which failed on the same version of Python if possible. Once the issue is fixed simply push your code again to the same branch and the PR will automatically update and run CI again.

## Support
For support in contributing to this project and like to follow along with any code changes to the library, please join the #code_pysyft Slack channel. [Click here to join our Slack community!](https://slack.openmined.org/)
