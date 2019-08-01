# Contributors Guidelines to PySyft

## Getting Started

### Slack

A great first place to join the Community is the Slack channel <http://slack.openmined.org>.

### Issues

On <https://github.com/OpenMined/PySyft/issues> you can find all open Issues. You can find a detailed explanation on how to work with issues below under [Issue Allocation](#issue-allocation).

## Setup

### Forking a Repository

To contribute to PySyft you will need to fork the OpenMind/PySyft repository.
Then you can work risk-free on your fork.

You will just need to fork once. After that you can call `git fetch upstream` and `git pull 'branch-name'` before you do your local changes to get the remote changes and be up-to-date

### Setting up Pre-Commit Hook

PySyft uses the python package `pre-commit` to make sure the correct formatting (black & flake) is applied.

You can install it via `pip install -r requirements_dev.txt` or directly doing `pip install pre-commit`

Then you just need to call `pre-commit install`

This can all also be done by running `make install_hooks`

### Syncing a Forked Repository

To sync your fork with the OpenMined/PySyft repository please see this [Guide](https://help.github.com/articles/syncing-a-fork/) on how to sync your fork.

### Installing PySyft after Cloning Repository

To install the development version of the package, once the `dev` version of the requirements have been satisified, one should follow the instructions as laid out in [INSTALLATION.md](https://github.com/OpenMined/PySyft/blob/dev/INSTALLATION.md) to complete the installation process. Effectively do the following two steps after a clone has been made on one's local machine at the terminal and that the pre-commit hook has been set up as described above in [Setting up Pre-Commit Hook](#syncing-a-forked-repository):
```bash
cd PySyft
pip install -e .
```
If you are using a virtual environment, please be sure to use the correct executable for `pip` or `python` instead. 

## Contributing

### Beginner Issues

If you are new to the project and want to get into the code, we recommend picking an issue with the label "good first issue". These issues should only require general programming knowledge and little to none insights into the project.

### Issue Allocation

Each issue someone is currently working on should have an assignee. If you want to contribute to an issue someone else is already working on please make sure to get in contact with that person via slack or github and organize yourself.

If you want to work on an open issue, please post a comment telling that you will work on that issue, we will assign you as the assignee then.

**Caution**: We try our best to keep the assignee up-to-date but as we are all humans with our own schedule delays are possible, so make sure to check the comments once before you start working on an issue even when no one is assigned to it.

### Writing Test Cases

Always make sure to create the necessary tests and keep test coverage at 100%. You can always ask for help in slack or via github if you don't feel confident about your tests.

We aim to have a 100% test coverage, and the Travis CI will fail if the coverage is below this value. You can evaluate your coverage using the following commands.

```bash
coverage run --omit=*/venv/*,setup.py,.eggs/* setup.py test
coverage report --fail-under 100 -m
```

PySyft is using `pytest` to execute the test cases.

#### Parametrize your Test Cases

Sometimes you want to test functions that hold multiple arguments, which again can have multiple values. To test this, please parametrize your tests.

Example:

```python
@pytest.mark.parametrize(
        "compress, compressScheme", [(True, "lz4"), (False, "lz4"), (True, "zstd"), (False, "zstd")]
    )
def test_hooked_tensor(self, compress, compressScheme):
    TorchHook(torch)

    t = Tensor(numpy.random.random((100, 100)))
    t_serialized = serialize(t, compress=compress, compressScheme=compressScheme)
    t_serialized_deserialized = deserialize(
        t_serialized, compressed=compress, compressScheme=compressScheme
        )
    assert (t == t_serialized_deserialized).all()
```

### Documentation and Codestyle

To ensure code quality and make sure other people can understand your changes, you have to document your code. For documentation we are using the Google Python Style Rules which can be found [here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). A well wrote example can we viewed [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

You documentation should not describe the obvious, but explain what's the intention behind the code and how you tried to realize your intention.

You should also document non self-explanatory code fragments e.g. complicated for-loops. Again please do not just describe what each line is doing but also explain the idea behind the code fragment and why you decided to use that exact solution.

#### Imports

For better merge compatibility each import is within a separate line. Multiple imports from one package are written in one line each.

Example:

```python
from syft.serde import serialize
from syft.serde import deserialize
```

#### Generating Documentation

```bash
sphinx-apidoc -f -o docs/modules/ syft/
```

### Keep it DRY (Don't repeat yourself)

As with any software project it's important to keep the amount of code to a minimum, so keep code duplication to a minimum!

### Creating a Pull Request

At any point in time you can create a pull request, so others can see your changes and give you feedback.
Please create all pull requests to the `dev` branch.
If your PR is still work in progress and not ready to be merged please add a `[WIP]` at the start of the title.
Example:`[WIP] Serialization of PointerTensor`

### Check CI and Wait for Reviews

After each commit TravisCI will check your new code against the formatting guidelines (should not cause any problems when you setup your pre-commit hook) and execute the tests to check if the test coverage is high enough.

We will only merge PRs that pass the TravisCI checks.

If your check fails don't worry you will still be able to make changes and make your code pass the checks.
