# Contributors Guidelines to PyGrid

## Getting Started

### Slack

A great first place to join the Community is the Slack channel <http://slack.openmined.org>.

### Issues

On <https://github.com/OpenMined/PyGrid/issues> you can find all open Issues. You can find a detailed explanation on how to work with issues below under [Issue Allocation](#Issue-Allocation).

## Setup

### Forking a Repository

To contribute to PyGrid you will need to fork the OpenMind/PyGrid repository.
Then you can work risk-free on your fork.

You will just need to fork once. After that you can call `git fetch upstream` and `git pull 'branch-name'` before you do your local changes to get the remote changes and be up-to-date

### Setting up Pre-Commit Hook

PyGrid uses the python package `pre-commit` to make sure the correct formatting (black & flake) is applied.

You can install it via `pip install pre-commit`

Then you just need to call `pre-commit install`

This can all also be done by running `make install_hooks`

### Syncing a Forked Repository

To sync your fork with the OpenMined/PyGrid repository please see this [Guide](https://help.github.com/articles/syncing-a-fork/) on how to sync your fork.

## Contributing

### Beginner Issues

If you are new to the project and want to get into the code, we recommend picking an issue with the label "good first issue". These issues should ony require general programming knowledge and little to none insights into the project.

### Issue Allocation

Each issue someone is currently working on should have an assignee. If you want to contribute to an issue someone else is already working on please make sure to get in contact with that person via slack or github and organize yourself.

If you want to work on an open issue, please post a comment telling that you will work on that issue, we will assign you as the assignee then.

**Caution**: We try our best to keep the assignee up-to-date but as we are all humans with our own schedule delays are possible, so make sure to check the comments once before you start working on an issue even when no one is assigned to it.

### Set up

#### Dependencies

You'll need to have the following dependencies installed:

Heroku Toolbelt: https://toolbelt.heroku.com/
Pip: https://www.makeuseof.com/tag/install-pip-for-python/
Git: https://gist.github.com/derhuerst/1b15ff4652a867391f03
PySyft: https://github.com/OpenMined/PySyft

You can install most of the dependencies by running `pip install -r requirements.txt`.

#### Building PyGrid

You can build grid by running: `python setup.py install`.

### Writing Test Cases

Always make sure to create the necessary tests and keep test coverage at 100%. You can always ask for help in slack or via github if you don't feel confidant about your tests.

```bash
coverage run --omit=*/venv/*,setup.py,.eggs/* setup.py test
```

### Documentation and Codestyle

To ensure code quality and make sure other people can understand your changes, you have to document your code. For documentation we are using the Google Python Style Rules which can be found [here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). A well wrote example can we viewed [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

You documentation should not describe the obvious, but explain what's the intention behind the code and how you tried to realize your intention.

You should also document non self-explanatory code fragments e.g. complicated for-loops. Again please do not just describe what each line is doing but also explain the idea behind the code fragment and why you decided to use that exact solution.

#### Imports

For better merge compatibility each import is within a separate line. Multiple imports from one package are written in one line each.

### Keep it DRY (Don't repeat yourself)

As with any software project it's important to keep the amount of code to a minimum, so keep code duplication to a minimum!

### Creating a Pull Request

At any point in time you can create a pull request, so others can see your changes and give you feedback.
Please create all pull requests to the `dev` branch.
If your PR is still work in progress and not ready to be merged please add a `[WIP]` at the start of the title.
Example:`[WIP] Websocket worker for PyGrid`

### Check CI and Wait for Reviews

After each commit GitHub Actions will check your new code against the formatting guidelines (should not cause any problems when you setup your pre-commit hook) and execute the tests to check if the test coverage is high enough.

We will only merge PRs that pass the GitHub Actions checks.

If your check fails don't worry you will still be able to make changes and make your code pass the checks.
