# Contributor Guidelines

## Getting Started

### Git and Github

All our development is done using Git and Github. If you're not too familiar with Git and Github, [start by reviewing this guide](https://guides.github.com/activities/hello-world).

### Slack

[Join our Slack community](http://slack.openmined.org).

### Issues

On <https://github.com/OpenMined/PyGrid/issues> you can find all open issues.

## Setup

### Start by "forking"

To contribute to PyGrid you will need to [fork the OpenMind/PyGrid repository](https://guides.github.com/activities/forking/). Then you can work risk-free on your fork without affecting development on the main repository.

You will just need to fork once. After that you can call `git fetch upstream` and `git pull 'branch-name'` before you do your local changes to get the remote changes and be up-to-date.

### Setting up Pre-Commit Hook

PyGrid uses the python package `pre-commit` to make sure the correct formatting (black & flake) is applied.

You can install it via `pip install pre-commit`

Then you just need to call `pre-commit install`

This can all also be done by running `make install_hooks`

### Syncing with the latest changes

To sync your fork with the main repository [please see this guide](https://help.github.com/articles/syncing-a-fork/).

## Installation

### Dependencies

[Start by installing Poetry](https://python-poetry.org/docs/), our dependency manager.

Then, inside each of the `apps/*` folders, run `poetry install`. This should install your dependencies. From there, you can make your changes and run the `./run.sh` file to run the app.

### Writing Test Cases

Always make sure to create the necessary tests and keep test coverage at 100%. You can always ask for help in Slack if you don't feel confident about your tests.

Make sure that when running the test suite, you have `cd`'d into the appropriate `apps/*` folder. From there, you can run the following to run the test suite:

```
poetry run coverage run -m pytest -v tests
```

To run the integration tests, make sure you're in the `apps/node` directory and run the following:

```
poetry run coverage run -m pytest -v ../../tests
```

## Tips

### Beginner Issues

If you are new to the project and want to get into the code, we recommend picking an issue with the label "Good first issue". These issues should only require general programming knowledge and little to no insights into technical aspects the project.

### Issue Allocation

Each issue someone is currently working on should have an assignee. If you want to contribute to an issue someone else is already working on please make sure to get in contact with that person via Slack or Github and organize yourself.

If you want to work on an open issue, please post a comment on that issue and we will assign you as the assignee.

**Note**: We try our best to keep the assignee up-to-date, but as we are all humans with our own scheduling issues, make sure to check the comments once before you start working on an issue even when no one is assigned to it.

### Writing Test Cases

Always make sure to create the necessary tests and keep test coverage as high or higher than it was before you. You can always ask for help in Slack or Github if you don't feel confident about your tests. We're happy to help!

### Documentation and Codestyle

To ensure code quality and make sure other people can understand your changes, you have to document your code. For documentation we are using the Google Python Style Rules which can be found [here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). A well wrote example can we viewed [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

You documentation should not describe the obvious, but explain what's the intention behind the code and how you tried to realize your intention.

You should also document non self-explanatory code fragments e.g. complicated for-loops. Again please do not just describe what each line is doing but also explain the idea behind the code fragment and why you decided to use that exact solution.

### Keep it DRY (Don't repeat yourself)

As with any software project, it's important to keep the amount of code to a minimum, so keep code duplication to a minimum!

### Creating a Pull Request

At any point in time you can create a pull request, so others can see your changes and give you feedback.

**Note**: If your PR is still work in progress and not ready to be merged please add a `[WIP]` at the start of the title.
Example:`[WIP] Serialization of PointerTensor`

### Check CI and Wait for Reviews

After each commit GitHub Actions will check your new code against the formatting guidelines, test acceptance, and code coverage. **We will only merge PR's that pass the GitHub Actions checks.**

If your check fails, don't worry, you will still be able to make changes and make your code pass the checks.
