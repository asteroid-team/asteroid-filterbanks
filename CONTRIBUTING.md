## How to contribute

The general way to contribute to Asteroid-filterbanks is to fork the main
repository on GitHub:
1. Fork the [main repo][asteroid] and `git clone` it.
2. Make your changes, test them, commit them and push them to your fork.
3. You can open a pull request on GitHub when you're satisfied.

__Things don't need to be perfect for PRs to be opened__.

If you made changes to the source code, you'll want to try them out without
installing asteroid-filterbanks everytime you change something.
To do that, install asteroid-filterbanks in develop mode either with pip
```pip install -e .[tests]``` or with python ```python setup.py develop```.

To avoid formatting roundtrips in PRs, Asteroid relies on [`ruff`](https://docs.astral.sh/ruff/)
and [`pre-commit-hooks`](https://github.com/pre-commit/pre-commit-hooks) to handle formatting
for us. You'll need to install `requirements.txt` and install git hooks with
`pre-commit install`.

Here is a summary:

```bash
### Install
git clone your_fork_url
cd asteroid-filterbanks
pip install -r requirements.txt
pip install -e .
pre-commit install  # To run `ruff format` before commit

# Make your changes
# Test them locally
# Commit your changes
# Push your changes
# Open a PR!
```

### Source code contributions
__All contributions to the source code of asteroid-filterbanks should be documented
and unit-tested__.
See [here](./tests) to run the tests with coverage reports.
Docstrings follow the [Google format][docstrings], have a look at other
docstrings in the codebase for examples. Examples in docstrings can
be bery useful, don't hesitate to add some!


### Writing new recipes.
Most new recipes should follow the standard format that is described
[here](./egs). We are not dogmatic about it, but another organization should
be explained and motivated.
We welcome any recipe on standard or new datasets, with standard or new
architectures. You can even link a paper submission with a PR number
if you'd like!

### Improving the docs.
If you found a typo, think something could be more explicit etc...
Improving the documentation is always welcome. The instructions to install
dependencies and build the docs can be found [here](./docs).
Docstrings follow the [Google format][docstrings], have a look at other
docstrings in the codebase for examples.

### Coding style

We use [pre-commit hooks](../.pre-commit-config.yaml) to format the code using
`ruff format`.
The code is checked for compliance with `ruff check` and `ruff format` on every
commit with GitHub actions. Remember, continuous integration is not here to be
all green, be to help us see where to improve!


If you have any question, [open an issue][issue] or [join the slack][slack],
we'll be happy to help you.

[asteroid]: https://github.com/mpariente/asteroid-filterbanks
[issue]: https://github.com/mpariente/asteroid-filterbanks/issues/new
[slack]: https://join.slack.com/t/asteroid-dev/shared_invite/zt-cn9y85t3-QNHXKD1Et7qoyzu1Ji5bcA
[pep8]: https://www.python.org/dev/peps/pep-0008/
[docstrings]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
