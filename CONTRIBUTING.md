# Contributing to DyNAS-T

Contributions are accepted in the form of:

* Submitting issues against the current code to report bugs or request features.
* Providing code to fix any of the issues.
* Extending DyNAS-T functionality with important features (e.g. to address community requests, improve usability, implement a new functionality, etc.)
* Adding well-defined patches that integrate DyNAS-T into third-party repositories.

Any contributions must not violate the repository's [LICENSE](./LICENSE.md) requirements.

## Installation

For development purposes it is advised to install DyNAS-T in editable mode. This will create a link between the codebase and your Python environment and install all dependencies required to run DyNAS-T, allowing for easier development and testing of the proposed changes. For testing it is also required to install all packages listed in `requirements_test.txt`.

```bash
cd "DyNAS-T's root directory"
pip install -e ".[test]"  # This will install both base and test dependencies.
```

## Code Style

To ensure consistency and quality in our codebase, we require that all submitted code be verified using specific code checking tools. Before submitting your code, please make sure you've checked it with mypy, isort, and black. These tools help maintain a uniform code format, type consistency, and import order.

If you're unfamiliar with the configurations we use for these tools, refer to our `linter.sh` script. This script contains the detailed configurations for each tool. Furthermore, for your convenience, you can execute `linter.sh`locally to automatically fix any code-style inconsistencies in your contribution.

Please note: the `linter.sh` script is also integrated into our Continuous Integration (CI) pipeline. When you submit a Pull Request (PR), this script will be run to validate your code's style. PRs that do not conform to our code style guidelines will be blocked until the necessary corrections are made.

By adhering to these guidelines, you help us maintain a clean and consistent codebase. We appreciate your understanding and effort in this regard.

> TL;DR: You can run automated linting and checks with:
>
> ```bash
> ./linter.sh
> ```
> It's advised to run this script before every `git commit`.


## Testing

### Unit Tests

```bash
pytest
```

It's advised to run `pytest` on every commit, but the bare minimum is to run it at least once before submitting a Pull Request for review.

### Additional checks (license headers, etc.)

```bash
pytest -c pytest.checks.ini
```

It's advised to run license check at least once before submitting a Pull Request for review.

### Functional checks

```bash
pytest -c pytest.functional.ini
```

### Test search results

Before each release tests must be run to confirm consistency in search results for all supported search spaces. To this end all scripts `tests/scripts/*_linas_long.sh` should be run and search result output inspected.
