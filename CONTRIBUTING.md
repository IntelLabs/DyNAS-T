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
pip install -e .
pip install -r requirements_test.txt
```

## Code Style

You can run automated linting and checks with:

```bash
./linter.sh
```

It's advised to run this script before every `git commit`.

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
