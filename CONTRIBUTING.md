# Contributing

Thanks for your interest in improving the CodeBERT-based webshell detection
project! This document describes the recommended workflow for proposing
changes.

## Getting Started

1. Fork the repository and clone your fork.
2. Create and activate a Python virtual environment.
3. Install the runtime and development dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. Run the test-suite to verify the local environment is set up correctly:

   ```bash
   pytest
   ```

## Development Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions. The
  repository includes configuration for [Black](https://github.com/psf/black)
  and [Ruff](https://github.com/astral-sh/ruff). Run both tools before opening a
  pull request:

  ```bash
  black src examples scripts tests
  ruff check src examples scripts tests
  ```

- Keep public APIs documented. Docstrings should explain parameters, return
  types and side-effects.
- Prefer small, focused commits. Each commit should represent a logical change
  and the message should describe the behaviour, not the implementation.
- Provide unit tests covering new features or bug fixes. Place them under the
  `tests/` directory following the existing structure.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue with:

- A clear description of the problem or desired change.
- Steps to reproduce the behaviour (if applicable).
- Details about your environment (operating system, Python version,
  dependencies).

## Code of Conduct

Please review the [Code of Conduct](CODE_OF_CONDUCT.md). By participating in the
project you are expected to uphold it.

## License

Unless stated otherwise, contributions are licensed under the same terms as the
project (MIT). Submitting a pull request implies that you agree to this.
