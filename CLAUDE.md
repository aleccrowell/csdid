# csdid — Claude Code Guidelines

## Build & dependency management

This project uses **Poetry** for dependency management and packaging.

- Install dependencies: `poetry install`
- Run tests: `poetry run pytest`
- Run a specific test: `poetry run pytest tests/path/to/test.py -v`
- Run a script / CLI: `poetry run circ <subcommand>`
- Add a dependency: `poetry add <package>`
- Add a dev dependency: `poetry add --group dev <package>`

**Never use bare `python3 -m pytest` or `pip install`.** Always prefix with `poetry run`.