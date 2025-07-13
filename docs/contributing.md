# Contributing

We love to chat about state-space-models! If you do too, you can get involved in the following ways:

- [Open an issue](https://github.com/state-space-models/cuthbert/issues) if you have a question, suggestion or spotted a bug or specific code that could be improved.
- [Start or join a discussion](https://github.com/state-space-models/cuthbert/discussions) for more general conversation regarding `cuthbert` code structure and design.
- [Open a pull request](https://github.com/state-space-models/cuthbert/pulls) to add code yourself (feel free to make it a draft at an early stage for feedback).
- [Join us on discord](https://discord.gg/EWTjkRjY) for everything else; including memes, research ideas and meetups.

## Opening a Pull Request

A pull request can be opened by following these steps:

1. Fork the repo from GitHub and clone it locally:
```bash
git clone git@github.com/YourUserName/cuthbert.git
cd cuthbert
```

2. Install the cloned version and pre-commit hooks:
```bash
pip install -e .
pre-commit install
```

3. **Add your code. Add your tests. Update the docs.**  

4. Make sure to run the linter, type checker, tests and check coverage:
```bash
pre-commit run --all-files
python -m pytest --cov=cuthbert --cov-report term-missing
```

!!! tip "VS Code Users"
    VS Code users can use the [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
    and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
    (with Type Checking Mode: standard)
    extensions for assistance with linting and type checking during development.

5. Commit your changes and push your new branch to your fork.

6. Open a [pull request on GitHub](https://github.com/state-space-models/cuthbert/pulls).