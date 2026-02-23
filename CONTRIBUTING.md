# Contributing

We love to chat about state-space-models! If you do too, you can get involved in the following ways:

- [Open an issue](https://github.com/state-space-models/cuthbert/issues) if you have a question, suggestion or spotted a bug or specific code that could be improved.
- [Start or join a discussion](https://github.com/state-space-models/cuthbert/discussions) for more general conversation regarding `cuthbert` code structure and design.
- [Open a pull request](https://github.com/state-space-models/cuthbert/pulls) to add code yourself, follow the steps below (feel free to make it a draft at an early stage for feedback).
- [Join us on discord](https://discord.gg/EWTjkRjY) for everything else; including memes, research ideas and meetups.


## Opening a Pull Request

If you would like to contribute to `cuthbert`, please follow these steps:

1. Fork the [repo](https://github.com/state-space-models/cuthbert/pulls) from GitHub and clone it locally:
```bash
git clone git@github.com/YourUserName/cuthbert.git
cd cuthbert
```

2. Install the package with the development dependencies and pre-commit hooks.

With `uv`:
```bash
uv sync --all-extras
pre-commit install
```

With `pip`:
```bash
pip install -e ./pkg/cuthbertlib
pip install -e "./pkg/cuthbert[tests,docs,examples]"
pre-commit install
```

3. **Add your code. Add your tests. Update the docs if needed.**

4. If you have made changes to the docs, check that they render nicely:
```bash
mkdocs serve
```

5. Make sure to run the linter, type checker, tests and check coverage:
```bash
pre-commit run --all-files
python -m pytest --cov=cuthbert --cov-report term-missing
```

    !!! tip "VS Code Users"
        VS Code users can use the [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
        and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
        (with Type Checking Mode: standard)
        extensions for assistance with linting and type checking during development.

6. Commit your changes and push your new branch to your fork.

7. Open a [pull request on GitHub](https://github.com/state-space-models/cuthbert/pulls).


## Adding an example

Examples in `cuthbert` are written in Markdown and tangled into Python scripts using [entangled](https://github.com/entangled/entangled.py) so they can be run as tests.

To add an example, you can use the following steps:

1. Write the example in Markdown in the `docs` directory.
2. Make sure to start each code block with `{.python #example_name-code_block_name}`. See the [`docs/quickstart.md`](https://github.com/state-space-models/cuthbert/blob/main/docs/quickstart.md) example for reference.  
(Note that it's important to include the example name here as code block names must be unique across all examples.)
3. Add a hidden code block at the end of the file with the following content:
```
```{.python file=examples_scripts/example_name.py}
<<example_name-code_block_name_1>>
<<example_name-code_block_name_2>>
...
```
Again, see the [`docs/quickstart.md`](https://github.com/state-space-models/cuthbert/blob/main/docs/quickstart.md) example for reference.

4. Add a reference to your new example in the [`docs/examples/index.md`](https://github.com/state-space-models/cuthbert/blob/main/docs/examples/index.md) file and the [`mkdocs.yml`](https://github.com/state-space-models/cuthbert/blob/main/mkdocs.yml) file.  
You may need to add any new dependencies to the [`pyproject.toml`](https://github.com/state-space-models/cuthbert/blob/main/pyproject.toml) file under `examples`.
5. Make sure your example ends with "Key Takeaways" and "Next Steps" sections,
linking to other documentation and examples where appropriate.
6. That's it!
If you want to generate the python script and run it locally to check it works, you can run:
```bash
entangled tangle
python examples_scripts/example_name.py
```
