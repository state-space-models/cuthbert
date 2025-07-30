# Documentation Website

The documentation website is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
For local development, install the required dependencies with

```bash
pip install -e ".[docs]"
```

To preview the website, use

```bash
mkdocs serve
```

The live preview should be available at `http://127.0.0.1:8000/cuthbert/`.

To build the website for deployment, use

```bash
mkdocs build
```
