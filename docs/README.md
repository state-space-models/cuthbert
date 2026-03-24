# Documentation Website

The documentation site is built with [Zensical](https://zensical.org/).
For local development, install the required dependencies with

```bash
pip install -e ".[docs]"
```

To preview the website, use

```bash
zensical serve
```

The live preview should be available at `http://localhost:8000`.

To build the website for deployment, use

```bash
zensical build
```

## Reused Markdown (Snippets)

Pages under `docs/` often pull shared copy from repository `README.md` files via
[PyMdown Snippets](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/)
(`--8<-- "path"` and named sections in HTML comments). See that guide for syntax and
options.
