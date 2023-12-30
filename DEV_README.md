## Magicsoup

This is a hybrid python-rust package using [Maturin/pyo3](https://github.com/PyO3/maturin).
The setup based on [this minimal example](https://vordeck.de/kn/python-rust-package).
Heavy calculations are done in [PyTorch](https://pytorch.org/),
for lots of string processing rust with [rayon](https://docs.rs/rayon/latest/rayon/) should be used.
Everything should have type hints and docstrings for nice developer experience.
Rust functions should be wrapped in typed python functions/methods.
Tests are all in python with a [PyTest](https://docs.pytest.org/en/7.4.x/) suite.
A RTD documentation is built using [MkDocs](https://www.mkdocs.org/).

- [python/](./python/) python package
- [rust/](./rust/) rust library
- [tests/](./tests/) test suite
- [scripts/](./scripts/) some bash scripts for development
- [performance/](./performance/) measuring performance
- [docs/](./docs/) markdown files and visualizations for documentation
- [.github/](./.github/) workflow that builds, tests (fast), releases

### Developing in rust

```
maturin develop  # quick build, creates python/magicsoup/_lib.*.so
maturin develop --release  # build with performance optimizations
```

### Testing

```
bash scripts/test.sh tests/fast  # only quick tests
bash scripts/test.sh tests/  # all tests
```

There is [docs/create_figures.py](./docs/create_figures.py) which creates a lot of plots.
Some of those plots also serve as sanity checks.
_E.g._ reaction kinetics and biochemical patterns should still make sense,
overall energy has to eventually decrease during a simulation, etc.
They should be checked before major updates.
Also run [scripts/check-performance.sh](./scripts/check-performance.sh) for checking
performance of all major simulation steps.
For performance-related updates check this on CPU and GPU
and also run [performance/run.py](./performance/run.py) and look at the times with _Tensorboard_.

### Release

```
bash scripts/release.sh  # creates a tag and pushes it
```

Update version in [pyproject.toml](./pyproject.toml) first.
If rust files changed also update [Cargo.toml](./Cargo.toml) to the same version.
The build pipeline in [.github/](./.github/) will be triggered by pushing the version tag.

### Documentation

```
bash scripts/serve-docs.sh  # builds and serves documentation
```

Markdown files and images are in [docs/](./docs/).
Docstrings from python are parsed and mounted in [docs/reference.md](./docs/reference.md).
This is configured in [mkdocs.yml](./mkdocs.yml).
Note that [docs/index.md](./docs/index.md) is the index for RTD
and [README.md](./README.md) is the start page on PyPI.
Docs are built for RTD on every push to main.
See if [docs/create_figures.py](./docs/create_figures.py) or other plots should be recreated.
