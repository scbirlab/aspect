# 🧐 aspect

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/aspect/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aspect)
![PyPI](https://img.shields.io/pypi/v/aspect)

**aspect** is a suite of python tools.

- [Installation](#installation)
- [Command-line interface](#command-line-interface)
- [Python API](#python-api)
- [Issues, problems, suggestions](#issues-problems-suggestions)
- [Documentation](#documentation)

## Installation

### The easy way

You can install the precompiled version directly using `pip`.

```bash
$ pip install aspect
```

If you want to use aspect for chemistry machine learning and AI, use:

```bash
$ pip install aspect[chem]
```

For integrating taxonomic information with [vectome](https://github.com/scbirlab/vectome), use:

```bash
$ pip install aspect[bio]
```

You can install both:

```bash
$ pip install aspect[bio,chem]
```

### From source

Clone the repository, then `cd` into it. Then run:

```bash
$ pip install -e .
```

## Command-line interface

**aspect** has a command-line interface for training and checkpointing the built-in models. 

```bash
$ aspect --help
```

In all cases, you can get further options with `aspect <command> --help`, for example:

```bash
aspect train --help
```

## Python API

You can import **aspect** for your own programs.

```python
import aspect as asp
```

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/scbirlab/aspect/issues).

## Documentation

(To come at [ReadTheDocs](https://aspect.readthedocs.org).)
