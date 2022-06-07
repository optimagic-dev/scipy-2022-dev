# Scipy 2022 Tutorials

## Development Repository

Get started by installing the conda environment.

```bash
cd into project root
mamba env create -f environment.yml
conda activate scipy_dev
pip install -e .
```

## Presentation

Our presentation is compiled to HTML using [marp](https://marp.app/), which needs to be
installed and made available to the PATH. To get a PDF version that correctly displays
pauses/breaks, we utilize [decktape](https://github.com/astefanutti/decktape).

### Installation of marp and decktape

To check that both executables are installed run

```bash
$ pytask src/scipy_dev/task_test_installations.py
```

**Installation:**

- marp: Can be installed from the [README instructions](https://github.com/marp-team/marp-cli)
- decktape: Can be installed via ``npm install -g decktape``

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[cookiecutter-pytask-project](https://github.com/pytask-dev/cookiecutter-pytask-project)
template.
