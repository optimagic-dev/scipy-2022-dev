# Scipy 2022 Tutorials

## Development Repository

Get started by installing the conda environment.

```bash
cd into project root
mamba env create -f environment.yml
conda activate scipy_dev
```

## Compiling the Presentation

The presentation is compiled to HTML using [marp](https://marp.app/), which needs to be
installed and made available to the PATH. To get a PDF version that correctly displays
pauses/breaks, we utilize [decktape](https://github.com/astefanutti/decktape).

To check that both executables are installed run

```bash
pytask
```

### Decktape

```bash
$ decktape generic -s 1280x720 --load-pause 3000 file.html file.pdf
```

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[cookiecutter-pytask-project](https://github.com/pytask-dev/cookiecutter-pytask-project)
template.
