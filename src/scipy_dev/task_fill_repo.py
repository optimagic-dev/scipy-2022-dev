import re
import shutil
import time

import pytask
from scipy_dev.config import PUBLIC
from scipy_dev.config import ROOT
from scipy_dev.config import SRC


# ======================================================================================
# copy environment file
# ======================================================================================


@pytask.mark.depends_on(ROOT.joinpath("environment.yml"))
@pytask.mark.produces(
    {
        "unix": PUBLIC.joinpath("environment.yml"),
        "windows": PUBLIC.joinpath("environment_windows.yml"),
    }
)
def task_copy_environment_file(depends_on, produces):

    with open(depends_on) as f:
        lines = f.readlines()

    # change environment name
    lines[0] = "name: scipy_estimagic"

    # find and delete misc
    indices = [i for i, e in enumerate(lines) if "Misc" in e]
    del lines[indices[0] : indices[1] + 1]

    with open(produces["unix"], "w") as f:
        f.writelines(lines)

    # delete jax for windows
    lines = [line for line in lines if "jax" not in line]

    with open(produces["windows"], "w") as f:
        f.writelines(lines)


# ======================================================================================
# copy files from source repo
# ======================================================================================


dependencies = list(SRC.joinpath("source_repo").iterdir())

for dep in dependencies:

    kwargs = {
        "depends_on": dep,
        "produces": PUBLIC.joinpath(dep.name),
    }

    @pytask.mark.task(id=dep.name, kwargs=kwargs)
    def task_copy_public_directory_file(depends_on, produces):
        shutil.copyfile(depends_on, produces)
        time.sleep(0.1)  # otherwise pytask won't find the product


# ======================================================================================
# copy exercise notebooks
# ======================================================================================


dependencies = list(SRC.joinpath("notebooks").rglob("*"))
# delete checkpoint files
dependencies = [d for d in dependencies if ".ipynb_checkpoints" not in str(d)]
# only select notebooks
dependencies = [d for d in dependencies if d.suffix == ".ipynb"]
# match only the exercise and solution notebooks
dependencies = [d for d in dependencies if re.match("(^0[0-9])", d.name)]

for dep in dependencies:

    if "solutions" in str(dep):
        produces = PUBLIC.joinpath("exercises", "solutions", dep.name)
        _id = dep.name + "-solutions"
    else:
        produces = PUBLIC.joinpath("exercises", dep.name)
        _id = dep.name

    kwargs = {"depends_on": dep, "produces": produces}

    @pytask.mark.task(id=_id, kwargs=kwargs)
    def task_copy_notebooks(depends_on, produces):
        shutil.copyfile(depends_on, produces)
        time.sleep(0.1)  # otherwise pytask won't find the product
