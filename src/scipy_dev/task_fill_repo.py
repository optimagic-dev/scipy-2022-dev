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
