import pytask
import yaml
from scipy_dev.config import BLD
from scipy_dev.config import SRC


DEPENDENCIES = [SRC.joinpath("presentation", f) for f in ["main.md", "structure.yaml"]]
DEPENDENCIES += list(SRC.joinpath("presentation", "sections").iterdir())
DEPENDENCIES = {dep.name: dep for dep in DEPENDENCIES}


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(BLD.joinpath("presentation", "main.md"))
def task_combine_sections(produces, depends_on):

    with open(depends_on["structure.yaml"]) as f:
        structure = yaml.safe_load(f)

    with open(depends_on["main.md"]) as f:
        file_contents = f.readlines()

    for section in structure["index"]:

        with open(depends_on[section]) as f:
            _section_contents = f.readlines()
            file_contents += _section_contents

    with open(produces, "w") as f:
        f.writelines(file_contents)
