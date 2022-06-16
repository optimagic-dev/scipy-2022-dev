import subprocess

import pytask
from scipy_dev.config import PUBLIC
from scipy_dev.config import SRC


for output_format in ["pdf"]:

    kwargs = {
        "depends_on": {
            "source": SRC.joinpath("presentation", "main.md"),
            "scss": SRC.joinpath("presentation", "custom.scss").resolve(),
        },
        "produces": PUBLIC.joinpath(f"slides.{output_format}"),
    }

    @pytask.mark.task(id=f"slides-{output_format}", kwargs=kwargs)
    def task_render_presentation(depends_on, produces):

        commands = [
            "marp",  # executable
            "--html",  # allows html code in markdown files
            "--allow-local-files",
            "--theme-set",
            str(depends_on["scss"]),  # use custom scss file
            "--output",
            str(produces),  # output file
        ]

        commands += ["--", depends_on["source"]]

        subprocess.call(commands)
