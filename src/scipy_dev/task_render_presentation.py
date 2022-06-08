import subprocess

import pytask
from scipy_dev.config import BLD
from scipy_dev.config import SRC


main_files = ["main"]

for file in main_files:

    for output_format in ["html"]:

        kwargs = {
            "depends_on": {
                "source": BLD.joinpath("presentation", f"{file}.md"),
                "css": SRC.joinpath("presentation", "custom.css").resolve(),
            },
            "produces": BLD.joinpath(
                "public", "presentation", f"{file}.{output_format}"
            ),
        }

        @pytask.mark.task(id=f"{file}-{output_format}", kwargs=kwargs)
        def task_render_presentation(depends_on, produces):

            commands = [
                "marp",  # executable
                "--html",  # allows html code in markdown files
                "--allow-local-files",
                "--theme-set",
                str(depends_on["css"]),  # use custom css file
                "--output",
                str(produces),  # output file
            ]

            commands += ["--", depends_on["source"]]

            subprocess.call(commands)
