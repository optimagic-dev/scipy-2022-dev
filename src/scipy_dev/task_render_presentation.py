import pytask
import subprocess

from scipy_dev.config import BLD
from scipy_dev.config import SRC


CSS_PATH = SRC.joinpath("presentation", "custom.css").resolve()


main_files = ["main"]

for file in main_files:

    for output_format in ["pdf", "html"]:

        kwargs = {
            "depends_on": SRC.joinpath("presentation", f"{file}.md"),
            "produces": BLD.joinpath("public", "presentation", f"{file}.{output_format}"),
        }

        @pytask.mark.task(id=f"{file}-{output_format}", kwargs=kwargs)
        def task_render_presentation(depends_on, produces):

            commands = [
                "marp",  # executable
                "--html",  # allows html code in markdown files
                "--theme-set", str(CSS_PATH),  # use custom css file
                "--output", str(produces),  # output file
                # meta data
                "--title", "Scipy 2022: Estimagic Tutorial",
                "--author", "Janos Gabler and Tim Mensinger",
            ]

            if "pdf" in produces.suffix:
                commands.append("--pdf")

            commands += ["--", depends_on]  # source file

            subprocess.call(commands)
