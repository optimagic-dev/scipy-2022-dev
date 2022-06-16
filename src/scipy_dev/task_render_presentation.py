import subprocess

import pytask
from scipy_dev.config import BLD
from scipy_dev.config import PUBLIC
from scipy_dev.config import SRC


GRAPHS = SRC.joinpath("graphs")
FIGURES = BLD.joinpath("figures")


src_figures = [
    "janos.jpg",
    "tim.jpeg",
    "klara.jpg",
    "sebi.jpg",
    "tobi.png",
    "hmg.jpg",
    "benchmark.png",
    "convergence_plot.png",
    "scaling_scipy_lbfgsb.png",
    "scaling_fides.png",
    "scaling_nag_dfols.png",
    "scaling_nlopt_bobyqa.png",
]

bld_figures = [
    "sphere.png",
    "grid_search.png",
    "gradient_descent.png",
    "curse_of_dimensionality_v.png",
    "criterion_plot.png",
    "params_plot.png",
    "alpine.png",
    "ackley.png",
]

src_figures = [SRC.joinpath("graphs", f) for f in src_figures]
bld_figures = [BLD.joinpath("figures", f) for f in bld_figures]

dependencies = {f.name: f for f in src_figures + bld_figures}

for output_format in ["pdf"]:

    kwargs = {
        "depends_on": {
            **dependencies,
            **{
                "source": SRC.joinpath("presentation", "main.md"),
                "scss": SRC.joinpath("presentation", "custom.scss").resolve(),
            },
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
