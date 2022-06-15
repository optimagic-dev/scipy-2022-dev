import pytask
import numpy as np
from scipy_dev.config import BLD
from scipy_dev.config import SRC
from scipy_dev.visualizations import create_curse_of_dimensionality_figure
from scipy_dev.visualizations import create_gradient_descent_figure
from scipy_dev.visualizations import create_grid_search_figure
from scipy_dev.visualizations import plot_function_3d


@pytask.mark.depends_on(SRC.joinpath("visualizations.py"))
@pytask.mark.produces(BLD.joinpath("figures", "grid_search.png"))
def task_create_grid_search_figure(produces):
    fig, _ = create_grid_search_figure(contour_line_width=3)
    fig.tight_layout()
    fig.savefig(produces, bbox_inches="tight")


@pytask.mark.depends_on(SRC.joinpath("visualizations.py"))
@pytask.mark.produces(BLD.joinpath("figures", "gradient_descent.png"))
def task_create_gradient_descent_figure(produces):
    fig, _ = create_gradient_descent_figure(contour_line_width=3)
    fig.tight_layout()
    fig.savefig(produces)


@pytask.mark.parametrize(
    "produces, depends_on, orientation",
    [
        (
            BLD.joinpath("figures", f"curse_of_dimensionality_{orient}.png"),
            SRC.joinpath("visualizations.py"),
            orient,
        )
        for orient in ["h", "v"]
    ],
)
def task_create_curse_of_dimensionality_figure(produces, orientation):
    fig, _ = create_curse_of_dimensionality_figure(
        marker_size=500, orientation=orientation
    )
    fig.tight_layout()
    fig.savefig(
        produces, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=True
    )


def alpine(x):
    out = -np.prod(np.sqrt(x) * np.sin(x))
    return out


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    temp = -a * np.exp(-b * np.sqrt(np.mean(x ** 2)))
    out = temp - np.exp(np.mean(np.cos(c * x))) + a + np.exp(1)
    return out


def sphere(x):
    out = np.sum(x ** 2)
    return out


inputs = {
    "alpine": {"func": alpine, "bounds": [0, 10]},
    "ackley": {"func": ackley, "bounds": [-32, 32]},
    "sphere": {"func": sphere, "bounds": [-5.12, 5.12]},
}

parametrization = []
for k, v in inputs.items():
    target = BLD.joinpath("figures", f"{k}.png")
    parametrization.append((target, v["func"], v["bounds"]))


@pytask.mark.parametrize("produces, func, bounds", parametrization)
def task_create_3d_plots(produces, func, bounds):
    fig = plot_function_3d(func, bounds[0], bounds[1], 50)
    fig.tight_layout()
    fig.savefig(produces, bbox_inches="tight")
