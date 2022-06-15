import pytask
import numpy as np
from scipy_dev.config import BLD
from scipy_dev.visualizations import create_curse_of_dimensionality_figure
from scipy_dev.visualizations import create_gradient_descent_figure
from scipy_dev.visualizations import create_grid_search_figure
from scipy_dev.visualizations import plot_function_3d

@pytask.mark.produces(BLD.joinpath("figures", "grid_search.png"))
def task_create_grid_search_figure(produces):
    fig, _ = create_grid_search_figure(contour_line_width=3)
    fig.tight_layout()
    fig.savefig(produces)


@pytask.mark.produces(BLD.joinpath("figures", "gradient_descent.png"))
def task_create_gradient_descent_figure(produces):
    fig, _ = create_gradient_descent_figure(contour_line_width=3)
    fig.tight_layout()
    fig.savefig(produces)


@pytask.mark.produces(BLD.joinpath("figures", "curse_of_dimensionality.png"))
def task_create_curse_of_dimensionality_figure(produces):
    fig, _ = create_curse_of_dimensionality_figure(marker_size=500)
    fig.savefig(produces)



def alpine(x):
    x = (x+0.5) * 5
    out = -np.prod(np.sqrt(x) * np.sin(x))
    return out

def ackley(x, a=20, b=0.2, c=2 * np.pi):
    x = (x - 0.5) * 32
    temp = -a * np.exp(-b * np.sqrt(np.mean(x ** 2)))
    out = temp - np.exp(np.mean(np.cos(c * x))) + a + np.exp(1)
    return out

def sphere(x):
    x = (x - 0.5) * 5.12
    out = np.sum(x ** 2)
    return out

parametrization = []
for func in [alpine, ackley, sphere]:
    target = BLD.joinpath("figures", f"{func.__name__}.png")
    parametrization.append((target, func))

@pytask.mark.parametrize("produces, func", parametrization)
def task_create_3d_plots(produces, func):
    fig = plot_function_3d(func, -0.5, 1.5, 50)
    fig.tight_layout()
    fig.savefig(produces)