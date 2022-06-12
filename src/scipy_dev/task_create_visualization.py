import pytask
from scipy_dev.config import BLD
from scipy_dev.visualizations import create_gradient_descent_figure
from scipy_dev.visualizations import create_grid_search_figure


@pytask.mark.produces(BLD.joinpath("figures", "grid_search.png"))
def task_create_grid_search_figure(produces):
    fig, _ = create_grid_search_figure(contour_line_width=3)
    fig.savefig(produces)


@pytask.mark.produces(BLD.joinpath("figures", "gradient_descent.png"))
def task_create_gradient_descent_figure(produces):
    fig, _ = create_gradient_descent_figure(contour_line_width=3)
    fig.savefig(produces)
