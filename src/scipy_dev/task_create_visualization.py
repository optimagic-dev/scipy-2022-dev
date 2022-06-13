import pytask
from scipy_dev.config import BLD
from scipy_dev.visualizations import create_curse_of_dimensionality_figure
from scipy_dev.visualizations import create_gradient_descent_figure
from scipy_dev.visualizations import create_grid_search_figure


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
