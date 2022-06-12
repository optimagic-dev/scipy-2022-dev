import matplotlib.pyplot as plt
import numpy as np


def create_grid_search_figure(contour_line_width=3):
    fig, ax = _get_contour_figure(contour_line_width)
    fig, ax = _add_grid(fig, ax)
    return fig, ax


def create_gradient_descent_figure(
    contour_line_width=3, arrowstyle="simple", head_width=0.4, tail_width=0.1
):
    arrowstyle = f"{arrowstyle}, head_width={head_width}, tail_width={tail_width}"
    fig, ax = _get_contour_figure(contour_line_width)
    fig, ax = _add_gradient_descent_path(fig, ax, arrowstyle)
    return fig, ax


def _get_contour_figure(contour_line_width):

    # data for contour lines
    grid = np.linspace(-0.1, 0.1, num=100)
    x, y = np.meshgrid(grid, grid)
    z = x**2 + y**2  # sphere

    # figure
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.contour(
        x,
        y,
        z,
        linewidths=contour_line_width,
        levels=[0.0001, 0.0007, 0.0022, 0.005, 0.009, 0.015],
        zorder=1,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def _add_grid(fig, ax):

    # data for dots
    grid = np.linspace(-0.09, 0.09, num=10)
    x, y = np.meshgrid(grid, grid)

    ax.scatter(x, y, color="grey", s=25, zorder=0)
    return fig, ax


def _add_gradient_descent_path(fig, ax, arrowstyle):

    # data for path and dots
    grid = np.linspace(-0.09, 0.09, num=10)
    x, y = np.meshgrid(grid, grid)

    kwargs = {
        "arrowprops": {"arrowstyle": arrowstyle, "color": "black"},
        "fontsize": 20,
    }

    # path
    path = [(x[0][1], y[8][0]), (x[0][2], y[2][0]), (x[5][4], y[5][5]), (0, 0)]

    for k in range(len(path) - 1):
        _from = path[k]
        _to = path[k + 1]
        ax.annotate("", xy=_to, xytext=_from, **kwargs)

    # dots
    ax.scatter(x=[p[0] for p in path], y=[p[1] for p in path], color="gray")
    return fig, ax
