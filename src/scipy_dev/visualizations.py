import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from estimagic import criterion_plot
from estimagic import minimize
from estimagic import params_plot


def create_criterion_plot(kwargs):
    res = _get_optimize_result_for_criterion_and_profile_plot()
    fig = criterion_plot(res, **kwargs)
    return fig


def create_params_plot(kwargs):
    res = _get_optimize_result_for_criterion_and_profile_plot()
    fig = params_plot(res, max_evaluations=300, **kwargs)
    return fig


def create_grid_search_figure(contour_line_width=3, cmap="Blues"):
    fig, ax = _get_contour_figure(contour_line_width, cmap=cmap)
    fig, ax = _add_grid(fig, ax)
    return fig, ax


def create_gradient_descent_figure(
    contour_line_width=3,
    arrowstyle="simple",
    head_width=0.4,
    tail_width=0.1,
    cmap="Blues",
):
    arrowstyle = f"{arrowstyle}, head_width={head_width}, tail_width={tail_width}"
    fig, ax = _get_contour_figure(contour_line_width, cmap=cmap)
    fig, ax = _add_gradient_descent_path(fig, ax, arrowstyle)
    return fig, ax


def create_curse_of_dimensionality_figure(
    figsize=None, marker_size=500, orientation="h"
):
    if figsize is None:
        if orientation == "h":
            figsize = (17, 8)
        else:
            figsize = (10, 12)
    fig = plt.figure()
    if orientation == "h":
        for dimension in (1, 2, 3):
            ax = fig.add_subplot(1, 3, dimension, projection="3d")
            _plot_curse_of_dimensionality_dimension(
                dimension, ax, marker_size=marker_size
            )
        fig.set_size_inches(*figsize)
    elif orientation == "v":
        for dimension in (1, 2, 3):
            ax = fig.add_subplot(3, 1, dimension, projection="3d")
            _plot_curse_of_dimensionality_dimension(
                dimension, ax, marker_size=marker_size
            )
        fig.set_size_inches(*figsize)

    fig.subplots_adjust(wspace=0, hspace=0)

    return fig, ax


def _get_contour_figure(contour_line_width, cmap):

    # data for contour lines
    grid = np.linspace(-0.1, 0.1, num=100)
    x, y = np.meshgrid(grid, grid)
    z = np.sqrt(x**2 + y**2)  # sphere

    # figure
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.contourf(
        x,
        y,
        z,
        linewidths=contour_line_width,
        zorder=1,
        alpha=0.5,
        cmap=cmap,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def _add_grid(fig, ax):

    # data for dots
    grid = np.linspace(-0.09, 0.09, num=10)
    x, y = np.meshgrid(grid, grid)

    ax.scatter(x, y, color="black", s=40, alpha=1, zorder=0)
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
    ax.scatter(x=[p[0] for p in path], y=[p[1] for p in path], color="dimgrey", s=50)
    return fig, ax


def _plot_curse_of_dimensionality_dimension(dimension, ax, marker_size):
    points = _create_points(dimension)
    alphas = _create_alphas(points, dimension)

    for alpha, p in zip(alphas, points):
        ax.scatter(*p, s=marker_size, color="tab:blue", alpha=alpha)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.axes.set_xlim3d(left=-0.1, right=1.1)
    ax.axes.set_ylim3d(bottom=-0.1, top=1.1)
    ax.axes.set_zlim3d(bottom=-0.1, top=1.1)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def _create_points(dimension):
    """Create points for curse of dimensionality plot."""
    grid = [0, 1 / 3, 2 / 3, 1]
    if dimension == 1:
        points = []
        for p in grid:
            points.append((1, p, 0))
    elif dimension == 2:
        points = []
        for p in grid:
            for q in grid:
                points.append((q, p, 0))
    elif dimension == 3:
        points = []
        for p in grid:
            for p in grid:
                for q in grid:
                    for g in grid:
                        points.append((q, p, g))
    return points


def _create_alphas(points, dimension):
    """Create alpha values such that closer markers are darker."""
    alphas = []
    for p in points:
        if dimension in (1, 2):
            alpha = np.exp(-0.4 * np.linalg.norm(np.array(p) - np.array([1, 0, 0])))
        else:
            candidates = np.exp(
                -1.5
                * np.linalg.norm(
                    np.array(p) - np.array([[1, 0, g] for g in (0, 1 / 3, 2 / 3, 1)]),
                    axis=1,
                )
            )
            alpha = candidates.max()
        alphas.append(alpha)
    return alphas


def plot_function_3d(
    func, lower_bound, upper_bound, n_gridpoints, cmap="coolwarm", figsize=(8, 8)
):
    grid = np.linspace(lower_bound, upper_bound, n_gridpoints)
    x_mesh, y_mesh = np.meshgrid(grid, grid)
    results = []
    for x, y in zip(x_mesh.flatten(), y_mesh.flatten()):
        results.append(func(np.array([x, y])))
    z_mesh = np.array(results).reshape(x_mesh.shape)

    fig = plt.figure()
    fig.set_size_inches(*figsize)
    ax = plt.axes(projection="3d")
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=cmap)
    ax.contour(x_mesh, y_mesh, z_mesh, levels=30, offset=np.min(z_mesh), cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    return fig


def plot_contour_2d(func, lower_bound, upper_bound, n_gridpoints):
    grid = np.linspace(lower_bound, upper_bound, n_gridpoints)
    x_mesh, y_mesh = np.meshgrid(grid, grid)
    results = []
    for x, y in zip(x_mesh.flatten(), y_mesh.flatten()):
        results.append(func(np.array([x, y])))
    z_mesh = np.array(results).reshape(x_mesh.shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(x_mesh, y_mesh, z_mesh, levels=30, cmap="coolwarm", alpha=0.7)
    # add labels and set equal aspect ratio
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect(aspect="equal")
    return fig


def _get_optimize_result_for_criterion_and_profile_plot():
    def dict_sphere(params):
        return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()

    res = minimize(
        criterion=dict_sphere,
        params={"a": 0, "b": 1, "c": pd.Series([2, 3, 4])},
        algorithm="scipy_neldermead",
    )

    return res
