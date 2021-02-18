from matplotlib import pyplot as plt
import numpy as np


def contourf(xlim, ylim, func, gridpoints=51, ncontours=20):
    """ Calls the matplotlib contourf function with most of the implementation
    hidden from the caller.

    Args:
        xlim: bounds on x-axis
        ylim: bounds on y-axis
        func: function to plot, must be a callable compatible with f([x, y])
        gridpoints: number of points to evaluate each variable
        ncontours: number of contour lines to draw
    Returns:
        fig, ax: The created figure and axes instances.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    _X = np.linspace(xlim[0], xlim[1], gridpoints)
    _Y = np.linspace(ylim[0], ylim[1], gridpoints)
    _X, _Y = np.meshgrid(_X, _Y)
    _F = []
    for x, y in zip(_X.flatten(), _Y.flatten()):
        _F.append(func([x, y]))
    _F = np.array(_F).reshape(_X.shape)
    ax.contourf(_X, _Y, _F, ncontours)
    return fig, ax
