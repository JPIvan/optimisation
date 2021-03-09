from matplotlib import pyplot as plt
import numpy as np


def contourf(xlim, ylim, func, gridpoints=51, contours=20):
    """ Calls the matplotlib contourf function with most of the implementation
    hidden from the caller.

    Args:
        xlim: bounds on x-axis
        ylim: bounds on y-axis
        func: function to plot, must be a callable compatible with f([x, y])
        gridpoints: number of points to evaluate each variable
        contours:
            int: number of contour lines to draw
            array-like: levels at which contours are drawn
    Returns:
        fig, ax: The created figure and axes instances.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    _X = np.linspace(xlim[0], xlim[1], gridpoints)
    _Y = np.linspace(ylim[0], ylim[1], gridpoints)
    _X, _Y = np.meshgrid(_X, _Y)
    _F = []
    for x, y in zip(_X.flatten(), _Y.flatten()):
        input_vector = np.array([x, y]).reshape(2, 1)
        _F.append(func(input_vector))
    _F = np.array(_F).reshape(_X.shape)
    ax.contourf(_X, _Y, _F, contours)
    return fig, ax
