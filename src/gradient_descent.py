import numpy as np
import src.line_search
from src.result import OptimisationResult


def _create_jac(func):
    """ Create a function which returns the jacobian of func using the
    central difference method.

    Args:
        func: function taking column vector input and returning a single value

    Returns:
        _jac: a function with argument 'x' which returns the jacobian of func
            at x.
    """
    def _jac(x):
        """ Numerical derivative using central difference.
        x must be a column vector.
        """
        if x.shape[1] != 1:
            raise ValueError(f"x must be a column vector, got {x}")

        normx = np.linalg.norm(x)
        if normx == 0:
            delta = 1E-4  # prevent division by 0 later
        else:
            delta = 1E-4*normx  # scale step-size appropriately
        jacx = np.zeros_like(x, dtype=float)
        for i in range(x.shape[0]):
            dxi = np.zeros_like(x, dtype=float)
            dxi[i][0] = delta  # difference in only one variable
            jacx[i] = (func(x + dxi) - func(x - dxi)) / (2*delta)
        return jacx
    return _jac

