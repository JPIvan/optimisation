import numpy as np


class ObjectiveFunctionWrapper:
    """ Functions passed to the library for optimisation should be wrapped.
    This wrapper provides a basic count of the number of times the associated
    function has been called.
    """
    def __init__(self, func, jac=None, hes=None):
        self._f = func
        self._jac = self._create_jac(self.f) if jac is None else jac
        # creating the jac using self.f means that function calls are correctly
        # counted. call will be missed if a numerical jacobian is created
        # directly from func
        self._hes = hes
        self.nfev = 0
        self.njev = 0
        self.nhev = 0

    def f(self, x):
        self.nfev += 1
        return self._f(x)

    def jac(self, x):
        self.njev += 1
        return self._jac(x)

    def hes(self, x):
        self.nhev += 1
        return self._hes(x)

    def _create_jac(self, func):
        """ Create a function which returns the jacobian of func using the
        central difference method.

        Args:
            func: function taking column vector input and returning a single
                value

        Returns:
            _jac: a function with argument 'x' which returns the jacobian of
                func at x.
        """
        def _jac(x):
            """ Numerical derivative using central difference.
            x must be a column vector.
            """
            if x.shape[1] != 1:
                raise ValueError(f"x must be a column vector, got {x}")

            normx = np.linalg.norm(x)
            if normx == 0:
                delta = 1E-6  # prevent division by 0 later
            else:
                delta = 1E-6*normx  # scale step-size appropriately
            jacx = np.zeros_like(x, dtype=float)
            for i in range(x.shape[0]):
                dxi = np.zeros_like(x, dtype=float)
                dxi[i][0] = delta  # difference in only one variable
                jacx[i] = (func(x + dxi) - func(x - dxi)) / (2*delta)
            return jacx
        return _jac
