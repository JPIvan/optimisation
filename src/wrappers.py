class ObjectiveFunctionWrapper:
    """ Functions passed to the library for optimisation should be wrapped.
    This wrapper provides a basic count of the number of times the associated
    function has been called.
    """
    def __init__(self, func, jac=None, hes=None):
        self._f = func
        self._jac = jac
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
