import numpy as np
from src import line_search
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


def steepest_descent(
            func,
            x0,
            ls="golden-section",
            jac=None,
            stop="jac-norm",
            tol=1E-6,
            maxiter=1024,
        ):
    """ Perform a steepest descent search. See Boyd 9.3 for details.

    Args:
        func: function to be minimised
        x0: start point
        ls: line search method, currently supports:
            "golden-section" [DEFAULT]
            "backtracking" [UNTESTED]
        jac: function returning jacobian of func
            defaults to None
            will evaluate gradient numerically if none given
        stop: stopping criterion, currently suports:
            "jac-norm" [DEFAULT] - stops when ||jac(x)|| < tol
        tol: tolerance for stopping condition
            defaults to 1E-6
        maxiter: maximum number of iterations

    Returns:
        OptimisationResult
    """
    if jac is None:
        jac = _create_jac(func)
    _nfev, _njev = 0, 0

    def _f(x):  # should only be used locally, pass func to other functions
        nonlocal _nfev
        _nfev += 1
        return func(x)

    def _jac(x):  # should only be used locally, pass jac to other functions
        nonlocal _njev
        _njev += 1
        return jac(x)

    _x = x0  # search from given start point
    _dx = _jac(_x)  # calculate gradient for first iteration
    for niter in range(maxiter):
        searchdir = -_dx

        if ls == "golden-section":
            lsres = line_search.goldensection(func, _x, dx=searchdir)
        elif ls == "backtracking":
            raise NotImplementedError(
                "Usage of backtracking search not permitted as backtracking "
                "line search has not been tested."
            )
        else:
            raise ValueError(f"No such search method: \"{ls}\".")

        _nfev += lsres["nfev"]
        _njev += lsres["njev"]
        _x = lsres["x"]

        if stop == "jac-norm":
            _dx = _jac(_x)  # will use as search direction in next loop
            # saves jacobian evaluations
            if np.linalg.norm(_dx) < tol:
                break
        else:
            raise ValueError(f"No such stopping criterion: \"{stop}\".")
    else:
        return OptimisationResult(
            success=False,
            x=_x,
            niter=maxiter,
            nfev=_nfev,
            njev=_njev,
            info="Maximum number of iterations exceeded.",
            jac=_jac(_x)
        )

    return OptimisationResult(
            success=True,
            x=_x,
            niter=niter,
            nfev=_nfev,
            njev=_njev,
            jac=_jac(_x)
        )
