import numpy as np
from src import line_search
from src.result import OptimisationResult
from src.wrappers import ObjectiveFunctionWrapper


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
            save_path=False,
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
        save_path: determines if intermediate solutions are saved
            False [DEFAULT] - will not create a list of solutions
            True - will create a list of solutions in result.solution_path

    Returns:
        OptimisationResult
    """
    objective = ObjectiveFunctionWrapper(
        func,
        _create_jac(func) if jac is None else jac,
    )

    _x = x0  # search from given start point
    _dx = objective.jac(_x)  # calculate gradient for first iteration
    if save_path:
        solution_path = []  # intermediate solution will be saved here
    else:
        solution_path = None  # default value expected by result
    for niter in range(maxiter):
        if save_path:
            solution_path.append(_x)

        searchdir = -_dx

        if ls == "golden-section":
            lsres = line_search.goldensection(objective.f, _x, dx=searchdir)
        elif ls == "backtracking":
            raise NotImplementedError(
                "Usage of backtracking search not permitted as backtracking "
                "line search has not been tested."
            )
        else:
            raise ValueError(f"No such search method: \"{ls}\".")

        _x = lsres.x

        if stop == "jac-norm":
            _dx = objective.jac(_x)  # save search direction for next loop
            # saves jacobian evaluations
            if np.linalg.norm(_dx) < tol:
                break
        else:
            raise ValueError(f"No such stopping criterion: \"{stop}\".")
    else:
        return OptimisationResult(
            success=False,
            x=_x,
            jac=objective.jac(_x),
            niter=maxiter,
            nfev=objective.nfev,
            njev=objective.njev,
            info="Maximum number of iterations exceeded.",
            path=solution_path,
        )

    return OptimisationResult(
            success=True,
            x=_x,
            jac=objective.jac(_x),
            niter=niter,
            nfev=objective.nfev,
            njev=objective.njev,
            path=solution_path,
        )
