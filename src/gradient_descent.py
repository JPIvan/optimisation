import numpy as np
from src import line_search
from src.result import OptimisationResult
from src.wrappers import ObjectiveFunctionWrapper


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
    objective = ObjectiveFunctionWrapper(func, jac)

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
