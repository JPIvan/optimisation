import numpy as np
from src.result import LineSearchResult
from src.wrappers import ObjectiveFunctionWrapper


class LineSearch:
    """ Class for performing line search.

    Examples:
    >>> objective = ObjectiveFunctionWrapper(func, jac, hess)
    >>> ls = LineSearch(objective)
    >>> lsresult = ls.backtracking(x=x0, dx=dx)
    """
    def __init__(self, objective):
        if isinstance(objective, ObjectiveFunctionWrapper):
            self._objective = objective
        elif callable(objective):
            self._objective = ObjectiveFunctionWrapper(objective)
            # attempt to make objective wrapper
        else:
            raise TypeError(
                f"Objective must be callable, got {type(objective)}"
            )

    def goldensection(self, x, dx, precision=1e-6):
        """ Golden-section search. Assumes minimum is in the direction dx,
        starting at x.

        Args:
            x: start point
            dx: search direction
            precision: uncertainty permitted in result

        Returns:
            LineSearchResult
        """
        iphi = (5**0.5 - 1) / 2  # 1/phi
        iphi2 = (3 - 5**0.5) / 2  # 1/phi^2

        def _gs(x1, x4, h=None, x2=None, x3=None, fx2=None, fx3=None):
            # We are going to divide the search space into three sections with
            # boundaries (x1, x2, x3, x4) and perform golden section search.
            # Function values are saved between iterations.
            if h is None:
                h = x4 - x1
            if np.linalg.norm(h) <= precision:
                return (x1 + x4) / 2
            if x2 is None:
                x2 = x1 + iphi2*h
            if x3 is None:
                x3 = x1 + iphi*h
            if fx2 is None:
                fx2 = self._objective.f(x2)
            if fx3 is None:
                fx3 = self._objective.f(x3)

            if fx2 < fx3:
                return _gs(x1, x3, h=h*iphi, x3=x2, fx3=fx2)
            else:
                return _gs(x2, x4, h=h*iphi, x2=x3, fx2=fx3)

        # check search direction
        for n in range(6):
            if self._objective.f(x + dx*10**-n) < self._objective.f(x):
                break
        else:  # did not find any step size for which function decreases.
            raise ValueError(
                "Function does not appear to decrease in search direction. "
                "Check if func is convex, start point, and search direction."
            )

        # first we need to bracket the minimum
        t = 1
        for _ in range(64):
            if self._objective.f(x + t*dx) > self._objective.f(x):
                break
            t *= 2
        else:
            raise RuntimeError(
                "Bracketing minimum failed. "
                "Check if func is convex, start point, and search direction."
            )
        # minimum is now definitely between f(x + t*dx) and f(x)
        # do golden section search
        xopt = _gs(x, x + t*dx)  # golden section actually finds the optimal x

        return LineSearchResult(
            success=True,
            x=xopt,
            t=np.average((xopt - x) / dx),  # x* = x + t*dx
        )

    def backtracking(self, x, dx, alpha=0.3, beta=0.8, maxiter=100):
        """ Backtracking search. Assumes minimum is in the direction dx,
        starting at x.

        Args:
            x: start point
            dx: search direction
            alpha: proportion of decrease expected when compared to linear
                extrapolation must be in (0, 0.5), (0.01, 0.3) recommended
            beta: 'crudeness' of search. Values closer to 1 check more
                points. (0.1, 0.8) recommended

        Returns:
            LineSearchResult
        """
        if alpha <= 0 or alpha >= 0.5:
            raise ValueError(
                f"Bounds on alpha not respected! Must be within (0, 0.5) but"
                f" {alpha} was given."
            )
        if beta <= 0 or beta >= 1:
            ValueError(
                f"Bounds on beta not respected! Must be within (0, 1) but"
                f" {beta} was given."
            )

        t = 1
        for _ in range(maxiter):
            if self._objective.f(x + t*dx) > \
                    self._objective.f(x) + alpha*t*self._objective.jac(x).T@dx:
                t *= beta
            else:
                break
        else:
            raise ValueError(
                f"Function does not appear to decrease in given search "
                f"direction."
                f"\nDirection: {dx}"
            )
        return LineSearchResult(
            success=True,
            x=x + t*dx,
            t=t,
        )
