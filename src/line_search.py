import numpy as np


def goldensection(func, x, dx, precision=1e-6):
    """ Golden-section search. Assumes minimum is in the direction dx,
    starting at x.

    Args:
        func: function being minimised
        x: start point
        dx: search direction
        precision: uncertainty permitted in result

    Returns:
        't': argmin_s f(x + s*dx)
        'x': optimal x found by golden section search
        some other metadata
    """
    iphi = (5**0.5 - 1) / 2  # 1/phi
    iphi2 = (3 - 5**0.5) / 2  # 1/phi^2
    nfev = 0

    def _f(x):
        nonlocal nfev
        nfev += 1
        return func(x)

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
            fx2 = _f(x2)
        if fx3 is None:
            fx3 = _f(x3)

        if fx2 < fx3:
            return _gs(x1, x3, h=h*iphi, x3=x2, fx3=fx2)
        else:
            return _gs(x2, x4, h=h*iphi, x2=x3, fx2=fx3)

    # first we need to bracket the minimum
    t = 1
    for _ in range(64):
        if _f(x + t*dx) > _f(x):
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

    return {
        'x': xopt,
        't': np.average((xopt - x) / dx),  # x* = x + t*dx
        'nfev': nfev,
    }


def backtracking(func, jac, x, dx, alpha=0.3, beta=0.8):
    """ Golden-section search. Assumes minimum is in the direction dx,
    starting at x.

    Args:
        func: function being minimised
        x: start point
        dx: search direction
        alpha: proportion of decrease expected when compared to linear
            extrapolation must be in (0, 0.5), (0.01, 0.3) recommended
        beta: 'crudeness' of search. Values closer to 1 check more
            points. (0.1, 0.8) recommended

    Returns:
        't': argmin_s f(x + s*dx)
        'x': optimal x
        some other metadata
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

    nfev = 0
    njacev = 0

    def _f(x):
        nonlocal nfev
        nfev += 1
        return func(x)

    def _jac(x):
        nonlocal njacev
        njacev += 1
        return jac(x)

    t = 1
    while _f(x + t*dx) > _f(x) + alpha*t*_jac(x)*dx:
        t *= beta
    return {
        'x': x + t*dx,
        't': t,
        'nfev': nfev,
        'njacev': njacev,
    }
