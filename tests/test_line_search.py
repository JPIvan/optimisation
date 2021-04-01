import numpy as np
from pytest import approx, fixture, raises, mark

import context  # noqa
from src.line_search import LineSearch
from src.least_squares import least_squares
from src.wrappers import ObjectiveFunctionWrapper


@fixture
def quadratic_objective_1d():
    objective = ObjectiveFunctionWrapper(
        func=lambda x: (x - 4)**2,
        jac=lambda x: np.array(2*(x - 4), ndmin=2),
    )
    minimum = 4
    start_points = [3, 5]
    search_directions = [-objective.jac(x) for x in start_points]
    return objective, minimum, start_points, search_directions


@fixture
def division_by_zero_objective_1d():
    objective = ObjectiveFunctionWrapper(
        func=lambda x: 1/x,
        jac=lambda x: np.array(-1/(x**2), ndmin=2),
    )
    invalid_start = 0  # division by zero
    token_search_direction = 1  # dummy value - no real choice at start point
    return objective, invalid_start, token_search_direction


@fixture
def defined_only_on_part_of_domain_objective_1d():
    objective = ObjectiveFunctionWrapper(
        func=lambda x: x**2 if abs(x) < 1 else None,
        jac=lambda x: 2*x if abs(x) < 1 else None
    )
    invalid_start = 2  # invalid start
    token_search_direction = np.array(-4, ndmin=2)  # okay search direction
    return objective, invalid_start, token_search_direction


@fixture
def set_of_quadratic_objectives_nd():
    objectives, start_points, search_directions, true_mins = [], [], [], []
    rng = np.random.default_rng(seed=8008135)
    for n in range(2, 11):
        A = rng.uniform(low=-1, high=1, size=(n, n))
        b = rng.uniform(low=-1, high=1, size=(n, 1))
        LS = least_squares(A, b)

        def _deriv(x, A=A, b=b):   # analytical derivative
            return 2*A.T@A@x - 2*A.T@b

        objectives.append(ObjectiveFunctionWrapper(
            func=least_squares(A, b),
            jac=_deriv,
        ))
        start_points.append(rng.uniform(low=-1, high=1, size=(n, 1)))
        search_directions.append(-objectives[-1].jac(start_points[-1]))
        # derivative at last start point
        true_mins.append(LS.solve_minimum()['x*'])
    return objectives, start_points, search_directions, true_mins


class TestLineSearch:
    @mark.parametrize("method,assertion", [
            ("goldensection", "minimum_found"),
            ("backtracking", "function_decreased"),
        ])
    def test_linesearch_correct_1d(
            self, method, assertion, quadratic_objective_1d):
        """ Check if one dimensional problems which are well specified are
        solved correctly.
        """
        objective, minimum, start, searchdir = quadratic_objective_1d
        linesearch = LineSearch(objective)  # minimum at x = 4
        linesearch_method = getattr(linesearch, method)
        for x0, dx in zip(start, searchdir):
            solution = linesearch_method(
                x=x0,
                dx=dx,
            )
            if assertion == "minimum_found":
                assert solution.x == approx(minimum)
            elif assertion == "function_decreased":
                assert np.linalg.norm(solution.x-minimum) < 1
            else:
                raise ValueError("Bad assertion.")

    @mark.parametrize("method", ["goldensection", "backtracking"])
    def test_linesearch_bad_search_direction(
        self, method, quadratic_objective_1d
    ):
        """ Check that search fail explicitly when a bad search direction is
        given and a minimum cannot be bracketed.
        """
        objective, minimum, start, searchdir = quadratic_objective_1d
        linesearch = LineSearch(objective)
        linesearch_method = getattr(linesearch, method)
        for x0, dx in zip(start, searchdir):
            with raises(ValueError):
                linesearch_method(x=x0, dx=-dx)
                # give bad search direction; negative of the direction
                # defined in the fixture

    @mark.parametrize("method", ["goldensection", "backtracking"])
    def test_linesearch_division_by_zero_start(
        self, method, division_by_zero_objective_1d,
    ):
        """ Check behaviour when a start point that causes division by zero
        is given.

        Search should not fail silently, an explicit error is expected.
        """
        objective, invalid_start, searchdir = division_by_zero_objective_1d
        linesearch = LineSearch(objective)
        linesearch_method = getattr(linesearch, method)
        with raises(ZeroDivisionError):
            linesearch_method(x=invalid_start, dx=searchdir)

    @mark.parametrize("method", ["goldensection", "backtracking"])
    def test_linesearch_undefined_domain_start(
        self, method, defined_only_on_part_of_domain_objective_1d,
    ):
        """ Check behaviour when an undefined start point is given.

        Search should not fail silently, an explicit error is expected.
        """
        objective, invalid_start, searchdir = \
            defined_only_on_part_of_domain_objective_1d
        linesearch = LineSearch(objective)
        linesearch_method = getattr(linesearch, method)
        with raises((ValueError, TypeError, AttributeError)):
            linesearch_method(x=invalid_start, dx=searchdir)

    @mark.parametrize("method", ["goldensection", "backtracking"])
    def test_linesearch_correct_nd(
        self, method, set_of_quadratic_objectives_nd,
    ):
        """ Check if n-dimensional problems which are well specified are solved
        correctly.
        """
        objectives, start_points, search_directions, true_mins = \
            set_of_quadratic_objectives_nd
        for i, objective in enumerate(objectives):
            linesearch = LineSearch(objectives[i])
            linesearch_method = getattr(linesearch, method)
            solution = linesearch_method(
                x=start_points[i],
                dx=search_directions[i],
            )
            norm_start = np.linalg.norm(start_points[i] - true_mins[i])
            norm_after_linesearch = np.linalg.norm(solution.x - true_mins[i])
            assert norm_start > norm_after_linesearch
