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


class TestGoldenSection:
    def test_correct_nd(self):
        """ Check if n-dimensional problems which are well specified are solved
        correctly.
        """
        for n in range(2, 10):
            A = np.random.uniform(
                low=-1,
                high=1,
                size=(n, n)
            )
            b = np.random.uniform(
                low=-1,
                high=1,
                size=(n, 1)
            )
            LS = least_squares(A, b)  # Least squares instance ||Ax-b||^2

            x0 = np.random.uniform(
                low=-1,
                high=1,
                size=(n, 1)
            )  # pick random starting point
            dx = 2*A.T@A@x0 - 2*A.T@b  # derivative of x^TA^TAx-2x^TA^Tb-b^Tb

            linesearch = LineSearch(LS)
            solution = linesearch.goldensection(x=x0, dx=-dx)

            eps = 1e-3  # small deviation
            # now check that we have found the minimum in this search direction
            # since function is convex we just have to check small deviations
            # from the solution
            assert LS(solution.x) < LS(solution.x + eps*dx)
            assert LS(solution.x) < LS(solution.x - eps*dx)

    def test_undefined_start(self):
        """ Check behaviour when an undefined start point is given.

        Search should not fail silently, an explicit error is expected.
        """
        linesearch = LineSearch(lambda x: 1/x)
        with raises(ZeroDivisionError):
            linesearch.goldensection(x=0, dx=1)
            # division by zero
            # dummy value, no sensible search dir. at start point

        linesearch = LineSearch(lambda x: x**2 if abs(x) < 1 else None)
        with raises((ValueError, TypeError)):
            linesearch.goldensection(x=2, dx=-4)
            # function undefined at this value
            # good search direction

    def test_bad_search_direction(self, quadratic_objective_1d):
        """ Check that search fail explicitly when a bad search direction is
        given and a minimum cannot be bracketed.
        """
        objective, minimum, start, searchdir = quadratic_objective_1d
        linesearch = LineSearch(objective)
        for x0, dx in zip(start, searchdir):
            with raises(ValueError):
                linesearch.goldensection(x=x0, dx=-dx)
                # give bad search direction; negative of the direction
                # defined in the fixture


class TestBacktracking:
    def test_correct_nd(self):
        """ Check if n-dimensional problems which are well specified are solved
        correctly.
        """
        for n in range(2, 10):
            A = np.random.uniform(
                low=-1,
                high=1,
                size=(n, n)
            )
            b = np.random.uniform(
                low=-1,
                high=1,
                size=(n, 1)
            )
            LS = least_squares(A, b)  # Least squares instance ||Ax-b||^2

            x0 = np.random.uniform(
                low=-1,
                high=1,
                size=(n, 1)
            )  # pick random starting point
            dx = 2*A.T@A@x0 - 2*A.T@b  # derivative of x^TA^TAx-2x^TA^Tb-b^Tb

            linesearch = LineSearch(ObjectiveFunctionWrapper(
                func=LS, jac=lambda x: 2*A.T@A@x - 2*A.T@b
            ))
            solution = linesearch.backtracking(x=x0, dx=-dx)

            xmin = LS.solve_minimum()['x*']
            assert np.linalg.norm(x0-xmin) > np.linalg.norm(solution.x-xmin)
            # backtracking does not find the min,
            # check that we are closer

    def test_undefined_start(self):
        """ Check behaviour when an undefined start point is given.

        Search should not fail silently, an explicit error is expected.
        """
        linesearch = LineSearch(ObjectiveFunctionWrapper(
            func=lambda x: 1/x, jac=lambda x: -1/(x**2)
        ))
        with raises(ZeroDivisionError):
            linesearch.backtracking(x=0, dx=np.array(1, ndmin=2))
            # x: division by zero
            # dx: dummy value, no sensible search dir. at start point

        linesearch = LineSearch(ObjectiveFunctionWrapper(
            func=lambda x: x**2 if abs(x) < 1 else None,
            jac=lambda x: 2*x if abs(x) < 1 else None
        ))
        with raises((ValueError, TypeError, AttributeError)):
            linesearch.backtracking(x=2, dx=np.array(-4, ndmin=2))
            # x: function undefined at this value
            # dx: reasonable  search direction

    def test_bad_search_direction(self, quadratic_objective_1d):
        """ Check that search fail explicitly when a bad search direction is
        given and a minimum cannot be bracketed.
        """
        objective, minimum, start, searchdir = quadratic_objective_1d
        linesearch = LineSearch(objective)
        for x0, dx in zip(start, searchdir):
            with raises(ValueError):
                linesearch.goldensection(x=x0, dx=-dx)
                # give bad search direction; negative of the direction
                # defined in the fixture
