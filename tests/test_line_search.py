import numpy as np
from pytest import approx, fixture, raises

import context  # noqa
from src import line_search
from src.least_squares import least_squares
from src.wrappers import ObjectiveFunctionWrapper


@fixture
def quadratic_objective_1d():
    return ObjectiveFunctionWrapper(
        func=lambda x: (x - 4)**2,
        jac=lambda x: np.array(2*(x - 4), ndmin=2),
    )


class TestGoldenSection:
    def test_correct_1d(self, quadratic_objective_1d):
        """ Check if one dimensional problems which are well specified are
        solved correctly.
        """
        solution = line_search.goldensection(
            func=quadratic_objective_1d.f,  # minimum at x = 4
            x=3,  # start
            dx=2,  # search direction: -f'(3)
        )
        assert solution.x == approx(4)

        solution = line_search.goldensection(
            func=quadratic_objective_1d.f,  # minimum at x = 4
            x=5,  # start
            dx=-2,  # search direction: -f'(3)
        )
        assert solution.x == approx(4)

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

            solution = line_search.goldensection(
                func=LS,
                x=x0,
                dx=-dx,
            )

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
        with raises(ZeroDivisionError):
            line_search.goldensection(
                func=lambda x: 1/x,
                x=0,  # division by zero
                dx=1,  # dummy value, no sensible search dir. at start point
            )
        with raises((ValueError, TypeError)):
            line_search.goldensection(
                func=lambda x: x**2 if abs(x) < 1 else None,
                x=2,  # function undefined at this value
                dx=-4,  # good search direction
            )

    def test_bad_search_direction(self, quadratic_objective_1d):
        """ Check that search fail explicitly when a bad search direction is
        given and a minimum cannot be bracketed.
        """
        with raises(ValueError):
            line_search.goldensection(
                func=quadratic_objective_1d.f,  # minimum at x = 4
                x=3,  # start
                dx=-1,  # bad search direction; away from minimum
            )


class TestBacktracking:
    def test_correct_1d(self, quadratic_objective_1d):
        """ Check if one dimensional problems which are well specified are
        solved correctly.
        """
        solution = line_search.backtracking(
            func=quadratic_objective_1d.f,  # minimum at x = 4
            jac=quadratic_objective_1d.jac,
            x=np.array(3, ndmin=2),  # start
            dx=np.array(2, ndmin=2),  # search direction: -f'(3)
        )
        assert np.linalg.norm(solution.x-4) < 1
        # backtracking does not find the min,
        # only ensures that we decrease the function

        solution = line_search.backtracking(
            func=quadratic_objective_1d.f,  # minimum at x = 4
            jac=quadratic_objective_1d.jac,
            x=np.array(5, ndmin=2),  # start
            dx=np.array(-2, ndmin=2),  # search direction: -f'(5)
        )
        assert np.linalg.norm(solution.x-4) < 1
        # backtracking does not find the min,
        # only ensures that we decrease the function

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

            solution = line_search.backtracking(
                func=lambda x: LS(x),
                jac=lambda x: 2*A.T@A@x - 2*A.T@b,
                x=x0,
                dx=-dx,
            )

            xmin = LS.solve_minimum()['x*']
            assert np.linalg.norm(x0-xmin) > np.linalg.norm(solution.x-xmin)
            # backtracking does not find the min,
            # check that we are closer

    def test_undefined_start(self):
        """ Check behaviour when an undefined start point is given.

        Search should not fail silently, an explicit error is expected.
        """
        with raises(ZeroDivisionError):
            line_search.backtracking(
                func=lambda x: 1/x,
                jac=lambda x: -1/(x**2),
                x=0,  # division by zero
                dx=np.array(1, ndmin=2),
                # dummy value, no sensible search dir. at start point
            )
        with raises((ValueError, TypeError, AttributeError)):
            line_search.backtracking(
                func=lambda x: x**2 if abs(x) < 1 else None,
                jac=lambda x: 2*x if abs(x) < 1 else None,
                x=2,  # function undefined at this value
                dx=np.array(-4, ndmin=2),  # good search direction
            )

    def test_bad_search_direction(self, quadratic_objective_1d):
        """ Check that search fail explicitly when a bad search direction is
        given and a minimum cannot be bracketed.
        """
        with raises(ValueError):
            line_search.backtracking(
                func=quadratic_objective_1d.f,  # minimum at x = 4
                jac=quadratic_objective_1d.jac,
                x=3,  # start
                dx=np.array(-1, ndmin=2),
                # bad search direction; away from minimum
            )
