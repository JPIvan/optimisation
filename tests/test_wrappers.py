import numpy as np
from pytest import approx, fixture

import context  # noqa
from src import wrappers, least_squares


@fixture()
def rng():
    rng = np.random.default_rng()
    return rng


class TestCreateJacobian:
    """ This is a class containing test cases for the _create_jac function
    which creates a numerical jacobian evaluation function.

    [x] - Functions with reasonable functions f: R -> R
    [x] - Functions with reasonable functions f: R^n -> R
    [ ] - Reasonable behaviour with functions f: R^n -> R^m
    [ ] - Reasonable behaviour for unfriendly functions f: R -> R
    [ ] - Reasonable behaviour for unfriendly  functions f: R^n -> R

    "unfriendly" is inteanded to mean things like functions which induce
    precision loss or numerical instability
    """
    def _random_polynomial_and_derivative(self, integer_coef=False):
        polydegree = np.random.randint(2, 5)
        # generate some random coefficients for the polynomial
        if integer_coef:
            k = np.random.randint(low=-5, high=5, size=polydegree+1)
        else:
            k = np.random.uniform(low=-1, high=1, size=polydegree+1)

        def _poly(x):
            _x = np.array([x**n for n in range(polydegree+1)]).flatten()
            # flatten in case we recieved e.g. x = array([[1]])
            # then _x would be _x = arry [ [[1]], [[2]], ... ]
            return np.sum(k*_x)

        def _polyderiv(x):
            _k = k[1:]  # differentiating drops constant term
            _x = np.array([(n+1)*x**n for n in range(polydegree)])
            # for example d/dx (x^3) = 3x^2
            _x = _x.flatten()
            return np.sum(_k*_x)

        return _poly, _polyderiv

    def _compare_evaluation_at_points(
                self,
                rng_function,
                func1,
                func2,
                n_points=10,
                shape=(1, 1),
                bounds=(-10, 10),
            ):
        for _ in range(n_points):
            x = rng_function(low=bounds[0], high=bounds[1], size=shape)
            assert func1(x) == approx(
                    func2(x),
                    abs=1E-6 if shape == (1, 1) and func2(x) == 0 else None,
                    # comparision to 0 unreasonably stringent
                )

    def test_create_jac_1d(self, rng):
        """ Check if the jacobian is correctly calculated for 1-D polynomials.
        """
        for _ in range(100):  # try 100 random polynomials
            poly, polyderiv = self._random_polynomial_and_derivative()
            objectivewrapper = wrappers.ObjectiveFunctionWrapper(poly)
            # if jac isn't specified one is created automatically

            self._compare_evaluation_at_points(
                rng.uniform,
                objectivewrapper.jac,
                polyderiv,
                n_points=10
            )  # compare result for 10 random points

    def test_create_jac_1d_integers(self, rng):
        """ Numpy arrrays do not upcast when individual elements are
        modified. Verify that the jacobian creator does not fail when
        passed a function defined with integers only.

        example:
            A = np.array([1, 2])  # created with dtype=int32 by default
            A[0] = 1.1  # will not upcast result will be array([1, 2])
        """
        for _ in range(10):  # try 10 random polynomials
            poly, polyderiv = self._random_polynomial_and_derivative(
                integer_coef=True
            )
            objectivewrapper = wrappers.ObjectiveFunctionWrapper(poly)
            # if jac isn't specified one is created automatically

            self._compare_evaluation_at_points(
                rng.integers,
                objectivewrapper.jac,
                polyderiv,
                n_points=10
            )  # compare result for 10 random points

    def test_create_jac_nd(self, rng):
        """ Check if the jacobian is correctly calculated for n-D least
        squares problems.
        """
        for _ in range(10):  # try 10 random least squares problems
            size = np.random.randint(2, 10)
            LS = least_squares.least_squares(
                A=np.random.uniform(low=-1, high=1, size=(size, size)),
                b=np.random.uniform(low=-1, high=1, size=size)
            )
            objectivewrapper = wrappers.ObjectiveFunctionWrapper(LS)
            # if jac isn't specified one is created automatically

            self._compare_evaluation_at_points(
                rng.uniform,
                objectivewrapper.jac,
                lambda x: 2*LS.A.T @ LS.A @ x - 2*LS.A.T @ LS.b,
                n_points=10,
                shape=(size, 1)
            )  # compare result for 10 random points with analytical solution

    def test_create_jac_nd_integers(self, rng):
        """ Check if the jacobian is correctly calculated for n-D least
        squares problems with integer coefficients.
        """
        for _ in range(10):  # try 10 random least squares problems
            size = np.random.randint(2, 10)
            LS = least_squares.least_squares(
                A=np.random.randint(low=-10, high=10, size=(size, size)),
                b=np.random.randint(low=-10, high=10, size=size)
            )
            objectivewrapper = wrappers.ObjectiveFunctionWrapper(LS)
            # if jac isn't specified one is created automatically

            self._compare_evaluation_at_points(
                rng.integers,
                objectivewrapper.jac,
                lambda x: 2*LS.A.T @ LS.A @ x - 2*LS.A.T @ LS.b,
                n_points=10,
                shape=(size, 1),
                bounds=(-100, 100)
            )  # compare result for 10 random points with analytical solution
