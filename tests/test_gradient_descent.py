import numpy as np
from pytest import approx

import context  # noqa
from src import gradient_descent, least_squares


class TestCreateJacobian:
    """ This is a class containing test cases for the _create_jac function
    which creates a numerical jacobian evaluation function.

    [ ] - Functions with reasonable functions f: R -> R
    [ ] - Functions with reasonable functions f: R^n -> R
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

    def test_create_jac_1d(self):
        """ Check if the jacobian is correctly calculated for 1-D polynomials.
        """
        for _ in range(100):  # try 100 random polynomials
            poly, polyderiv = self._random_polynomial_and_derivative()

            numericaljac = gradient_descent._create_jac(poly)
            for _ in range(10):  # 10 random points on function
                x = np.random.uniform(low=-10, high=10, size=(1, 1))
                assert numericaljac(x).item() == approx(
                    polyderiv(x),
                    abs=1E-6 if polyderiv(x) == 0 else None,
                    # comparision to 0 unreasonably stringent
                )

    def test_create_jac_1d_integers(self):
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

            numericaljac = gradient_descent._create_jac(poly)
            for _ in range(10):  # 10 random integer points on function
                x = np.random.randint(low=-10, high=10, size=(1, 1))
                assert numericaljac(x).item() == approx(
                    polyderiv(x),
                    abs=1E-6 if polyderiv(x) == 0 else None,
                    # comparision to 0 unreasonably stringent
                )

    def test_create_jac_nd(self):
        """ Check if the jacobian is correctly calculated for n-D least
        squares problems.
        """
        for _ in range(10):  # try 10 random least squares problems
            size = np.random.randint(2, 10)
            LS = least_squares.least_squares(
                A=np.random.uniform(low=-1, high=1, size=(size, size)),
                b=np.random.uniform(low=-1, high=1, size=size)
            )
            numericaljac = gradient_descent._create_jac(LS)

            for _ in range(10):  # 10 random points on function
                x = np.random.uniform(low=-10, high=10, size=(size, 1))
                assert numericaljac(x) == approx(
                        2*LS.A.T @ LS.A @ x - 2*LS.A.T @ LS.b
                    )
                # Compare with analytical solution
