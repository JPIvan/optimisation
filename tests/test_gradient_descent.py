import numpy as np
from pytest import approx

import context  # noqa
from src import gradient_descent


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
    def test_create_jac_1d(self):
        """ Check if the jacobian is correctly calculated for 1-D polynomials.
        """
        for _ in range(10):  # try 10 random polynomials
            polydegree = np.random.randint(2, 5)
            polydegree = 3
            k = np.random.uniform(low=-1, high=1, size=polydegree+1)
            k = [1, 2, 1, 1]
            # some random coefficients for the polynomial

            def _poly(x):
                _x = np.array([x**n for n in range(polydegree+1)]).flatten()
                return np.sum(k*_x)

            def _polyderiv(x):
                _k = k[1:]  # differentiating drops constant term
                _x = np.array([(n+1)*x**n for n in range(polydegree)])
                _x = _x.flatten()
                # for example d/dx (x^3) = 3x^2
                # flatten in case we recieved e.g. x = array([[1]])
                # then _x would be _x = arry [ [[1]], [[2]], ... ]
                return np.sum(_k*_x)

            numericaljac = gradient_descent._create_jac(_poly)
            for _ in range(10):  # 10 random points on function
                x = np.random.uniform(low=-10, high=10, size=(1, 1))
                x = np.array(2, ndmin=2)
                print(
                    f"Polynomial: {k[0]}",
                    *(f" + {k[n]}x^{n}" for n in range(1, polydegree+1)),
                    f"\nDerivative: {k[1]}",
                    *(f" + {(n+1)*k[n+1]}x^{n}" for n in range(1, polydegree)),
                    f"\nEvaluated at: {x}"
                    f"\nNumerical: {numericaljac(x).item()}"
                    f", Analytic: {_polyderiv(x)}"
                )
                assert numericaljac(x).item() == approx(_polyderiv(x))
