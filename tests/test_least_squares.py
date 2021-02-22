import numpy as np
from pytest import approx, raises

import context  # noqa
from src.least_squares import least_squares


class TestLeastSquaresClass:
    """ This is a class containing test cases for the least_squares
    class.

    The following tests are desired:
    [ ] - Fault tolerant constructor.
    [x] - Reasonable assurance that norms are correctly calculated.
    [ ] - Reasonable assurance that minima are correctly calculated.
    """
    class TestNorm:
        """ Class for testing norm properties of least_squares norm calculations
        performed using the __call__ function.

        Many tests check norm properties. Since the actual calculation of
        the norm is done using numpy, errors are likely to be caused by
        arguments being treated incorrectly.
        """
        def random_obj(self, size=5):
            """ Create random least-squares object for testing.
            """
            return least_squares(
                A=np.random.uniform(low=-1, high=1, size=(size, size)),
                b=np.random.uniform(low=-1, high=1, size=size)
            )

        def test_iterables(self):
            """ Check if norm is calculated the same way with different
            iterables as arguments.
            """
            LS = self.random_obj()
            assert LS(range(5)) == approx(LS([0, 1, 2, 3, 4]))
            assert LS(range(5)) == approx(LS(np.array(range(5))))
            print(f"Failed on A = {LS.A}, b = {LS.b}")

        def test_input_size(self):
            """ Check failure on wrong-size input.
            """
            LS = self.random_obj()
            with raises(ValueError):
                LS(range(4))
            with raises(ValueError):
                LS(range(6))
            print(f"Failed on A = {LS.A}, b = {LS.b}")

        def test_norm_properties(self):
            """ Check that the calculated norm has expected properties.
            """
            LS = self.random_obj()
            for x in np.random.uniform(low=-1, high=1, size=(5, 5)):
                x = x.reshape((-1, 1))
                for a in np.random.uniform(low=-1, high=1, size=5):
                    assert not (LS(a*x) == approx(a**2*LS(x)))
                    # since we are calculating ||Ax - b||^2 we should not have
                    # ||aAx - b||^2 == a^2||Ax - b||^2

                    scaledLS = least_squares(LS.A, a*LS.b)
                    assert a**2*LS(x) == approx(scaledLS(a*x))
                    # for a norm a^2||Ax - b||^2 == ||aAx - ab||^2

                    assert LS(x) <= (
                        np.linalg.norm(LS.A@x) + np.linalg.norm(LS.b))**2
                    # for a norm ||Ax - b||^2 <= (||Ax|| + ||-b||)^2
                    # ||-b|| = ||b||

                    assert LS(x) >= 0
                    # norms are always larger than 0

                    print(
                        f"Failed on A = {LS.A}, b = {LS.b}, with "
                        f"x = {x} and a = {a}"
                    )

        def test_empty_norm(self):
            """ Empty norms should always be zero.
            """
            LS = least_squares(
                A=np.zeros(shape=(5, 5)),
                b=np.zeros(5)
            )
            x = np.random.uniform(low=-1, high=1, size=5)
            assert LS(x) == approx(0)  # ||0|| = 0
