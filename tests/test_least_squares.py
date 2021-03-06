import numpy as np
from pytest import approx, raises

import context  # noqa
from src.least_squares import least_squares


class TestLeastSquaresClass:
    """ This is a class containing test cases for the least_squares
    class.

    The following tests are desired:
    [x] - Fault tolerant constructor.
    [x] - Reasonable assurance that norms are correctly calculated.
    [x] - Reasonable assurance that minima are correctly calculated.
    """
    def random_obj(size=5):
        """ Create random least-squares object for testing.
        """
        return least_squares(
            A=np.random.uniform(low=-1, high=1, size=(size, size)),
            b=np.random.uniform(low=-1, high=1, size=size)
        )

    class TestInit:
        """ Class for checking robust initialisation.
        """
        def test_shapes(self):
            """ Only allow initialisation if dim(A) = n*m, dim(b) = n.
            """
            with raises(ValueError):
                least_squares(
                    A=[[1], [2]],
                    b=1
                )  # A too large
            with raises(ValueError):
                least_squares(
                    A=2,
                    b=[1, 2]
                )  # b too large

        def test_one_dimensional(self):
            """  Works edge case of for n=1, m=1?
            """
            LS = least_squares(
                A=1,
                b=1
            )
            assert LS(1) == approx(0)  # 1*1 - 1 = 0

    class TestNorm:
        """ Class for testing norm properties of least_squares norm
        calculations performed using the __call__ function.

        Many tests check norm properties. Since the actual calculation of
        the norm is done using numpy, errors are likely to be caused by
        arguments being treated incorrectly.
        """
        def test_iterables(self):
            """ Check if norm is calculated the same way with different
            iterables as arguments.

            Current test: lists and np.arrays
            """
            LS = TestLeastSquaresClass.random_obj()
            assert LS(range(5)) == approx(LS([0, 1, 2, 3, 4]))
            assert LS(range(5)) == approx(LS(np.array(range(5))))
            print(f"Failed on A = {LS.A}, b = {LS.b}")

        def test_input_size(self):
            """ Check failure on wrong-size input.
            """
            LS = TestLeastSquaresClass.random_obj()  # 5x5 LS instance
            with raises(ValueError):
                LS(range(4))  # 5x5 should fail on 4x1 input vector
            with raises(ValueError):
                LS(range(6))  # 5x5 should fail on 6x1 input vector
            print(f"Failed on A = {LS.A}, b = {LS.b}")

        def test_norm_properties(self):
            """ Check that the calculated norm has expected properties.
            """
            LS = TestLeastSquaresClass.random_obj()
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

    class TestMinimiser:
        """ Class for testing least_squares.solve_minimum().
        """
        def test_underdetermined(self):
            """ Solutions to underdetermined and determined systems should
            be exactly 0.
            """
            LS = least_squares(
                A=[[1, 2, 3], [4, 5, 6]],
                b=[1, 2]
            )  # underdetermined
            x_ = LS.solve_minimum()['x*']
            assert LS(x_) == approx(0)

        def test_determined(self):
            """ Solutions to underdetermined and determined systems should
            be exactly 0.
            """
            LS = least_squares(
                A=[[1, 2, 3], [4, 5, 6], [2, 5, 7]],
                b=[1, 2, 3]
            )  # determined
            x_ = LS.solve_minimum()['x*']
            assert LS(x_) == approx(0)

        def test_overdetermined(self):
            """ Solutions to overdetermined systems should
            be local (and global) minima.

            Right now test methodology is just to check that solution
            is a local minimum. More complete tests are desireable.
            """
            LS = least_squares(
                A=[[1, 2], [3, 5], [7, 11]],  # primes ensure independence
                b=[13, 17, 19]
            )  # overdetermined
            x_, residuals, rank, _ = LS.solve_minimum().values()
            assert rank < LS.b.shape[0]  # ensure system is overdetermined
            assert residuals > 0  # ensure system is overdetermined
            assert np.linalg.norm(x_) > 0  # overdetermined -> no solution at 0
            for i in range(100):
                random_perturbation = np.random.uniform(
                    low=-1,
                    high=1,
                    size=x_.shape,
                )
                random_perturbation *= 1e-3 * np.linalg.norm(x_)
                # ensure perturbation is small compared to x_
                assert LS(x_) < LS(x_ + random_perturbation)
                # check that x_ is a local minimum
