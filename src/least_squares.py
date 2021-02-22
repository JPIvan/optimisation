import numpy as np


class least_squares:
    """ The least_squares class is a convenient representation of a
    least-squares problem instance. That is, for what x do we minimise
    ||Ax - b||^2 ?

    A least_squares object requires an A matrix and a b vector:
    ls = least_squares(A, b)

    Methods (see each method for more details):

    __call__(self, x):
        Returns the squared norm ||Ax -b||^2 for the given x.

    solve_minimum(self):
        A wrapper around the np.linalg.lstsq function which returns a
        dictionary:
        {
            "x*": minimiser
            "r": residuals, if appropriate
            "rk": rank of A
            "s": singular values of A
        }
    """
    def __init__(self, A, b):
        """ Create a least squares function using the given parameters.
        Instances of this class can be called with x as an argument to return
        || A*x - b ||_2^2.

        Args:
            A: 2-d numpy matrix
            b: 2-d numpy matrix (column vector)
        Returns:
            None
        """
        self.A = np.array(A, ndmin=2)
        self.b = np.array(b).reshape((-1, 1))

        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(
                "A and b have bad shapes: \n{A.shape}, {A}"
                "\n{b.shape}, {b}."
            )

        return

    def __call__(self, x):
        """ Given x, returns the squared norm of Ax - b.

        Args:
            x: 2-d numpy matrix (column vector)
        Returns:
            || A*x - b ||_2^2

        A = np.array([[1, 2], [3, 4]])
        b = np.array([6, 7])
        ls = least_squares(A, b)
        ls([1, 1])
        >> 9.0
        """
        _x = x
        if not isinstance(_x, np.ndarray):  # if not numpy array try to convert
            _x = np.array(_x).reshape(-1, 1)
        elif _x.ndim != 2 or x.shape[1] != 1:  # if not column vector fix
            _x = _x.reshape(-1, 1)

        if _x.shape[0] != self.A.shape[1]:  # column vector of wrong size
            raise ValueError(
                    f"Shape mismatch, A: {self.A.shape}, x: {_x.shape}."
                )
        return np.linalg.norm(self.A @ _x - self.b)**2

    def solve_minimum(self):
        """ Finds the x* which minimises the least-squares problem instance.
        See numpy.linalg.lstsq for more details.

        Args:
            None
        Returns:
            {
                "x*": minimiser
                "r": residuals, if appropriate
                "rk": rank of A
                "s": singular values of A
            }

        # determined system
        A = np.array([[1, 2], [3, 4]])
        b = np.array([6, 7])
        ls = least_squares(A, b)
        ls.solve_minimum()
        >> {
            'x*': array([[-5], [ 5.5]]),
            'r': array([], dtype=float64),
            'rk': 2,
            's': array([5.4649857 , 0.36596619])
            }

        # overdetermined system
        A = np.array([[1, 2], [3, 4], [3, 5]])
        b = np.array([6, 7, 8])
        ls = least_squares(A, b)
        ls.solve_minimum()
        >> {
            'x*': array([[-1.78571429], [ 2.92857143]]),
            'r': array([5.78571429]),
            'rk': 2,
            's': array([7.98626929, 0.4685113 ])
            }

        # underdetermined system, will return one possible minimiser
        A = np.array([[1, 2]])
        b = np.array([6])
        ls = least_squares(A, b)
        ls.solve_minimum()
        >> {
            'x*': array([[1.2], [2.4]]),
            'r': array([], dtype=float64),
            'rk': 1,
            's': array([2.23606798])
            }
        """
        _soln = np.linalg.lstsq(self.A, self.b, rcond=None)
        return {
            "x*": _soln[0],  # minimiser
            "r": _soln[1],  # residuals, if appropriate
            "rk": _soln[2],  # rank of A
            "s": _soln[3]  # singular values of A
        }
