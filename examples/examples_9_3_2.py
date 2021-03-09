import numpy as np


# This file contains the suggested examples from Section 9.3.2
# they all assume the input x is of the correct dimension, and a column vector

def obj_quadratic_r2(x, gamma=2):
    """ This is the objective function:
    f(x) = 1/2 * (x_1^2 + gamma*x_2^2)

    Args:
        x: input, a 2x1 numpy array
        gamma: defaults to 2, can be used to control the condition number of
            the sublevel sets of the function.
            condition number = max(gamma, 1/gamma)
    """
    _A = np.array(
        [[1, 0],
         [0, gamma]], dtype=float
    )
    return 0.5 * x.T @ _A @ x


def obj_quadratic_r2_jac(x, gamma=2):
    """ This is the objective function:
    f(x) = 1/2 * (x_1^2 + gamma*x_2^2)

    Args:
        x: input, a 2x1 numpy array
        gamma: defaults to 2, can be used to control the condition number of
            the sublevel sets of the function.
            condition number = max(gamma, 1/gamma)
    """
    _A = np.array(
        [[1, 0],
         [0, gamma]], dtype=float
    )
    return _A @ x


def obj_nonquadratic_r2(x):
    """ This is the objective function:
    f(x) = exp(x_1 + 3x_2 - 0.1) + exp(x_1 - 3x_2 - 0.1) + exp(-x_1 - 0.1)

    Args:
        x: input, a 2x1 numpy array. Recommend keeping inputs reasonably small.
    """
    x1, x2 = x.flatten()
    return np.exp(x1+3*x2-0.1) + np.exp(x1-3*x2-0.1) + np.exp(-x1-0.1)
