import numpy as np

class least_squares:
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
        self.A = A
        self.b = b
    
        return
    
    def __call__(self, x):
        """ Given x, returns the squared norm of Ax - b.

        Args:
            x: 2-d numpy matrix (column vector)
        Returns:
            || A*x - b ||_2^2
        """
        _x = x
        if not isinstance(_x, np.ndarray): # if not numpy array try to make one
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
        This uses np.linalg.lstsq.

        Args:
            None
        Returns:
            {
                "x*": minimiser
                "r": residuals, if appropriate
                "rk": rank of A
                "s": singular values of A
            }
        """
        _soln = np.linalg.lstsq(self.A, self.b, rcond=None)
        return {
            "x*": _soln[0],  # minimiser
            "r": _soln[1],  # residuals, if appropriate
            "rk": _soln[2],  # rank of A
            "s": _soln[3]  # singular values of A
        }