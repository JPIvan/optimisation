class OptimisationResult:
    """ A simple structure for storing the results of optimisation procedures.
    """
    def __init__(self, success, x, niter, nfev, njev=0, nhev=0, **kwargs):
        """
        Args:
            success: optimisation terminated successfully
            x: result of optimisation
            niter: number of iterations of optimisation procedure
            nfev: number of function evaluations performed
            njev: number of jacobian evaluations performed
            nhev: number of hessian evaluations performed
        """
        self.success = success
        self.x = x
        self.niter = niter
        self.nfev = nfev
        self.njev = njev
        self.nhev = nhev
