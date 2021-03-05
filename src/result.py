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

        Kwargs:
            info: additional information about the optimisation
            jac: value of jacobian at result
        """
        self.success = success
        self.x = x
        self.niter = niter
        self.nfev = nfev
        self.njev = njev
        self.nhev = nhev
        self.info = kwargs["info"] if "info" in kwargs else None
        self.jac = kwargs["jac"] if "jac" in kwargs else None

    def __repr__(self):
        return (
            f"success: {self.success}\n"
            f"x*: {self.x}\n"
            f"niter: {self.niter}\n"
            f"nfev: {self.nfev}\n"
            f"njev: {self.njev}\n"
            f"nhev: {self.nhev}\n"
            f"info: {self.info}\n"
            f"jac(x*): {self.jac}\n"
        )
