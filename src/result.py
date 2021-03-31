class LineSearchResult:
    """ A simple structure for storing the results of line search procedures.

    't': argmin_s f(x + s*dx)
    'x': optimal x found by golden section search
    """
    def __init__(self, success, x, t):
        """
        Args:
            success: line search terminated successfully
            x: result of line search
            t: stepsize from start point
        """
        self.success = success
        self.x = x
        self.t = t

    def __repr__(self):
        return (
            f"success: {self.success}\n"
            f"x*: \n{self.x}\n"
            f"t: {self.t}\n"
        )


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
            path: list of intermediate solutions, useful for plotting a path
        """
        self.success = success
        self.x = x
        self.jac = kwargs["jac"] if "jac" in kwargs else None
        self.niter = niter
        self.nfev = nfev
        self.njev = njev
        self.nhev = nhev
        self.info = kwargs["info"] if "info" in kwargs else None
        self.solution_path = kwargs['path'] if 'path' in kwargs else None

    def __repr__(self):
        if self.solution_path is not None:
            solution_path_status = "available"
        else:
            solution_path_status = "unavailable"
        return (
            f"success: {self.success}\n"
            f"x*: \n{self.x}\n"
            f"jac(x*): \n{self.jac}\n"
            f"niter: {self.niter}\n"
            f"nfev: {self.nfev}\n"
            f"njev: {self.njev}\n"
            f"nhev: {self.nhev}\n"
            f"info: {self.info}\n"
            f"soution path: {solution_path_status}"
        )
