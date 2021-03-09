![Build Status](https://github.com/JPIvan/optimisation/workflows/python-testing/badge.svg)

# Optimisation

This repo is intended to be an instructive collection of optimisation algorithms, following the book "Convex Optimisation" by Stephan Boyd.

- `/examples` contains code for reproducing examples from the textbook.
- `/notebooks` contains pedagogical notebooks, developing these is the primary goal of the repo.
- `/plotting` contains functions for generation explanatory plots for the notebooks.
- `/src` contains the majority of the implementation of algorithms from the book.
- `/tests` contains... tests.

At the moment the focus will be on implementing algorithms in Python, as such, no additions/corrections to the explanatory notebooks are expected in the short-term.

## Installation

The best way to use this repo is through the notebooks which demonstrate and discuss usage.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JPIvan/optimisation/HEAD?filepath=https%3A%2F%2Fgithub.com%2FJPIvan%2Foptimisation%2Fblob%2Fmain%2Fnotebooks%2FExamples_9_3_2.ipynb) currently only the `/notebooks/examples_9_3_2.ipynb` notebook is complete. You can launch the notebook in binder by clicking the badge.

Currently steepest descent with exact line search is the only optimisation procedure that is complete and sufficiently tested. If you wish to use it, the suggested notebook should be enough to get started.

**Example usage**

```python
import numpy as np
from src.gradient_descent import steepest_descent
from examples import obj_quadratic_r2  # quadratic objective

initial_value = np.random.uniform(-5, 5, (2, 1))
result = steepest_descent(
    func=obj_quadratic_r2,
    x0=initial_value,
)
result.x  # minimum
```

The suggested notebook contains snippets that can be used for visualising the intermediate results of the procedure.

This repo is currently not available as a package. Packaging the repo is currently not a priority, so users who wish to experiment with the implementations outside of the suggested notebook must clone the repo. Ensure you have Python 3.8.6 or higher installed and install the requirements in `requirements.txt.`.