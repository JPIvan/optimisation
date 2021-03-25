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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JPIvan/optimisation/HEAD) currently only the `/notebooks/examples_9_3_2.ipynb` notebook is complete. You can launch the notebook in binder by clicking the badge. The server may take some time to start depending on when it was last accessed. Navigate to `/notebooks/examples_9_3_2.ipynb` and you should be able to run the notebook in-browser.

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

## Commit Conventions

The project is being updated to use [Coventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#summary) where commits are structured as follows:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

`<type>` should be one of `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`

A breaking change must have `!` appended to its type. See the Conventional Commits specification for more details.
