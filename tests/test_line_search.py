import numpy as np
from pytest import approx, raises

import context  # noqa
from src import line_search


class TestGoldenSection:
    def test_correct_1d(self):
        """ Check if one dimensional problems which are well specified are
        solved correctly.
        """
        solution = line_search.goldensection(
            func=lambda x: (x - 4)**2,  # minimum at x = 4
            x=3,  # start
            dx=2,  # search direction: -f'(3)
        )
        assert solution['x'] == approx(4)

        solution = line_search.goldensection(
            func=lambda x: (x - 4)**2,  # minimum at x = 4
            x=5,  # start
            dx=-2,  # search direction: -f'(3)
        )
        assert solution['x'] == approx(4)
