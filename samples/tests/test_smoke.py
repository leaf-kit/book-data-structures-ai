import numpy as np

import dsai


def test_package_imports():
    assert dsai.__doc__


def test_numpy_present():
    assert np.arange(4).sum() == 6
