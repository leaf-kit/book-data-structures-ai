"""문제 1.5. 3차원 텐서의 두 축을 바꾼 뷰의 스트라이드.

손으로 계산하고 NumPy 와 맞춘다.
"""
import numpy as np

from dsai.tensor import View, numpy_view, row_major_strides


def swap_axes(v: View, i: int, j: int) -> View:
    shape = list(v.shape)
    strides = list(v.strides)
    shape[i], shape[j] = shape[j], shape[i]
    strides[i], strides[j] = strides[j], strides[i]
    return View(tuple(shape), tuple(strides), v.offset)


def test_swap_matches_numpy():
    a = np.zeros((2, 3, 4), dtype=np.float32)
    assert row_major_strides(a.shape, 4) == (48, 16, 4)
    v = swap_axes(numpy_view(a), 0, 2)
    assert v.strides == np.swapaxes(a, 0, 2).strides == (4, 16, 48)
