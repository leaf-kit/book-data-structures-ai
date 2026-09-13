import numpy as np
import pytest

from dsai.batch import broadcast_shape, intensity_batched, scores_batch, scores_one
from dsai.tensor import (View, col_major_strides, elements_per_line, is_contiguous,
                         lines_for_traversal, numpy_view, offset, row_major_strides)
from dsai.tiling import strided_sum, transpose_blocked, transpose_copy


def test_row_major_matches_numpy():
    a = np.zeros((3, 4, 5), dtype=np.float32)
    assert row_major_strides(a.shape, 4) == a.strides
    assert is_contiguous(a.shape, a.strides, 4)


def test_col_major_is_fortran():
    a = np.asfortranarray(np.zeros((3, 4), dtype=np.float64))
    assert col_major_strides(a.shape, 8) == a.strides


def test_offset_matches_numpy_address():
    a = np.arange(24, dtype=np.int32).reshape(4, 6)
    v = numpy_view(a)
    for i in range(4):
        for j in range(6):
            assert a.flat[v.address((i, j)) // 4] == a[i, j]


def test_transpose_is_free_and_correct():
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    v = numpy_view(a).transpose()
    assert v.shape == (4, 3) and v.strides == a.T.strides
    assert a.flat[v.address((1, 2)) // 4] == a.T[1, 2]


def test_step_view():
    a = np.arange(10, dtype=np.int64)
    v = numpy_view(a).step(3)
    assert v.shape == (4,) and v.strides == (24,)
    assert a.flat[v.address((2,)) // 8] == a[::3][2]


def test_lines_for_traversal():
    assert elements_per_line(4, 64) == 16
    assert elements_per_line(128, 64) == 1
    assert lines_for_traversal(1024, 4, 64) == 64
    assert lines_for_traversal(1024, 64, 64) == 1024


def test_strided_sum_equals_slice():
    a = np.arange(100, dtype=np.float32)
    assert strided_sum(a, 7) == a[::7].sum()


def test_blocked_transpose_equals_copy():
    a = np.random.default_rng(1).standard_normal((96, 64), dtype=np.float32)
    assert np.array_equal(transpose_blocked(a, 16), transpose_copy(a))
    assert transpose_blocked(a, 16).flags["C_CONTIGUOUS"]


def test_broadcast_rules():
    assert broadcast_shape((4, 1), (3,)) == (4, 3)
    assert broadcast_shape((5, 1, 3), (1, 4, 1)) == (5, 4, 3)
    with pytest.raises(ValueError):
        broadcast_shape((4,), (3,))


def test_batch_matches_one():
    xs = np.random.default_rng(2).standard_normal((100, 8), dtype=np.float32)
    qs = np.random.default_rng(3).standard_normal((5, 8), dtype=np.float32)
    got = scores_batch(xs, qs)
    for j in range(5):
        assert np.allclose(got[:, j], scores_one(xs, qs[j]))


def test_intensity_grows_with_batch():
    i1 = intensity_batched(10**6, 128, 1)
    i64 = intensity_batched(10**6, 128, 64)
    assert abs(i1 - 0.5) < 0.01
    assert 20 < i64 < 32
