"""문제 16.7. 열 저장에 압축을 얹는다.
런 길이 부호화가 정렬된 열에서 얼마나 줄이는지.
"""
import numpy as np

from dsai.columnar import bytes_touched


def rle_encode(col: np.ndarray):
    change = np.flatnonzero(np.diff(col)) + 1
    starts = np.concatenate([[0], change])
    values = col[starts]
    lengths = np.diff(np.concatenate([starts, [len(col)]]))
    return values, lengths


def rle_decode(values, lengths):
    return np.repeat(values, lengths)


def rle_sum(values, lengths):
    return int((values.astype(np.int64) * lengths).sum())


def test_rle_on_sorted_column_is_small_and_sums_equal():
    rng = np.random.default_rng(2)
    col = np.sort(rng.integers(0, 100, 1_000_000))
    values, lengths = rle_encode(col)
    assert np.array_equal(rle_decode(values, lengths), col)
    assert rle_sum(values, lengths) == int(col.sum())
    assert len(values) == 100
    assert values.nbytes + lengths.nbytes < col.nbytes / 100
    assert bytes_touched(1_000_000, 64, 8, "col") == col.nbytes
