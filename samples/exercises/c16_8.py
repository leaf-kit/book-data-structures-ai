"""문제 16.8. 블록 크기를 바꾸며 블록 섞기의 무작위성과 순차성을 잰다."""
import numpy as np

from dsai.loader import block_shuffle, sequential_reads


def displacement(order: np.ndarray) -> float:
    """원소가 원래 자리에서 얼마나 멀리 갔는지의 평균. 완전 무작위면 n/3 안팎이다."""
    return float(np.abs(order - np.arange(len(order))).mean())


def test_larger_blocks_more_sequential_less_random():
    n = 100_000
    seq, disp = [], []
    for block in (1, 100, 10_000, 100_000):
        order = block_shuffle(n, block, np.random.default_rng(3))
        assert np.array_equal(np.sort(order), np.arange(n))
        seq.append(sequential_reads(order, max(block, 1)) / (n - 1))
        disp.append(displacement(order))
    assert seq[0] < seq[1] < seq[2] <= seq[3]
    # 블록 순서도 섞이므로 어느 크기든 멀리 간다
    assert all(d > n / 4 for d in disp)
