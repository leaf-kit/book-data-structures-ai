"""문제 1.8. 배치 밀도가 능선점을 넘는 최소 배치 크기."""
from dsai.batch import intensity_batched


def min_batch_over_ridge(n: int, d: int, ridge: float) -> int:
    b = 1
    while intensity_batched(n, d, b) < ridge:
        b *= 2
        if b > n:
            raise ValueError("이 크기에서는 능선점을 넘지 못한다")
    return b


def test_min_batch():
    # 능선점 8 이면 배치 16 부터 넘는다 (명제 1.3 에서 I(b) 는 대략 b/2)
    assert min_batch_over_ridge(262_144, 128, 8.0) == 32
    assert min_batch_over_ridge(262_144, 128, 0.5) == 1
