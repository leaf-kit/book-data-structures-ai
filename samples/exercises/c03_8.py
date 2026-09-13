"""문제 3.8. 용량 인자.
전문가마다 받을 수 있는 토큰 수에 상한을 두면 얼마나 버려지는가.
"""
import numpy as np

from dsai.moe import route_top1


def dropped_fraction(assign: np.ndarray, num_experts: int, capacity: int) -> float:
    counts = np.bincount(assign, minlength=num_experts)
    dropped = np.clip(counts - capacity, 0, None).sum()
    return dropped / len(assign)


def test_capacity_drops_some_at_skewed_routing():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((256, 8))
    logits[:, 0] += 1.0                       # 전문가 0 으로 쏠리게
    assign = route_top1(logits)
    cap = int(1.25 * 256 / 8)                 # 용량 인자 1.25
    frac = dropped_fraction(assign, 8, cap)
    assert 0.0 < frac < 0.5
    assert dropped_fraction(assign, 8, 256) == 0.0
