"""문제 4.6. 등급 비율 루트 2 의 크기 등급 풀. 내부 단편화가 명제 4.4 를 따르는가."""
import numpy as np

from dsai.freelist import SlabPool


def sqrt2_classes(lo: int, hi: int) -> list[int]:
    out, s = [], float(lo)
    while s <= hi * 1.0001:
        out.append(int(round(s)))
        s *= 2 ** 0.5
    return sorted(set(out))


def test_sqrt2_waste_lower_than_pow2():
    rng = np.random.default_rng(3)
    lengths = rng.integers(16, 2048, 5000)
    pow2 = SlabPool([16, 32, 64, 128, 256, 512, 1024, 2048], 1)
    s2 = SlabPool(sqrt2_classes(16, 2048), 1)
    w2 = pow2.internal_waste(lengths)
    ws = s2.internal_waste(lengths)
    assert ws < w2
    assert len(s2.sizes) > len(pow2.sizes)
