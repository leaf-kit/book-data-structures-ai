"""문제 6.7. 부호 해시. 겹친 벡터를 더할 때 부호를 곱하면 기대값에서 간섭이 사라진다."""
import numpy as np

from dsai.hashing import mix64


def signed_fold(full: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = len(full)
    ids = np.arange(v, dtype=np.int64)
    slots = (mix64(ids) % np.uint64(m)).astype(np.int64)
    top_bit = (mix64(ids + 7) >> np.uint64(63)) == 1
    signs = np.where(top_bit, 1.0, -1.0).astype(np.float32)
    folded = np.zeros((m, full.shape[1]), dtype=np.float32)
    np.add.at(folded, slots, full * signs[:, None])
    return folded, slots, signs


def test_signed_dot_is_unbiased():
    rng = np.random.default_rng(1)
    v, d, m = 20000, 32, 4096
    full = rng.standard_normal((v, d), dtype=np.float32)
    folded, slots, signs = signed_fold(full, m)
    a, b = rng.integers(0, v, 3000), rng.integers(0, v, 3000)
    true = np.einsum("ij,ij->i", full[a], full[b])
    est = np.einsum("ij,ij->i", folded[slots[a]] * signs[a][:, None],
                    folded[slots[b]] * signs[b][:, None])
    # 서로 다른 토큰의 내적은 0 근처. 부호 덕에 간섭의 평균이 0 이다
    assert abs((est - true).mean()) < 0.5
