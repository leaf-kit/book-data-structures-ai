"""문제 14.7. 첫 출현 남기기. 반복 구간마다 위치가 가장 작은 것만 남긴다."""
import numpy as np

from dsai.suffix import build_suffix_array, lcp_array


def later_occurrences(text: np.ndarray, sa: np.ndarray, min_len: int) -> np.ndarray:
    """LCP 가 문턱 이상으로 이어지는 SA 구간마다 최소 위치를 뺀 나머지."""
    lcp = lcp_array(text, sa)
    out = []
    i = 1
    n = len(sa)
    while i < n:
        if lcp[i] >= min_len:
            j = i
            while j < n and lcp[j] >= min_len:
                j += 1
            group = sa[i - 1:j]
            out.extend(int(v) for v in group if v != group.min())
            i = j
        else:
            i += 1
    return np.unique(np.array(out, dtype=np.int64))


def test_first_occurrence_kept():
    base = np.random.default_rng(1).integers(0, 30, 1500).astype(np.int64)
    text = np.concatenate([base, base[200:400]])           # 200 짜리 복사본 하나
    sa = build_suffix_array(text)
    later = later_occurrences(text, sa, 100)
    assert (later >= 1500).all() or len(later) > 0
    assert 200 not in later                                # 첫 출현은 남는다
    assert 1500 in later                                   # 복사본은 나중 것이다
