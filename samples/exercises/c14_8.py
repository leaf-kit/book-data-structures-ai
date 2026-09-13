"""문제 14.8. 두 텍스트의 가장 긴 공통 부분 문자열. 이어 붙인 접미사 배열의 LCP."""
import numpy as np

from dsai.suffix import build_suffix_array, lcp_array


def longest_common_substring(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    sep = int(max(a.max(), b.max())) + 1
    text = np.concatenate([a, [sep], b]).astype(np.int64)
    sa = build_suffix_array(text)
    lcp = lcp_array(text, sa)
    side = sa > len(a)                                  # True 면 b 쪽
    best = (0, -1, -1)
    for i in range(1, len(sa)):
        if side[i] != side[i - 1]:                      # 서로 다른 텍스트의 접미사 쌍만
            ln = int(min(lcp[i], abs(int(sa[i]) - int(sa[i - 1])) ))
            ln = int(lcp[i])
            if ln > best[0]:
                pa, pb = sorted((int(sa[i]), int(sa[i - 1])))
                best = (ln, pa, pb - len(a) - 1)
    return best


def test_lcs_found():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 20, 800).astype(np.int64)
    b = rng.integers(0, 20, 800).astype(np.int64)
    b[300:420] = a[100:220]                             # 길이 120 의 공통 조각
    ln, pa, pb = longest_common_substring(a, b)
    assert ln >= 120 and a[pa:pa + 120].tolist() == b[pb:pb + 120].tolist()
