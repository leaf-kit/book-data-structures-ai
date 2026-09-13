"""14장. 긴 중복 찾기. 말뭉치 안의 복사본.

접미사 배열의 이웃한 두 접미사가 L 이상의 공통 접두사를 가지면,
그 자리에서 길이 L 의 부분 문자열이 두 번 이상 나온다.
LCP 배열을 한 번 훑으면 모든 중복 구간이 나온다.
7장의 MinHash 가 문서 단위였다면 이것은 부분 문자열 단위다.
"""
from __future__ import annotations

import numpy as np

from dsai.suffix import lcp_array


def duplicated_positions(text: np.ndarray, sa: np.ndarray, min_len: int) -> np.ndarray:
    """알고리즘 14.4. 길이 min_len 이상의 부분 문자열이 다른 곳에도 나오는 시작 위치들.

    lcp[i] >= min_len 이면 sa[i-1] 과 sa[i] 둘 다 그런 위치다. 정렬된 위치 배열이다.
    """
    lcp = lcp_array(text, sa)
    hit = lcp >= min_len
    pos = np.concatenate([sa[hit], sa[np.flatnonzero(hit) - 1]])
    return np.unique(pos)


def covered_tokens(positions: np.ndarray, min_len: int, n: int) -> int:
    """중복 구간이 덮는 토큰 수. 위치마다 [p,
    p + min_len) 을 덮고 겹침은 한 번만 센다.
    """
    if len(positions) == 0:
        return 0
    starts = np.sort(positions)
    ends = np.minimum(starts + min_len, n)
    covered, cur_s, cur_e = 0, int(starts[0]), int(ends[0])
    for s, e in zip(starts[1:].tolist(), ends[1:].tolist()):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    return covered + cur_e - cur_s


def longest_repeat(text: np.ndarray, sa: np.ndarray) -> tuple[int, int, int]:
    """가장 긴 반복 부분 문자열. (길이, 위치 1, 위치 2)."""
    lcp = lcp_array(text, sa)
    i = int(np.argmax(lcp))
    return int(lcp[i]), int(sa[i - 1]), int(sa[i])
