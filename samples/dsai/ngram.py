"""14장. n-그램 세기. 접미사 배열 위의 무한 n-그램.

n-그램 표는 n 을 정해 두고 개수를 dict 에 센다. 접미사 배열은 n 을 정하지 않는다.
어떤 문맥이든 그 문맥으로 시작하는 접미사의 구간을 찾으면 개수이고,
구간 안에서 다음 토큰의 분포가 나온다.
문맥이 말뭉치에 없으면 뒤에서부터 줄여 가장 긴 있는 문맥을 쓴다.
이것이 무한 n-그램의 백오프다.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from dsai.suffix import find_range


class NgramTable:
    """디딤돌. n 을 정한 표. (n-1 토큰 문맥) -> 다음 토큰의 개수."""

    def __init__(self, tokens: np.ndarray, n: int):
        self.n = n
        self.table: dict[tuple, Counter] = {}
        t = tokens.tolist()
        for i in range(len(t) - n + 1):
            ctx = tuple(t[i:i + n - 1])
            self.table.setdefault(ctx, Counter())[t[i + n - 1]] += 1

    def predict(self, context: list[int]) -> int | None:
        c = self.table.get(tuple(context[-(self.n - 1):]) if self.n > 1 else ())
        return c.most_common(1)[0][0] if c else None

    def nbytes(self) -> int:
        import sys
        return sum(sys.getsizeof(k) + sys.getsizeof(v) + 60 * len(v)
                   for k, v in self.table.items())


class InfiniGram:
    """정의 14.3. 접미사 배열 하나로 모든 n 의 개수를 낸다. 표가 없다."""

    def __init__(self, tokens: np.ndarray, sa: np.ndarray):
        self.text = tokens
        self.sa = sa

    def count(self, pattern: list[int]) -> int:
        pat = np.array(pattern, dtype=self.text.dtype)
        lo, hi = find_range(self.text, self.sa, pat)
        return hi - lo

    def next_token_counts(self, context: list[int]) -> Counter:
        """알고리즘 14.3. 문맥의 구간 안에서 다음 토큰을 센다.
        구간이 크면 표본만 본다.
        """
        ctx = np.array(context, dtype=self.text.dtype)
        lo, hi = find_range(self.text, self.sa, ctx)
        if hi <= lo:
            return Counter()
        pos = self.sa[lo:hi] + len(context)
        pos = pos[pos < len(self.text)]
        if len(pos) > 4096:
            pos = pos[:: len(pos) // 4096]
        vals, cnt = np.unique(self.text[pos], return_counts=True)
        return Counter(dict(zip(vals.tolist(), cnt.tolist())))

    def predict(self, context: list[int], min_count: int = 2) -> tuple[int | None, int]:
        """백오프. 가장 긴 문맥부터, 구간이 min_count 이상인 첫 문맥으로 예측. (토큰,
        쓴 문맥 길이).
        """
        for k in range(min(len(context), 64), 0, -1):
            c = self.next_token_counts(context[-k:])
            if sum(c.values()) >= min_count:
                return c.most_common(1)[0][0], k
        return None, 0
