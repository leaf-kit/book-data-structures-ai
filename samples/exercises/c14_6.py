"""문제 14.6. 구간 좁히기로 백오프. 짧은 문맥에서 시작해 구간 안에서만 찾는다."""
import numpy as np

from dsai.ngram import InfiniGram
from dsai.suffix import build_suffix_array


def narrow(text, sa, lo, hi, offset, tok):
    """구간 [lo, hi) 의 접미사 중 위치 offset 의 토큰이 tok 인 것들의 부분 구간."""
    def key(i):
        p = sa[i] + offset
        return text[p] if p < len(text) else -1
    a, b = lo, hi
    while a < b:
        m = (a + b) // 2
        if key(m) < tok:
            a = m + 1
        else:
            b = m
    start = a
    a, b = start, hi
    while a < b:
        m = (a + b) // 2
        if key(m) <= tok:
            a = m + 1
        else:
            b = m
    return start, a


def predict_narrowing(ig: InfiniGram, context: list[int], min_count: int = 2):
    """뒤에서 앞으로 문맥을 늘리며 좁힌다. 구간이 작아지기 직전의 구간으로 예측한다."""
    text, sa = ig.text, ig.sa
    best = None
    lo, hi = 0, len(sa)
    # 문맥의 마지막 토큰부터 앞으로 늘리는 대신,
    # 접미사가 뒤 k 토큰으로 시작하도록 앞에서부터 좁힌다
    for k in range(1, len(context) + 1):
        ctx = context[-k:]
        lo2, hi2 = 0, len(sa)
        for off, tok in enumerate(ctx):
            lo2, hi2 = narrow(text, sa, lo2, hi2, off, tok)
            if hi2 - lo2 < min_count:
                break
        if hi2 - lo2 < min_count:
            break
        best = (lo2, hi2, k)
    if best is None:
        return None, 0
    lo, hi, k = best
    pos = sa[lo:hi] + k
    pos = pos[pos < len(text)]
    vals, cnt = np.unique(text[pos], return_counts=True)
    return int(vals[np.argmax(cnt)]), k


def test_narrowing_agrees_with_backoff():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 12, 6000).astype(np.int64)
    ig = InfiniGram(toks, build_suffix_array(toks))
    for p in range(100, 6000, 733):
        ctx = toks[p - 10:p].tolist()
        a = ig.predict(ctx)
        b = predict_narrowing(ig, ctx)
        assert a[1] == b[1]                              # 같은 길이의 문맥을 쓴다
