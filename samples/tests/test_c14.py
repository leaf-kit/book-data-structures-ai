import numpy as np

from dsai.dupspan import covered_tokens, duplicated_positions, longest_repeat
from dsai.ngram import InfiniGram, NgramTable
from dsai.suffix import build_suffix_array, count, find_range, lcp_array


def text_of(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8).astype(np.int64)


def test_suffix_array_is_sorted_and_counts():
    t = text_of("banana")
    sa = build_suffix_array(t)
    assert sa.tolist() == [5, 3, 1, 0, 4, 2]
    assert count(t, sa, text_of("ana")) == 2
    assert count(t, sa, text_of("nan")) == 1
    assert count(t, sa, text_of("x")) == 0
    lcp = lcp_array(t, sa)
    assert lcp.tolist() == [0, 1, 3, 0, 0, 2]
    rng = np.random.default_rng(0)
    big = rng.integers(0, 4, 3000).astype(np.int64)
    sa2 = build_suffix_array(big)
    for i in range(1, 3000):
        a, b = big[sa2[i - 1]:].tolist(), big[sa2[i]:].tolist()
        assert a < b


def test_infinigram_matches_table_when_context_seen():
    rng = np.random.default_rng(1)
    toks = rng.integers(0, 20, 5000).astype(np.int64)
    sa = build_suffix_array(toks)
    ig = InfiniGram(toks, sa)
    tbl = NgramTable(toks, 3)
    ctx = toks[100:102].tolist()
    manual = sum(1 for i in range(len(toks) - 1) if toks[i:i + 2].tolist() == ctx)
    assert ig.count(ctx) == manual
    assert set(ig.next_token_counts(ctx).items()) == set(tbl.table[tuple(ctx)].items())
    tok, k = ig.predict(toks[200:210].tolist())
    assert tok is not None and 1 <= k <= 10


def test_duplicate_spans():
    base = np.random.default_rng(2).integers(0, 50, 2000).astype(np.int64)
    text = np.concatenate([base, base[500:700], base[1800:2000]])   # 200 짜리 복사본 둘
    sa = build_suffix_array(text)
    pos = duplicated_positions(text, sa, 50)
    assert covered_tokens(pos, 50, len(text)) >= 2 * 200 + 2 * 200 - 100
    ln, a, b = longest_repeat(text, sa)
    assert ln >= 200 and text[a:a + ln].tolist() == text[b:b + ln].tolist()
