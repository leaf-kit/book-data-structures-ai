import numpy as np

from dsai.choose import (Candidate, envelope_seconds, filter_by_questions, rank,
                         within_factor)
from dsai.pipeline import (count_top_bigram, dedup_docs, embed_docs, index_recall,
                           radix_hit_rate, split_docs, tokenize_docs)


def test_envelope_and_rank():
    assert envelope_seconds(1e9, 0, 10, 100) == 0.1
    assert envelope_seconds(0, 1e9, 10, 100) == 0.01
    a = Candidate("배열 전부", 1e9, 1e8)
    b = Candidate("인덱스", 1e7, 1e6, exact=False, static=True)
    assert [c.name for c in filter_by_questions([a, b], True, False)] == ["배열 전부"]
    assert [c.name for c in filter_by_questions([a, b], False, True)] == ["배열 전부"]
    assert rank([a, b], 10, 100)[0][0].name == "인덱스"
    assert within_factor(1.0, 2.5) and not within_factor(1.0, 4.0)


def test_pipeline_end_to_end_small():
    rng = np.random.default_rng(0)
    words = [f"w{i}".encode() for i in range(60)]
    text = b" ".join(words[j] for j in rng.integers(0, 60, 3000))
    docs = split_docs(text, 400, 0.5, rng)
    kept = dedup_docs(docs)
    assert len(kept) < len(docs)
    tokens, vocab = tokenize_docs(kept, 20)
    assert vocab == 276 and all(len(t) > 0 for t in tokens)
    (a, b), n = count_top_bigram(tokens)
    assert n >= 1
    x = embed_docs(tokens, 16)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    rec, seen = index_recall(x, 3, 2, 2, rng)
    assert rec == 1.0 and seen <= len(x)
    assert radix_hit_rate(tokens, 8, 10_000) == 1.0
