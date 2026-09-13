import numpy as np

from dsai.bloom import BloomFilter, bits_per_key_for, false_positive_rate, optimal_k
from dsai.dedup import DedupPipeline
from dsai.minhash import (LSHIndex, candidate_probability, estimate_jaccard, jaccard,
                          minhash, shingles, word_shingles)
from dsai.sketch import CountMinSketch, HyperLogLog, cms_depth_for, cms_width_for


def test_bloom_no_false_negatives():
    keys = np.arange(1000, dtype=np.int64)
    bf = BloomFilter(16_000, optimal_k(16_000, 1000))
    bf.add(keys)
    assert bf.contains(keys).all()
    others = np.arange(10_000, 20_000, dtype=np.int64)
    fpr = bf.contains(others).mean()
    assert fpr < 3 * false_positive_rate(16_000, 1000, bf.k) + 0.01


def test_bloom_formulas():
    assert optimal_k(10_000, 1000) == 7
    assert 9 < bits_per_key_for(0.01) < 10


def test_cms_never_underestimates():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 200, 20_000).astype(np.int64)
    exact = np.bincount(keys, minlength=200)
    cms = CountMinSketch(cms_width_for(0.01), cms_depth_for(0.01))
    cms.add(keys)
    est = cms.estimate(np.arange(200, dtype=np.int64))
    assert np.all(est >= exact)
    assert np.all(est - exact <= 0.01 * len(keys) * 5)


def test_hll_close():
    rng = np.random.default_rng(2)
    keys = rng.choice(1 << 40, 50_000, replace=False).astype(np.int64)
    hll = HyperLogLog(12)
    hll.add(keys)
    assert abs(hll.estimate() - 50_000) / 50_000 < 0.05


def test_minhash_estimates_jaccard():
    a = set(range(0, 1000))
    b = set(range(500, 1500))
    j = jaccard(a, b)
    est = estimate_jaccard(minhash(a, 256), minhash(b, 256))
    assert abs(est - j) < 0.1


def test_lsh_finds_similar():
    a = set(range(0, 1000))
    b = set(range(50, 1050))          # 자카드 약 0.9
    c = set(range(5000, 6000))         # 0
    idx = LSHIndex(16, 4)
    idx.insert(1, minhash(a, 64))
    assert 1 in idx.candidates(minhash(b, 64))
    assert 1 not in idx.candidates(minhash(c, 64))
    assert candidate_probability(0.9, 16, 4) > 0.99


def test_dedup_pipeline():
    p = DedupPipeline(expected=100, threshold=0.5)
    assert p.offer(0, "the quick brown fox jumps over the lazy dog")
    assert not p.offer(1, "The Quick Brown Fox jumps over the lazy dog")   # 정확 중복
    assert not p.offer(2, "the quick brown fox jumps over the lazy cat")   # 근사 중복
    assert p.offer(3, "completely different text about data structures")
    assert p.stats.exact_dups == 1 and p.stats.near_dups == 1
    assert len(shingles("abcd", 3)) == 2
    assert len(word_shingles("a b c d", 2)) == 3
