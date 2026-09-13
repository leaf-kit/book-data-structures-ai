import numpy as np

from dsai.hashing import (ChainingTable, CuckooTable, LinearProbingTable,
                          expected_probes_chaining, expected_probes_linear,
                          universal_hash)
from dsai.parhash import (batch_insert_conflicts, batch_insert_rounds, count_serial,
                          count_sorted)
from dsai.vocab import (EmbeddingTable, SortedVocab, build_vocab, collision_rate,
                        hashed_ids, string_hash)


def test_universal_hash_in_range():
    keys = np.arange(1000, dtype=np.int64)
    h = universal_hash(keys, 12345 * 2 + 1, 128)
    assert h.min() >= 0 and h.max() < 128


def test_tables_find_inserted_keys():
    keys = np.random.default_rng(1).choice(1 << 30, 500, replace=False).tolist()
    for T in (ChainingTable, LinearProbingTable, CuckooTable):
        t = T(2048)
        for k in keys:
            t.insert(k)
        assert all(t.probes(k) >= 1 for k in keys)
    ck = CuckooTable(2048)
    for k in keys:
        ck.insert(k)
    assert max(ck.probes(k) for k in keys) <= 2


def test_expected_probes_monotone():
    assert expected_probes_chaining(0.5) < expected_probes_chaining(0.9)
    assert expected_probes_linear(0.5) < expected_probes_linear(0.9)
    assert expected_probes_linear(0.9) > 5


def test_vocab_and_sorted_lookup():
    toks = "a b a c b a d".split()
    v = build_vocab(toks)
    assert v["a"] == 0 and v["b"] == 1
    sv = SortedVocab(v)
    assert sv.lookup("c") == v["c"] and sv.lookup("zz") == -1


def test_hashing_trick_deterministic():
    assert string_hash("hello", 100) == string_hash("hello", 100)
    ids = hashed_ids(["x", "y", "x"], 50)
    assert ids[0] == ids[2] and ids.max() < 50


def test_embedding_lookup_is_gather():
    e = EmbeddingTable(10, 4)
    out = e.lookup(np.array([3, 3, 7]))
    assert np.array_equal(out[0], out[1]) and out.shape == (3, 4)


def test_collision_rate_bounds():
    assert collision_rate(1000, 1 << 30) < 1e-5
    assert 0.6 < collision_rate(1000, 1000) < 0.7


def test_count_sorted_matches_serial():
    keys = np.random.default_rng(2).integers(0, 50, 1000)
    u, c = count_sorted(keys)
    d = count_serial(keys)
    assert dict(zip(u.tolist(), c.tolist())) == d


def test_batch_insert_rounds_terminate():
    rng = np.random.default_rng(3)
    keys = rng.choice(1 << 30, 3000, replace=False).astype(np.int64)
    conflicted, written = batch_insert_conflicts(keys, 4096, 7)
    assert conflicted + written == 3000
    assert batch_insert_rounds(keys, 4096, 7) >= 2
