from collections import Counter

from dsai.bpe import encode_by_merges, merges_to_vocab, train_bpe
from dsai.constrained import allowed_by_scan, allowed_by_trie, number_dfa
from dsai.radix import RadixCache
from dsai.trie import Trie, byte_vocab, tokenize_by_dict


def small_vocab():
    words = Counter({b" low": 5, b" lower": 2, b" newest": 6, b" widest": 3})
    merges = train_bpe(words, 10)
    return merges, merges_to_vocab(merges)


def test_bpe_merges_frequent_pairs_first():
    merges, vocab = small_vocab()
    assert len(merges) == 10
    assert vocab[256] in (b"es", b"st", b" l", b"lo")   # 첫 병합은 빈도 최대 쌍
    rank = {m: i for i, m in enumerate(merges)}
    ids = encode_by_merges(b" newest", rank)
    assert b"".join(vocab[i] for i in ids) == b" newest"


def test_trie_longest_match_equals_dict():
    _, vocab = small_vocab()
    by_bytes = {w: t for t, w in vocab.items()}
    trie = Trie()
    for t, w in vocab.items():
        trie.insert(w, t)
    text = b" lowest newest widest low"
    a = trie.tokenize(text)
    b = tokenize_by_dict(by_bytes, text, max(len(w) for w in vocab.values()))
    assert a == b
    assert b"".join(vocab[i] for i in a) == text
    assert len(byte_vocab()) == 256


def test_radix_shares_prefix_and_evicts_leaves():
    c = RadixCache(capacity_tokens=30)
    assert c.insert([1, 2, 3, 4]) == 4
    assert c.insert([1, 2, 3, 4, 5, 6]) == 2        # 접두사 넷은 다시 안 계산
    assert c.insert([1, 2, 9]) == 1                  # 간선 중간에서 갈라진다
    matched, _ = c.match_prefix([1, 2, 3, 7])
    assert matched == 3
    for k in range(10):
        c.insert([50 + k] * 5)
    assert c.size <= 30
    matched, _ = c.match_prefix([59] * 5)
    assert matched == 5                              # 가장 최근 것은 남는다
    assert c.match_prefix([50] * 5)[0] == 0          # 가장 오래된 잎부터 나간다


def test_constrained_trie_equals_scan():
    _, vocab = small_vocab()
    vocab[300] = b"12"
    vocab[301] = b"3.5"
    trie = Trie()
    for t, w in vocab.items():
        trie.insert(w, t)
    dfa = number_dfa()
    for s in range(4):
        assert allowed_by_scan(dfa, vocab, s) == allowed_by_trie(dfa, trie, s)
    assert 301 in allowed_by_trie(dfa, trie, 0)
    assert 46 not in allowed_by_trie(dfa, trie, 0)   # 점 (바이트 46) 은 정수부 뒤에만
    assert 46 in allowed_by_trie(dfa, trie, 1)
