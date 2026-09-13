"""문제 17.6.
파이프라인의 임베딩 단계를 해싱 트릭 대신 어휘 사전으로 바꾸면 메모리가 얼마나 다른가.
"""
import numpy as np

from dsai.pipeline import embed_docs


def embed_docs_vocab(token_lists, vocab_size):
    """어휘 크기만큼의 밀집 벡터. 해싱 트릭 없이 토큰마다 한 칸."""
    x = np.zeros((len(token_lists), vocab_size), dtype=np.float32)
    for i, ids in enumerate(token_lists):
        np.add.at(x[i], np.array(ids, dtype=np.int64), 1.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def test_hashing_trick_is_smaller_and_keeps_similar_neighbors():
    rng = np.random.default_rng(0)
    tokens = [rng.integers(0, 756, 300).tolist() for _ in range(200)]
    small = embed_docs(tokens, 64)
    big = embed_docs_vocab(tokens, 756)
    assert small.nbytes * 10 < big.nbytes
    q = 0
    near_small = set(np.argsort(-(small @ small[q]))[:10].tolist())
    near_big = set(np.argsort(-(big @ big[q]))[:10].tolist())
    assert q in near_small and q in near_big
