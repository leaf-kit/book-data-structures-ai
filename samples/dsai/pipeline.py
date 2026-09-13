"""17장. 사례 연구. dsai 를 처음부터 끝까지.

말뭉치 하나가 중복 제거, 토큰화, 세기, 임베딩, 인덱스, 서빙 캐시를 차례로 지난다.
단계마다 앞 장의 구조 하나를 그대로 쓴다. 새 코드는 잇는 것뿐이다.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from dsai.bpe import encode_by_merges, train_bpe
from dsai.dedup import DedupPipeline
from dsai.ivf import IVF, kmeans
from dsai.radix import RadixCache
from dsai.suffix import build_suffix_array, count


def split_docs(text: bytes, size: int, dup_frac: float,
               rng: np.random.Generator) -> list[str]:
    """말뭉치를 문서로 자르고 일부를 복제해 섞는다. 7장의 입력이다."""
    docs = [text[i:i + size].decode("utf-8", "ignore")
            for i in range(0, len(text), size)]
    extra = rng.choice(len(docs), int(len(docs) * dup_frac), replace=True)
    docs += [docs[i] for i in extra]
    rng.shuffle(docs)
    return docs


def dedup_docs(docs: list[str]) -> list[str]:
    """7장. 블룸 필터로 정확 중복, MinHash 와 LSH 로 근사 중복을 거른다."""
    p = DedupPipeline(expected=len(docs), k_sig=64, bands=16, threshold=0.7)
    return [d for i, d in enumerate(docs) if p.offer(i, d)]


def tokenize_docs(docs: list[str], num_merges: int) -> tuple[list[list[int]], int]:
    """9장. 단어 빈도로 BPE 를 학습하고 문서를 토큰 열로 바꾼다. (토큰 열들,
    어휘 크기).
    """
    words: Counter = Counter()
    for d in docs:
        for w in d.encode().split(b" "):
            if w:
                words[b" " + w] += 1
    merges = train_bpe(words, num_merges)
    rank = {pair: i for i, pair in enumerate(merges)}
    out = []
    for d in docs:
        ids: list[int] = []
        for w in d.encode().split(b" "):
            if w:
                ids.extend(encode_by_merges(b" " + w, rank))
        out.append(ids)
    return out, 256 + len(merges)


def count_top_bigram(token_lists: list[list[int]]) -> tuple[tuple[int, int], int]:
    """14장. 토큰 전부를 한 배열로 잇고 접미사 배열로 가장 흔한 두 토큰을 센다."""
    flat = np.array([t for ids in token_lists for t in ids], dtype=np.int64)
    pairs = Counter(zip(flat[:-1].tolist(), flat[1:].tolist()))
    top = pairs.most_common(1)[0][0]
    sa = build_suffix_array(flat)
    return top, count(flat, sa, np.array(top, dtype=np.int64))


def embed_docs(token_lists: list[list[int]], d: int) -> np.ndarray:
    """6장의 해싱 트릭. 토큰 번호를 d 로 접어 센 벡터. (문서 수, d) float32, 정규화."""
    x = np.zeros((len(token_lists), d), dtype=np.float32)
    for i, ids in enumerate(token_lists):
        np.add.at(x[i], np.array(ids, dtype=np.int64) % d, 1.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def index_recall(x: np.ndarray, k: int, nlist: int, nprobe: int,
                 rng: np.random.Generator) -> tuple[float, int]:
    """13장. 역파일을 만들고 문서 전부를 질의로 재현율@k 를 잰다. (재현율,
    본 벡터 평균).
    """
    ivf = IVF(x, kmeans(x, nlist, rng, iters=10))
    hits, seen = 0, 0
    for q in x:
        truth = set(np.argsort(((x - q) ** 2).sum(axis=1))[:k].tolist())
        found, n = ivf.search(q, nprobe, k)
        hits += len(set(found.tolist()) & truth)
        seen += n
    return hits / (k * len(x)), seen // len(x)


def radix_hit_rate(token_lists: list[list[int]], prefix: int, capacity: int) -> float:
    """9장. 문서 앞 토큰들을 기수 캐시에 넣고,
    같은 앞머리를 다시 물었을 때 맞는 비율.
    """
    cache = RadixCache(capacity)
    for ids in token_lists:
        cache.insert(ids[:prefix])
    matched = sum(cache.match_prefix(ids[:prefix])[0] for ids in token_lists)
    return matched / sum(min(len(ids), prefix) for ids in token_lists)
