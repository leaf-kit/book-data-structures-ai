"""문제 9.6. 힙으로 병합 순서 부호화. 이웃 쌍의 순위를 힙에 넣고 최소를 합친다."""
import heapq
from collections import Counter

from dsai.bpe import encode_by_merges, train_bpe


def encode_with_heap(word: bytes, rank: dict[tuple[int, int], int]) -> list[int]:
    seq = list(word)
    alive = [True] * len(seq)
    nxt = list(range(1, len(seq) + 1))
    prv = list(range(-1, len(seq) - 1))
    heap = []
    for i in range(len(seq) - 1):
        r = rank.get((seq[i], seq[i + 1]))
        if r is not None:
            heapq.heappush(heap, (r, i))
    while heap:
        r, i = heapq.heappop(heap)
        j = nxt[i]
        if not alive[i] or j >= len(seq) or not alive[j]:
            continue
        if rank.get((seq[i], seq[j])) != r:              # 낡은 항목
            continue
        seq[i] = 256 + r
        alive[j] = False
        nxt[i] = nxt[j]
        if nxt[i] < len(seq):
            prv[nxt[i]] = i
            r2 = rank.get((seq[i], seq[nxt[i]]))
            if r2 is not None:
                heapq.heappush(heap, (r2, i))
        if prv[i] >= 0:
            r3 = rank.get((seq[prv[i]], seq[i]))
            if r3 is not None:
                heapq.heappush(heap, (r3, prv[i]))
    return [s for s, a in zip(seq, alive) if a]


def test_heap_encoding_matches_scan():
    words = Counter({b" lower": 4, b" lowest": 3, b" newer": 5, b" wider": 2,
                     b" low": 7})
    merges = train_bpe(words, 12)
    rank = {m: i for i, m in enumerate(merges)}
    for w in (b" lower", b" newest", b" slow", b" lowlow"):
        assert encode_with_heap(w, rank) == encode_by_merges(w, rank)
