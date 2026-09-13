"""9장. 바이트 쌍 부호화. 자주 붙는 쌍을 하나로 합치는 것을 되풀이해 어휘를 만든다.

학습은 쌍의 빈도를 세고 최대를 합치는 일의 반복이다.
부호화는 학습한 병합 순서를 그대로 따른다.
같은 어휘라도 최장 일치와 병합 순서는 다른 토큰 열을 낸다.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def train_bpe(words: Counter, num_merges: int) -> list[tuple[int, int]]:
    """알고리즘 9.2. 단어별 빈도에서 병합 규칙 num_merges 개를 순서대로 만든다.

    단어는 바이트 열이고 토큰 번호 0..255 에서 시작한다. 새 토큰은 256 부터 순서대로.
    쌍의 빈도는 한 번 세고, 병합할 때 그 쌍이 든 단어만 다시 센다.
    """
    seqs = {w: list(w) for w in words}
    pairs: Counter = Counter()
    where: dict[tuple[int, int], set[bytes]] = {}
    for w, s in seqs.items():
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] += words[w]
            where.setdefault((a, b), set()).add(w)
    merges: list[tuple[int, int]] = []
    next_id = 256
    for _ in range(num_merges):
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        for w in list(where.get(best, ())):
            old = seqs[w]
            f = words[w]
            for a, b in zip(old, old[1:]):          # 이 단어의 옛 쌍을 뺀다
                pairs[(a, b)] -= f
                if pairs[(a, b)] <= 0:
                    del pairs[(a, b)]
                where[(a, b)].discard(w)
            new = _apply(old, best, next_id)
            seqs[w] = new
            for a, b in zip(new, new[1:]):          # 새 쌍을 더한다
                pairs[(a, b)] += f
                where.setdefault((a, b), set()).add(w)
        next_id += 1
    return merges


def _apply(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def encode_by_merges(word: bytes, rank: dict[tuple[int, int], int]) -> list[int]:
    """정의 9.6. 병합 순서대로 부호화. 매번 순위가 가장 낮은 (먼저 배운) 쌍을 합친다."""
    seq = list(word)
    while len(seq) > 1:
        best, best_rank = None, None
        for i, (a, b) in enumerate(zip(seq, seq[1:])):
            r = rank.get((a, b))
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = i, r
        if best is None:
            break
        new_id = 256 + best_rank
        seq = seq[:best] + [new_id] + seq[best + 2:]
    return seq


def merges_to_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """토큰 번호에서 바이트 열로. 트라이에 넣기 위한 것이다."""
    vocab = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    return vocab
