"""실험 6.2. 어휘 사전 조회. 해시 사전, 정렬된 배열, 해싱 트릭.

토큰 32,000 개의 사전에서 조회 백만 번. 메모리와 조회당 시간을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.vocab import SortedVocab, build_vocab, string_hash  # noqa: E402

REPEAT = 5
V = 32_000
N = 1_000_000


def synthetic_tokens(rng: np.random.Generator) -> list[str]:
    """길이 2 에서 12 인 문자열 토큰.
    자주 쓰는 것이 더 자주 나오게 지프 분포로 뽑는다.
    """
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = ["".join(rng.choice(alphabet, rng.integers(2, 13))) for _ in range(V)]
    ranks = np.arange(1, V + 1)
    p = 1.0 / ranks
    p /= p.sum()
    idx = rng.choice(V, N, p=p)
    return [words[i] for i in idx]


def dict_size(d: dict) -> int:
    return sys.getsizeof(d) + sum(sys.getsizeof(k) + 28 for k in d)


def main() -> None:
    rng = np.random.default_rng(SEED)
    tokens = synthetic_tokens(rng)
    vocab = build_vocab(tokens)
    sv = SortedVocab(vocab)
    rows = []

    t = median_ms(lambda: [vocab[x] for x in tokens], REPEAT)
    rows.append(["해시 사전 (dict)", dict_size(vocab) / 1e6, t, t * 1e6 / N])
    t = median_ms(lambda: [sv.lookup(x) for x in tokens], REPEAT)
    size = (sys.getsizeof(sv.keys) + sum(sys.getsizeof(k) for k in sv.keys)
            + sys.getsizeof(sv.ids))
    rows.append(["정렬 배열 + 이진 탐색", size / 1e6, t, t * 1e6 / N])
    t = median_ms(lambda: [string_hash(x, V) for x in tokens], REPEAT)
    rows.append(["해싱 트릭 (사전 없음)", 0.0, t, t * 1e6 / N])
    cols = ["방법", "사전 MB", "조회 백만 회 ms", "조회당 ns"]
    cond = (f"토큰 {V:,}개 사전, 지프 분포로 뽑은 조회 {N:,}회. "
            "사전 크기는 키 문자열 포함")
    write_tsv("c06-vocab", cols, rows, "bench/c06_vocab.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
