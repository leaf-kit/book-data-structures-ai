"""실험 15.1. 연관 배열 둘. 해시 조회와 어텐션 조회,
그리고 상위 k 개 키만으로 하는 근사.

키 n 개, 차원 64. dict 로 정확한 키 하나를 찾는 시간과,
어텐션이 모든 키를 견주는 시간을 잰다.
그리고 점수가 큰 키 k 개만으로 가중 합을 만들면 출력이 얼마나 어긋나는지를 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.attention import attention_naive  # noqa: E402

D = 64


def topk_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, top: int) -> np.ndarray:
    """점수가 큰 키 top 개만 남긴 어텐션. 12장의 kNN 이 고르는 자리다."""
    s = (k @ q[0]) / np.sqrt(D)
    idx = np.argpartition(-s, top - 1)[:top]
    p = np.exp(s[idx] - s[idx].max())
    p /= p.sum()
    return (p @ v[idx])[None, :]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        k = rng.standard_normal((n, D)).astype(np.float32)
        v = rng.standard_normal((n, D)).astype(np.float32)
        table = {row.tobytes(): i for i, row in enumerate(k)}
        probe = k[n // 2].tobytes()
        # 질의는 어느 키의 배수에 잡음을 더한 것.
        # 그 키의 점수가 16 이라 소프트맥스가 뾰족하다
        kj = k[n // 2]
        q = kj * (16 * np.sqrt(D) / (kj @ kj)) + 0.3 * rng.standard_normal(D)
        q = q.astype(np.float32)[None, :]
        ms_hash = median_ms(lambda: table[probe], 7) * 1e3          # 마이크로초
        ms_attn = median_ms(lambda: attention_naive(q, k, v), 5) * 1e3
        ref = attention_naive(q, k, v)
        errs = []
        for top in [16, 256, 4096]:
            top = min(top, n)
            gap = np.abs(topk_attention(q, k, v, top) - ref).max()
            errs.append(float(gap / np.abs(ref).max()) * 100)
        rows.append([n, ms_hash, ms_attn] + errs)
    cols = ["키 n", "해시 us", "어텐션 us", "오차 16 퍼센트", "오차 256 퍼센트",
            "오차 4096 퍼센트"]
    cond = ("차원 64, float32. 해시 조회는 키 바이트열을 dict 에서 찾는 것. "
            "어텐션 조회는 질의 하나가 키 전부와 점수를 내고 소프트맥스 가중 합을 "
            "만드는 것. 오차는 상위 k 개만 남긴 출력과 전체 출력의 최대 차이를 "
            "출력의 최대 절댓값으로 나눈 것. 질의는 어느 키의 배수에 잡음을 더한 "
            "것이라 그 키의 점수가 16")
    write_tsv("c15-lookup", cols, rows, "bench/c15_lookup.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
