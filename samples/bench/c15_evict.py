"""실험 15.2. KV 캐시 축출. LRU 와 헤비 히터가 정확한 어텐션과 얼마나 어긋나는가.

길이 2,048 의 무작위 질의와 키에서 예산을 256 으로 두고 두 정책의 출력 오차를 잰다.
키는 일부 토큰이 자주 참조되도록 (헤비 히터가 있도록) 만든다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.kvevict import decode_with_policy, full_attention_causal  # noqa: E402

N, D, BUDGET = 2048, 64, 256


def main() -> None:
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((N, D)).astype(np.float64)
    k = rng.standard_normal((N, D)).astype(np.float64)
    v = rng.standard_normal((N, D)).astype(np.float64)
    hot = rng.choice(N // 4, 32, replace=False)     # 앞쪽 토큰 32 개가 자주 참조된다
    direction = rng.standard_normal(D)
    k[hot] += 3.0 * direction
    q += 0.5 * direction
    ref = full_attention_causal(q, k, v)
    rows = []
    for name, policy, recent in [("전부 유지", "full", 0),
                                 ("LRU", "lru", 0),
                                 ("헤비 히터", "heavy", 32)]:
        budget = BUDGET if policy != "full" else N
        out, kept = decode_with_policy(q, k, v, budget, policy, recent)
        err = np.linalg.norm(out - ref, axis=1) / np.linalg.norm(ref, axis=1)
        rows.append([name, int(kept.max()), float(np.mean(err)) * 100,
                     float(np.max(err)) * 100])
    cols = ["정책", "캐시 토큰", "평균 오차 퍼센트", "최대 오차 퍼센트"]
    cond = (f"길이 {N:,}, 차원 {D}, 예산 {BUDGET}. "
            "앞쪽 토큰 32 개의 키를 질의 방향으로 밀어 헤비 히터를 만듦. "
            "오차는 정확한 인과 어텐션 출력과의 상대 차이")
    write_tsv("c15-evict", cols, rows, "bench/c15_evict.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
