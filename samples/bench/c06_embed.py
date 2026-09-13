"""실험 6.3. 해싱 트릭의 자리 수 m 에 따른 충돌과 간섭.

어휘 백만 개를 자리 m 개로 접는다. 다른 토큰과 자리를 나누는 비율과,
그 때문에 생기는 벡터 합의 오차를 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.hashing import mix64  # noqa: E402
from dsai.vocab import EmbeddingTable, collision_rate  # noqa: E402

V = 1_000_000
D = 64
MS = [1 << 24, 1 << 20, 1 << 18, 1 << 16]
DOCS = 2000
DOC_LEN = 32


def main() -> None:
    rng = np.random.default_rng(SEED)
    full = EmbeddingTable(V, D, seed=1)
    docs = rng.integers(0, V, (DOCS, DOC_LEN))
    truth = full.lookup(docs).sum(axis=1)             # 문서 벡터. 토큰 벡터의 합
    rows = [["어휘 그대로", V, full.nbytes() / 1e6, 0.0, 0.0, 0.0]]
    for m in MS:
        table = EmbeddingTable(m, D, seed=2)
        # 접은 표의 벡터를 원래 표에서 자리마다 합쳐 만든다.
        # 같은 자리의 토큰은 같은 벡터를 나눈다
        slots = (mix64(np.arange(V, dtype=np.int64)) % np.uint64(m)).astype(np.int64)
        measured = 1.0 - np.count_nonzero(np.bincount(slots, minlength=m) == 1) / V
        folded = np.zeros((m, D), dtype=np.float32)
        np.add.at(folded, slots, full.table)
        table.table = folded
        got = table.lookup(slots[docs]).sum(axis=1)
        err = float(((got - truth) ** 2).sum() / (truth ** 2).sum())
        rows.append([f"자리 {m:,}", m, table.nbytes() / 1e6, collision_rate(V, m) * 100,
                     measured * 100, err * 100])
    cols = ["표", "행 수", "MB", "충돌 예측 퍼센트", "충돌 실측 퍼센트", "오차 퍼센트"]
    cond = (f"어휘 {V:,}개, 차원 {D}, 문서 {DOCS}개 (토큰 {DOC_LEN}개). "
            "오차는 문서 벡터 합의 상대 제곱 오차")
    write_tsv("c06-embed", cols, rows, "bench/c06_embed.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
