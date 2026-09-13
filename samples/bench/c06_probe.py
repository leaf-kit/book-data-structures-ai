"""실험 6.1. 적재율에 따른 탐사 수. 체이닝, 선형 탐사, 쿠쿠.

자리 m 개의 표에 키를 채우며 성공 탐색의 평균 탐사 수를 잰다.
정리 6.1 의 식과 나란히 놓는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.hashing import (ChainingTable, CuckooTable, LinearProbingTable,  # noqa: E402
                          expected_probes_chaining, expected_probes_linear)

M = 1 << 16
LOADS = [0.25, 0.5, 0.75, 0.9, 0.95]
SAMPLE = 5000


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = rng.choice(1 << 32, size=int(M * 0.95), replace=False).astype(np.int64)
    rows = []
    for alpha in LOADS:
        n = int(M * alpha)
        ch, lp, ck = ChainingTable(M), LinearProbingTable(M), CuckooTable(M)
        for k in keys[:n].tolist():
            ch.insert(k); lp.insert(k)
        cuckoo_ok = True
        try:
            for k in keys[:n].tolist():
                ck.insert(k)
        except RuntimeError:
            cuckoo_ok = False
        probe = rng.choice(keys[:n], SAMPLE).tolist()
        p_ch = np.mean([ch.probes(k) for k in probe])
        p_lp = np.mean([lp.probes(k) for k in probe])
        p_ck = np.mean([ck.probes(k) for k in probe]) if cuckoo_ok else float("nan")
        ck_txt = f"{p_ck:.2f}" if cuckoo_ok else "실패"
        rows.append([alpha, p_ch, expected_probes_chaining(alpha), p_lp,
                     expected_probes_linear(alpha), ck_txt])
    cols = ["적재율", "체이닝", "체이닝 예측", "선형 탐사", "선형 예측", "쿠쿠"]
    cond = (f"자리 {M:,}개, 무작위 정수 키, 성공 탐색 {SAMPLE}개의 평균 탐사 수. "
            f"쿠쿠는 표 둘 합쳐 자리 {2*M:,}개")
    write_tsv("c06-probe", cols, rows, "bench/c06_probe.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
