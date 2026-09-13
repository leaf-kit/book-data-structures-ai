"""실험 17.1. 뒷봉투 계산과 실측. 이 책의 실험 다섯 개를 루프라인의 하한과 견준다.

이동량과 연산 수는 명제의 값이고, 실측은 앞 장의 측정 출력에서 읽는다.
비가 구현과 기계 사이의 거리다.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import OUT, write_tsv  # noqa: E402
from dsai.choose import envelope_seconds, within_factor  # noqa: E402

BW, PEAK = 16.7, 192.0          # 0장에서 잰 이 기계의 대역폭 GB/s 와 최고 GFLOPS


def read_cell(name: str, key_col: str, key: str, col: str) -> float:
    with open(OUT / f"{name}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row[key_col] == key:
                return float(row[col])
    raise KeyError(f"{name}: {key}")


def main() -> None:
    n16 = 16384
    cases = [
        # 이름, 이동 바이트, 연산 수, 실측 ms
        ["0장 순차 합", 4e6, 1e6,
         read_cell("c00-seq-vs-random", "원소 수", "1000000", "순차 ms")],
        ["15.1 어텐션 조회", 512e6, 2 * 1e6 * 64 * 2,
         read_cell("c15-lookup", "키 n", "1000000", "어텐션 us") / 1e3],
        ["15.2 타일 어텐션", 16.8e6, 4.0 * n16 * n16 * 64,
         read_cell("c15-attention", "길이 n", str(n16), "타일 ms")],
        ["16.3 열 하나의 합", 16e6, 2e6,
         read_cell("c16-columnar", "열", "f3", "열 저장 ms")],
        ["16.4 순차 읽기", 64e6, 1e6,
         read_cell("c16-shuffle", "순열", "순차 (섞지 않음)", "읽기 ms")],
    ]
    rows = []
    for name, nbytes, flops, ms in cases:
        pred = envelope_seconds(nbytes, flops, BW, PEAK) * 1e3
        rows.append([name, nbytes / 1e6, flops / 1e9, pred, ms, ms / pred,
                     "예" if within_factor(pred, ms, 3.0) else "아니오"])
    cols = ["실험", "이동 MB", "연산 GFLOP", "뒷봉투 ms", "실측 ms", "비", "3배 안"]
    cond = (f"대역폭 {BW} GB/s, 최고 {PEAK} GFLOPS 는 0장의 루프라인 측정값. "
            "이동량과 연산 수는 "
            "각 절의 명제. 실측은 해당 실험의 출력 파일에서 그대로 읽음")
    write_tsv("c17-envelope", cols, rows, "bench/c17_envelope.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
