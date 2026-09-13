"""실험 16.4. 순열 네 가지. 만드는 시간, 저장 바이트, 그 순서로 64 MB 파일을 읽는 시간.

레코드 100만 개, 64 바이트씩을 메모리 맵으로 두고 순열 순서로 읽어 더한다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.loader import (block_shuffle, feistel_permutation, fisher_yates,  # noqa: E402
                         sequential_reads)

N, REC, BLOCK = 1_000_000, 64, 4096


def main() -> None:
    rng = np.random.default_rng(SEED)
    keys = rng.integers(0, 2**31, 4)
    makers = {
        "순차 (섞지 않음)": lambda: np.arange(N),
        "Fisher-Yates (파이썬)": lambda: fisher_yates(N, np.random.default_rng(SEED)),
        "전체 순열 (NumPy)": lambda: np.random.default_rng(SEED).permutation(N),
        f"블록 섞기 ({BLOCK})":
            lambda: block_shuffle(N, BLOCK, np.random.default_rng(SEED)),
        "Feistel": lambda: feistel_permutation(np.arange(N), N, keys),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.bin"
        data = rng.integers(0, 256, (N, REC), dtype=np.uint8)
        data.tofile(path)
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(N, REC))
        rows = []
        for name, make in makers.items():
            order = make()
            assert np.array_equal(np.sort(order), np.arange(N))
            ms_make = median_ms(make, 1 if "파이썬" in name else 3)
            stored = 0 if "Feistel" in name or "순차" in name else order.nbytes
            frac = sequential_reads(order, BLOCK) / (N - 1) * 100
            def read():
                return int(mm[order, 0].astype(np.int64).sum())
            ms_read = median_ms(read, 3)
            rows.append([name, ms_make, stored / 1e6, frac, ms_read])
    cols = ["순열", "만들기 ms", "저장 MB", "블록 연속 퍼센트", "읽기 ms"]
    cond = (f"레코드 {N:,} 개, {REC} 바이트. 블록은 레코드 {BLOCK:,} 개 (256 KB). "
            "메모리 맵에서 레코드 첫 바이트를 순열 순서로 읽어 더함. "
            "파일은 페이지 캐시에 있음. Fisher-Yates 는 한 번, 나머지는 세 번 중앙값")
    write_tsv("c16-shuffle", cols, rows, "bench/c16_shuffle.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
