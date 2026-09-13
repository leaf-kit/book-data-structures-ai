"""실험 16.3. 행 저장과 열 저장. 열 하나를 합칠 때 닿는 바이트와 시간.

여덟 열 (int64 넷, float64 넷), 행 200만 개, 128 MB. 메모리 맵으로 열 하나를 더한다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.columnar import (bytes_touched, column_sum_column_major,  # noqa: E402
                           column_sum_row_major, write_column_major, write_row_major)

ROWS = 2_000_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    table = {f"i{k}": rng.integers(0, 1000, ROWS) for k in range(4)}
    table.update({f"f{k}": rng.standard_normal(ROWS) for k in range(4)})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows_arr = write_row_major(tmp / "rows.bin", table)
        write_column_major(tmp / "cols", table)
        dtype = rows_arr.dtype
        rows = []
        for name in ["i0", "f3"]:
            expect = float(table[name].astype(np.float64).sum())
            ctype = table[name].dtype

            def by_row():
                return column_sum_row_major(tmp / "rows.bin", dtype, name)

            def by_col():
                return column_sum_column_major(tmp / "cols", name, ctype)

            assert abs(by_row() - expect) < 1e-3 and abs(by_col() - expect) < 1e-3
            rows.append([name, median_ms(by_row, 5), median_ms(by_col, 5),
                         bytes_touched(ROWS, dtype.itemsize, 8, "row") / 2**20,
                         bytes_touched(ROWS, dtype.itemsize, 8, "col") / 2**20])
    cols = ["열", "행 저장 ms", "열 저장 ms", "행 닿는 MB", "열 닿는 MB"]
    cond = (f"행 {ROWS:,} 개, 열 8 개 (int64 4, float64 4), 레코드 64 바이트. "
            "메모리 맵 읽기. "
            "파일은 방금 써서 페이지 캐시에 있음. 다섯 번 중앙값")
    write_tsv("c16-columnar", cols, rows, "bench/c16_columnar.py", cond, 5)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
