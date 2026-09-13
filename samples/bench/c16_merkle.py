"""실험 16.2. 머클 트리. 블록 크기에 따른 다시 저장하는 양과 다른 블록을 찾는 시간.

16 MB 체크포인트에서 200 곳을 바꾼 새 체크포인트를 만든다.
블록 크기를 바꾸며 내용 주소 저장소에 새로 드는 바이트, 다른 블록 수, 찾는 시간을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.merkle import (ContentStore, block_hashes, changed_blocks,  # noqa: E402
                         merkle_root)

SIZE = 16 * 2**20
CHANGES = 200


def main() -> None:
    rng = np.random.default_rng(SEED)
    old = rng.integers(0, 256, SIZE, dtype=np.uint8)
    new = old.copy()
    pos = rng.choice(SIZE, CHANGES, replace=False)
    new[pos] ^= 1
    old_b, new_b = old.tobytes(), new.tobytes()
    rows = []
    for kb in [4, 64, 1024]:
        block = kb * 1024
        store = ContentStore()
        store.put(old_b, block)
        before = store.stored
        store.put(new_b, block)
        ho, hn = block_hashes(old_b, block), block_hashes(new_b, block)
        _, lo = merkle_root(ho)
        _, ln = merkle_root(hn)
        diff = changed_blocks(lo, ln)
        ms_hash = median_ms(lambda: block_hashes(new_b, block), 3)
        ms_find = median_ms(lambda: changed_blocks(lo, ln), 3)
        ms_full = median_ms(lambda: [a == b for a, b in zip(ho, hn)], 3)
        rows.append([kb, len(hn), (store.stored - before) / 2**20, len(diff),
                     ms_hash, ms_full, ms_find])
    cols = ["블록 KB", "블록 수", "새로 든 MB", "다른 블록", "해시 ms", "비교 ms",
            "트리 ms"]
    cond = (f"{SIZE // 2**20} MB 무작위 바이트. 새 판은 {CHANGES} 바이트를 바꿈. "
            "SHA-256. 새로 든 MB 는 내용 주소 저장소에 두 판을 넣었을 때 "
            "둘째 판이 더한 양. 세 번 중앙값")
    write_tsv("c16-merkle", cols, rows, "bench/c16_merkle.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
