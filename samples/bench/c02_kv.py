"""실험 2.3 과 2.4. KV 캐시 세 방식의 예약 대비 사용, 그리고 접두사 공유.

요청 길이를 미리 모르니 최대 길이만큼 잡거나, 두 배씩 늘리거나, 블록으로 나눈다.
같은 길이 분포에서 셋이 예약한 자리 중 실제로 쓴 비율을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.dynarray import total_copied  # noqa: E402
from dsai.kvcache import BlockPool, PagedKV, bytes_per_token  # noqa: E402

MAX_LEN = 4096
REQUESTS = 2000
DIM = 8            # 시뮬레이션용. 바이트 계산은 bytes_per_token 으로 따로 한다
BLOCKS = [1, 16, 64, 256]


def lengths(rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.lognormal(6.0, 0.8, REQUESTS).astype(int), 8, MAX_LEN)


def main() -> None:
    rng = np.random.default_rng(SEED)
    lens = lengths(rng)
    used = int(lens.sum())
    rows = []
    # 최대 길이 예약
    reserved = MAX_LEN * REQUESTS
    rows.append(["최대 길이 예약", 0, reserved / used, 0.0])
    # 두 배 늘리기. 예약은 마지막 용량, 옮긴 양은 정리 2.1
    cap = np.array([1 << int(np.ceil(np.log2(x))) for x in lens])
    copied = sum(total_copied(int(x), 2.0, 8) for x in lens)
    rows.append(["두 배 늘리기", 0, cap.sum() / used, copied / used])
    # 페이지
    for b in BLOCKS:
        pool = BlockPool(num_blocks=used // b + REQUESTS + 1,
                         block_size=b, dim=DIM)
        kvs = []
        for ln in lens:
            kv = PagedKV(pool)
            for _ in range(int(ln)):
                kv.append(np.zeros(DIM, dtype=np.float32))
            kvs.append(kv)
        reserved = sum(kv.reserved() for kv in kvs)
        rows.append([f"블록 {b}", b, reserved / used, 0.0])
    cond = (f"요청 {REQUESTS}개, 최대 길이 {MAX_LEN}, 길이는 로그정규 분포. "
            "비율은 사용 토큰 1 기준")
    write_tsv("c02-kv", ["방식", "블록", "예약 대비 사용", "옮긴 비율"], rows,
              "bench/c02_kv.py", cond, 1)

    # 접두사 공유. 같은 접두사 P 토큰 뒤에 서로 다른 S 토큰을 붙인 요청 K 개
    rows2 = []
    for prefix, suffix, k in [(1024, 128, 8), (1024, 128, 64), (256, 256, 64)]:
        b = 16
        pool = BlockPool(num_blocks=(prefix + suffix) * k // b + k + 8,
                         block_size=b, dim=DIM)
        root = PagedKV(pool)
        for _ in range(prefix):
            root.append(np.zeros(DIM, dtype=np.float32))
        forks = [root.fork() for _ in range(k)]
        for f in forks:
            for _ in range(suffix):
                f.append(np.ones(DIM, dtype=np.float32))
        shared = pool.used_blocks() * b
        separate = (prefix + suffix) * k
        rows2.append([prefix, suffix, k, separate, shared, separate / shared])
    cols = ["접두사", "접미사", "요청 수", "따로 저장", "공유 저장", "절약 배"]
    write_tsv("c02-prefix", cols, rows2, "bench/c02_kv.py",
              "블록 16, 단위는 토큰 자리 수", 1)

    per_tok = bytes_per_token(layers=40, heads=40, head_dim=128, elem_bytes=2)
    print(f"토큰당 바이트 (40층, 40헤드, 128차원, fp16): {per_tok:,}")
    for r in rows + rows2:
        print(r)


if __name__ == "__main__":
    main()
