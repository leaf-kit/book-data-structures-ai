"""실험 17.3. 판정을 뒤집는 조건. 같은 짝을 규모와 차원을 바꾸며 다시 잰다.

kd-트리 대 전부 비교는 차원이, 역파일 대 전부 비교는 크기가 판정을 바꾼다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.ivf import IVF, kmeans  # noqa: E402
from dsai.kdtree import KDTree, brute_nearest  # noqa: E402


def sweep_dims(n: int, dims: list[int], rng: np.random.Generator) -> list[list]:
    """축 하나. 차원을 바꾸며 kd-트리와 전부 비교를 잰다.
    둘 다 정확하므로 재현율 100.
    """
    rows = []
    for d in dims:
        pts = rng.standard_normal((n, d))
        qs = rng.standard_normal((20, d))
        tree = KDTree(pts)
        ms_old = median_ms(lambda: [brute_nearest(pts, q) for q in qs], 3) / 20
        ms_new = median_ms(lambda: [tree.nearest(q) for q in qs], 3) / 20
        winner = "kd-트리" if ms_new < ms_old else "전부 비교"
        rows.append(["kd-트리 대 전부", f"차원 {d}, {n // 10000}만 개", ms_old, ms_new,
                     100.0, winner])
    return rows


def sweep_sizes(d: int, sizes: list[int], rng: np.random.Generator) -> list[list]:
    """축 하나. 크기를 바꾸며 역파일과 전부 비교를 잰다.
    역파일은 재현율@10 을 같이 적는다.
    """
    rows = []
    for n in sizes:
        pts = rng.standard_normal((n, d)).astype(np.float32)
        qs = rng.standard_normal((20, d)).astype(np.float32)
        ivf = IVF(pts, kmeans(pts, max(4, int(np.sqrt(n))), rng, iters=5))
        def brute():
            return [np.argpartition(((pts - q) ** 2).sum(axis=1), 10)[:10] for q in qs]

        ms_old = median_ms(brute, 3) / 20
        ms_new = median_ms(lambda: [ivf.search(q, 2, 10) for q in qs], 3) / 20
        hits = 0
        for q in qs:
            truth = set(np.argpartition(((pts - q) ** 2).sum(axis=1), 10)[:10].tolist())
            hits += len(set(ivf.search(q, 2, 10)[0].tolist()) & truth)
        rows.append(["역파일 대 전부", f"차원 {d}, {n:,} 개", ms_old, ms_new,
                     hits / 200 * 100, "역파일" if ms_new < ms_old else "전부 비교"])
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = sweep_dims(20_000, [2, 8, 32], rng)
    rows += sweep_sizes(64, [1_000, 10_000, 100_000], rng)
    cols = ["짝", "조건", "전통 ms", "새 구조 ms", "재현율", "빠른 쪽"]
    cond = ("질의 20 개의 평균. kd-트리는 잎 16, 역파일은 군집 sqrt(n) 에 nprobe 2. "
            "무작위 정규 분포 점. 세 번 중앙값")
    write_tsv("c17-scale", cols, rows, "bench/c17_scale.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
