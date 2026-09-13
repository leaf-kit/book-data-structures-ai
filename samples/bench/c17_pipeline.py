"""실험 17.2. dsai 를 처음부터 끝까지. 단계마다 시간과 최대 메모리.

표준 라이브러리 소스 400 KB 를 2 KB 문서로 자르고 3 할을 복제한 뒤 여섯 단계를 지난다.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version  # noqa: E402
from bench._util import SEED, write_tsv  # noqa: E402
from dsai.pipeline import (count_top_bigram, dedup_docs, embed_docs,  # noqa: E402
                           index_recall, radix_hit_rate, split_docs, tokenize_docs)

N = 400_000


def timed(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    ms = (time.perf_counter() - t0) * 1e3
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, ms, peak / 2**20


def main() -> None:
    rng = np.random.default_rng(SEED)
    text = stdlib_text(N)
    rows = []
    docs, ms, mb = timed(lambda: split_docs(text, 2048, 0.3, rng))
    rows.append(["자르기와 복제", "배열", f"문서 {len(docs)}", ms, mb])
    kept, ms, mb = timed(lambda: dedup_docs(docs))
    rows.append(["중복 제거", "블룸, MinHash, LSH (7장)", f"문서 {len(kept)}", ms, mb])
    (tokens, vocab), ms, mb = timed(lambda: tokenize_docs(kept, 500))
    ntok = sum(len(t) for t in tokens)
    rows.append(["토큰화", "BPE (9장)", f"토큰 {ntok // 10000}만, 어휘 {vocab}",
                 ms, mb])
    (big, cnt), ms, mb = timed(lambda: count_top_bigram(tokens))
    rows.append(["세기", "접미사 배열 (14장)", f"최다 쌍 {cnt} 번", ms, mb])
    x, ms, mb = timed(lambda: embed_docs(tokens, 64))
    rows.append(["임베딩", "해싱 트릭 (6장)", "64차원", ms, mb])
    (rec, seen), ms, mb = timed(lambda: index_recall(x, 5, 16, 2, rng))
    rows.append(["인덱스와 질의", "역파일 (13장)", f"재현율 {rec * 100:.0f} 퍼센트",
                 ms, mb])
    hit, ms, mb = timed(lambda: radix_hit_rate(tokens, 32, 100_000))
    rows.append(["서빙 캐시", "기수 트리 (9장)", f"적중 {hit * 100:.0f} 퍼센트",
                 ms, mb])
    cols = ["단계", "쓴 구조", "출력", "ms", "최대 MB"]
    cond = (f"{version()} 표준 라이브러리 소스 {N:,} 바이트. "
            "2 KB 문서로 자르고 3 할을 복제. "
            "BPE 병합 500. 메모리는 tracemalloc 의 단계별 최대. 한 번 측정")
    write_tsv("c17-pipeline", cols, rows, "bench/c17_pipeline.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
