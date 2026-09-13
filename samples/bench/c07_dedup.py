"""실험 7.4. 중복 제거 파이프라인. 정확 중복과 근사 중복을 심어 놓고 얼마나 걸러 내는가.

문서 만 개 중 정확 중복 천 개, 몇 단어만 바꾼 근사 중복 천 개를 섞는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.dedup import DedupPipeline  # noqa: E402

N_BASE = 8000
N_EXACT = 1000
N_NEAR = 1000
WORDS = 60


def corpus(rng: np.random.Generator) -> tuple[list[str], set[int]]:
    vocab = [f"w{i}" for i in range(5000)]
    base = [" ".join(rng.choice(vocab, WORDS)) for _ in range(N_BASE)]
    docs = list(base)
    dup_ids: set[int] = set()
    for _ in range(N_EXACT):
        docs.append(base[rng.integers(N_BASE)].upper())      # 대소문자만 다른 정확 중복
        dup_ids.add(len(docs) - 1)
    for _ in range(N_NEAR):
        words = base[rng.integers(N_BASE)].split()
        for j in rng.choice(WORDS, 3, replace=False):      # 단어 셋만 바꾼 근사 중복
            words[j] = vocab[rng.integers(5000)]
        docs.append(" ".join(words))
        dup_ids.add(len(docs) - 1)
    return docs, dup_ids


def main() -> None:
    rng = np.random.default_rng(SEED)
    docs, dup_ids = corpus(rng)
    order = rng.permutation(len(docs))

    def run() -> DedupPipeline:
        p = DedupPipeline(expected=len(docs), k_sig=64, bands=16, threshold=0.7)
        for i in order:
            p.offer(int(i), docs[i])
        return p

    t = median_ms(run, 3)
    p = run()
    kept = set(p.stats.kept)
    # 원본이 뒤에 오고 중복이 앞에 온 경우가 있어,
    # 남은 것 중 심은 중복의 수가 아니라 걸러진 총수로 본다
    removed = len(docs) - len(kept)
    rows = [
        ["문서 수", len(docs)], ["심은 정확 중복", N_EXACT], ["심은 근사 중복", N_NEAR],
        ["걸러 낸 정확 중복", p.stats.exact_dups],
        ["걸러 낸 근사 중복", p.stats.near_dups],
        ["남은 문서", len(kept)],
        ["후보 비교 횟수", p.stats.near_candidates],
        ["걸러 낸 총수", removed], ["시간 ms", round(t)],
        ["블룸 필터 KB", p.bloom.nbytes() // 1000],
    ]
    cond = (f"문서 {len(docs):,}개, 단어 {WORDS}개씩. "
            "서명 64, 띠 16, 문턱 0.7. 세 번 중앙값")
    write_tsv("c07-dedup", ["항목", "값"], rows, "bench/c07_dedup.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
