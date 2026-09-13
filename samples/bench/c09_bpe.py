"""실험 9.2. 병합 수와 압축률, 그리고 최장 일치와 병합 순서의 차이.

병합 수를 늘리며 토큰당 바이트를 재고,
같은 어휘에서 두 부호화가 내는 토큰 열이 얼마나 다른지 센다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version, word_counts  # noqa: E402
from bench._util import write_tsv  # noqa: E402
from dsai.bpe import encode_by_merges, merges_to_vocab, train_bpe  # noqa: E402
from dsai.trie import Trie  # noqa: E402


def main() -> None:
    text = stdlib_text()
    words = word_counts(text)
    sample_words = [b" " + w for w in text[:200_000].split(b" ") if w]
    rows = []
    for n_merges in [256, 1024, 4096]:
        t0 = time.perf_counter()
        merges = train_bpe(words, n_merges)
        ms_train = (time.perf_counter() - t0) * 1e3
        vocab = merges_to_vocab(merges)
        rank = {m: i for i, m in enumerate(merges)}
        trie = Trie()
        for t, w in vocab.items():
            trie.insert(w, t)
        n_bpe = sum(len(encode_by_merges(w, rank)) for w in sample_words)
        n_lm = sum(len(trie.tokenize(w)) for w in sample_words)
        differ = sum(encode_by_merges(w, rank) != trie.tokenize(w)
                     for w in sample_words)
        nbytes = sum(len(w) for w in sample_words)
        rows.append([n_merges, ms_train / 1e3, nbytes / n_bpe, nbytes / n_lm,
                     differ / len(sample_words) * 100])
    cols = ["병합 수", "학습 초", "병합 순서 바이트", "최장 일치 바이트",
            "다른 단어 퍼센트"]
    cond = (f"{version()} 표준 라이브러리 소스 1 MB 에서 상위 8,000 단어로 학습, "
            "앞 200 KB 로 부호화")
    write_tsv("c09-bpe", cols, rows, "bench/c09_bpe.py", cond, 1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
