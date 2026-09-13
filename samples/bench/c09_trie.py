"""실험 9.1. 최장 일치 토큰화. dict 에 길이를 줄여 가며 묻기와 트라이 한 번 내려가기.

같은 어휘 (바이트 256 개 + BPE 병합 1,000 개) 로 말뭉치 1 MB 를 토큰화한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version, word_counts  # noqa: E402
from bench._util import median_ms, write_tsv  # noqa: E402
from dsai.bpe import merges_to_vocab, train_bpe  # noqa: E402
from dsai.trie import Trie, tokenize_by_dict  # noqa: E402

MERGES = 1000


def main() -> None:
    text = stdlib_text()
    merges = train_bpe(word_counts(text), MERGES)
    vocab = merges_to_vocab(merges)
    by_bytes = {w: t for t, w in vocab.items()}
    max_len = max(len(w) for w in vocab.values())
    trie = Trie()
    for t, w in vocab.items():
        trie.insert(w, t)
    sample = text[:200_000]
    rows = []
    a = tokenize_by_dict(by_bytes, sample, max_len)
    b = trie.tokenize(sample)
    assert a == b
    ms_dict = median_ms(lambda: tokenize_by_dict(by_bytes, sample, max_len), 3)
    ms_trie = median_ms(lambda: trie.tokenize(sample), 3)
    per = 1e6 / len(sample)
    rows.append(["dict 에 길이 줄여 묻기", max_len, ms_dict, ms_dict * per])
    rows.append(["트라이 한 번 내려가기", max_len, ms_trie, ms_trie * per])
    rows.append(["트라이 노드 수", trie.n, 0.0, 0.0])
    cols = ["방식", "최장 항목 또는 노드", "ms", "바이트당 ns"]
    cond = (f"{version()} 표준 라이브러리 소스 200 KB. 어휘 {len(vocab):,}개, "
            f"토큰 {len(b):,}개. 세 번 중앙값")
    write_tsv("c09-trie", cols, rows, "bench/c09_trie.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
