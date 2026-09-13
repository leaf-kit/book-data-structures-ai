"""실험 9.4. 상태마다 허용 토큰 집합 만들기. 어휘 전부 훑기와 트라이 위의 오토마톤.

숫자 문법의 상태 넷에서 허용 토큰 집합을 두 방법으로 만들고 시간을 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version, word_counts  # noqa: E402
from bench._util import median_ms, write_tsv  # noqa: E402
from dsai.bpe import merges_to_vocab, train_bpe  # noqa: E402
from dsai.constrained import allowed_by_scan, allowed_by_trie, number_dfa  # noqa: E402
from dsai.trie import Trie  # noqa: E402


def main() -> None:
    text = stdlib_text()
    dfa = number_dfa()
    rows = []
    for n_merges in [1024, 4096]:
        vocab = merges_to_vocab(train_bpe(word_counts(text), n_merges))
        trie = Trie()
        for t, w in vocab.items():
            trie.insert(w, t)
        for s in range(4):
            assert allowed_by_scan(dfa, vocab, s) == allowed_by_trie(dfa, trie, s)
        states = range(4)
        ms_scan = median_ms(lambda: [allowed_by_scan(dfa, vocab, s) for s in states], 3)
        ms_trie = median_ms(lambda: [allowed_by_trie(dfa, trie, s) for s in states], 3)
        allowed = len(allowed_by_trie(dfa, trie, 1))
        total_bytes = sum(len(w) for w in vocab.values())
        rows.append([len(vocab), total_bytes, trie.n, allowed, ms_scan, ms_trie])
    cols = ["어휘 크기", "어휘 바이트 합", "트라이 노드", "상태 1 허용 토큰",
            "전부 훑기 ms", "트라이 ms"]
    cond = (f"{version()} 표준 라이브러리 소스로 학습한 어휘. 문법은 정수 또는 소수, "
            "상태 넷 전부의 집합을 만드는 시간. 세 번 중앙값")
    write_tsv("c09-constrained", cols, rows, "bench/c09_constrained.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
