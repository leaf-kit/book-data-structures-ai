"""실험 14.2. n-그램 표와 무한 n-그램. 다음 바이트 맞히기와 메모리.

앞 90 퍼센트로 만들고 뒤 10 퍼센트에서 다음 바이트를 맞힌다.
n 을 정한 표와 접미사 배열의 백오프.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._corpus import stdlib_text, version  # noqa: E402
from bench._util import median_ms, write_tsv  # noqa: E402
from dsai.ngram import InfiniGram, NgramTable  # noqa: E402
from dsai.suffix import build_suffix_array  # noqa: E402

N = 400_000
Q = 2000


def main() -> None:
    text = np.frombuffer(stdlib_text(N), dtype=np.uint8).astype(np.int64)
    split = int(N * 0.9)
    train, test = text[:split], text[split:]
    rng = np.random.default_rng(0)
    qpos = rng.integers(64, len(test) - 1, Q)
    rows = []
    for n in [3, 5, 8]:
        tbl = NgramTable(train, n)
        ok = sum(tbl.predict(test[p - n + 1:p].tolist()) == test[p] for p in qpos)
        miss = sum(tbl.predict(test[p - n + 1:p].tolist()) is None for p in qpos)
        ms = median_ms(
            lambda: [tbl.predict(test[p - n + 1:p].tolist()) for p in qpos[:200]], 3)
        rows.append([f"{n}-그램 표", n, ok / Q * 100, miss / Q * 100,
                     tbl.nbytes() / 1e6, ms / 200 * 1e3])
    sa = build_suffix_array(train)
    ig = InfiniGram(train, sa)
    preds = [ig.predict(test[p - 64:p].tolist()) for p in qpos]
    ok = sum(t == test[p] for (t, _), p in zip(preds, qpos))
    ks = [k for _, k in preds]
    ms = median_ms(lambda: [ig.predict(test[p - 64:p].tolist()) for p in qpos[:200]], 3)
    rows.append(["무한 n-그램 (접미사 배열)", float(np.mean(ks)), ok / Q * 100, 0.0,
                 (sa.nbytes + train.nbytes) / 1e6, ms / 200 * 1e3])
    cols = ["방법", "문맥", "맞힘 퍼센트", "없음 퍼센트", "MB", "예측 us"]
    cond = (f"{version()} 표준 라이브러리 소스 {N:,} 바이트. "
            f"앞 90 퍼센트로 만들고 뒤에서 {Q:,}개를 맞힘. "
            "무한 n-그램의 문맥 길이는 백오프가 쓴 길이의 평균")
    write_tsv("c14-ngram", cols, rows, "bench/c14_ngram.py", cond, 3)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
