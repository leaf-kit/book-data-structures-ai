"""실험 5.2. 입력 n 개의 기울기. 역방향 한 번과 전진 방향 n 번.

f(x) = sum(tanh(W x)) 의 기울기를 잰다. 테이프는 한 번 앞으로 가고 한 번 뒤로 온다.
방향 미분은 입력마다 한 번씩 앞으로 간다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._util import SEED, median_ms, write_tsv  # noqa: E402
from dsai.tape import Tape, forward_mode_grad  # noqa: E402

REPEAT = 5
SIZES = [16, 64, 256, 1024]
HIDDEN = 256


def make_f(w: np.ndarray):
    def f(x: np.ndarray) -> float:
        return float(np.tanh(w @ x).sum())
    return f


def reverse_grad(w: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int]:
    """행렬 곱을 원소별 곱과 합으로 풀어 테이프에 적는다. 테이프 항목 수도 돌려준다."""
    t = Tape()
    xi = t.var(x)
    rows = []
    for r in w:
        wr = t.var(r)
        rows.append(t.sum(t.mul(wr, xi)))
    # 행마다 tanh 를 취해 더한다
    acc = t.tanh(rows[0])
    for h in rows[1:]:
        acc = t.add(acc, t.tanh(h))
    grads = t.backward(acc)
    return grads[xi], len(t.entries)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in SIZES:
        w = rng.standard_normal((HIDDEN, n)) * 0.1
        x = rng.standard_normal(n)
        f = make_f(w)
        t_rev = median_ms(lambda: reverse_grad(w, x), REPEAT)
        _, entries = reverse_grad(w, x)

        def all_forward() -> np.ndarray:
            g = np.empty(n)
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                g[i] = forward_mode_grad(f, x, e)
            return g
        t_fwd = median_ms(all_forward, REPEAT)
        rows.append([n, entries, t_rev, t_fwd, t_fwd / t_rev])
    cols = ["입력 n", "테이프 항목", "역방향 ms", "전진 n번 ms", "비율"]
    cond = (f"f(x) = sum(tanh(W x)), W 는 {HIDDEN} x n. "
            "역방향은 테이프 한 번, 전진은 방향 미분 n 번")
    write_tsv("c05-tape", cols, rows, "bench/c05_tape.py", cond, REPEAT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
