"""5장. 테이프. 역방향 자동 미분의 스택.

앞으로 계산하며 연산 하나마다 (입력, 국소 미분) 을 테이프 끝에 적는다.
뒤로 갈 때 테이프를 끝에서부터 읽어 미분을 입력 쪽으로 흘린다.
추가 전용 배열을 거꾸로 읽는 것이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Entry:
    """테이프 항목 하나. 출력 변수 번호, 입력 변수 번호들, 입력마다의 국소 미분."""

    out: int
    inputs: tuple[int, ...]
    local: tuple[np.ndarray, ...]


@dataclass
class Tape:
    """정의 5.2. 추가 전용 배열. 변수는 번호로, 값은 vals 에, 연산은 entries 에."""

    vals: list[np.ndarray] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)

    def var(self, x: np.ndarray) -> int:
        self.vals.append(np.asarray(x, dtype=np.float64))
        return len(self.vals) - 1

    def record(self, value: np.ndarray, inputs: tuple[int, ...],
               local: tuple[np.ndarray, ...]) -> int:
        out = self.var(value)
        self.entries.append(Entry(out, inputs, local))
        return out

    # 원시 연산 셋. 벡터에 원소별로 작용한다.
    def mul(self, a: int, b: int) -> int:
        va, vb = self.vals[a], self.vals[b]
        return self.record(va * vb, (a, b), (vb, va))

    def add(self, a: int, b: int) -> int:
        va, vb = self.vals[a], self.vals[b]
        one = np.ones_like(va)
        return self.record(va + vb, (a, b), (one, one))

    def tanh(self, a: int) -> int:
        t = np.tanh(self.vals[a])
        return self.record(t, (a,), (1.0 - t * t,))

    def sum(self, a: int) -> int:
        va = self.vals[a]
        return self.record(np.array(va.sum()), (a,), (np.ones_like(va),))

    def backward(self, out: int) -> list[np.ndarray]:
        """알고리즘 5.2. 테이프를 끝에서부터 읽으며 수반 변수를 쌓는다."""
        adj = [np.zeros_like(v) for v in self.vals]
        adj[out] = np.ones_like(self.vals[out])
        for e in reversed(self.entries):
            g = adj[e.out]
            for i, d in zip(e.inputs, e.local):
                contrib = g * d
                if contrib.shape != adj[i].shape:
                    contrib = contrib.reshape(adj[i].shape)
                adj[i] = adj[i] + contrib
        return adj

    def nbytes(self) -> int:
        return sum(v.nbytes for v in self.vals) + sum(
            sum(d.nbytes for d in e.local) for e in self.entries)


def forward_mode_grad(f, x: np.ndarray, eps_dir: np.ndarray) -> float:
    """방향 미분 하나. 입력 n 개의 기울기를 다 얻으려면 n 번 부른다."""
    h = 1e-6
    return float((f(x + h * eps_dir) - f(x - h * eps_dir)) / (2 * h))
