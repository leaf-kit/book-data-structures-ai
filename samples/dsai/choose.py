"""17장. 측정 없이 고르지 않는다. 뒷봉투 계산과 고르는 절차.

고르기는 네 질문이다. 접근이 균일한가, 단위가 스칼라인가, 답이 정확해야 하는가,
구조가 고정인가. 답이 후보를 거르고, 뒷봉투 계산이 후보의 순서를 정하고,
측정이 그 순서를 확인한다. 이 모듈은 그 셋을 함수로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    """후보 구조 하나. 한 번의 연산이 움직이는 바이트와 연산 수, 그리고 성질 둘."""
    name: str
    nbytes: float
    flops: float
    exact: bool = True          # 정확한 답을 주는가
    static: bool = False        # 갱신을 못 하는 구조인가


def envelope_seconds(nbytes: float, flops: float, bandwidth_gbs: float,
                     peak_gflops: float) -> float:
    """정의 17.1. 뒷봉투 시간. 이동과 연산 중 오래 걸리는 쪽. 루프라인의 하한이다."""
    return max(nbytes / (bandwidth_gbs * 1e9), flops / (peak_gflops * 1e9))


def filter_by_questions(cands: list[Candidate], need_exact: bool,
                        need_updates: bool) -> list[Candidate]:
    """알고리즘 17.1 의 첫 단계. 셋째와 넷째 질문이 후보를 거른다."""
    return [c for c in cands
            if (c.exact or not need_exact) and (not c.static or not need_updates)]


def rank(cands: list[Candidate], bandwidth_gbs: float,
         peak_gflops: float) -> list[tuple[Candidate, float]]:
    """둘째 단계. 뒷봉투 시간 순으로 세운다."""
    scored = [(c, envelope_seconds(c.nbytes, c.flops, bandwidth_gbs, peak_gflops))
              for c in cands]
    return sorted(scored, key=lambda t: t[1])


def within_factor(predicted: float, measured: float, factor: float = 3.0) -> bool:
    """셋째 단계의 판정. 실측이 예측의 1/f 배와 f 배 사이면 뒷봉투가 맞은 것이다."""
    return predicted / factor <= measured <= predicted * factor
