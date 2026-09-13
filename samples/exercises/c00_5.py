"""문제 0.5. 원소 크기를 float64 로 바꾸면 갈라지는 자리가 절반이 되는가.

시간을 재는 대신 정리 0.1 로 캐시를 넘는 원소 수를 계산해 비교한다.
실제 시간은 bench/c00_cost.py 의 dtype 을 바꿔 돌리면 된다.
"""
from dsai.cost import moves_random_expected, moves_sequential

CACHE = 8 * 1024 * 1024
LINE = 64


def crossover_elements(elem_bytes: int) -> int:
    """캐시를 정확히 채우는 원소 수. 이보다 크면 정리 0.1 의 조건이 성립한다."""
    return CACHE // elem_bytes


def ratio(n: int, elem_bytes: int) -> float:
    seq = moves_sequential(n, elem_bytes, LINE)
    rnd = moves_random_expected(n, elem_bytes, LINE, CACHE // LINE)
    return rnd / seq


def test_crossover_halves():
    assert crossover_elements(8) * 2 == crossover_elements(4)


def test_ratio_grows_after_crossover():
    n = 4 * crossover_elements(8)
    assert ratio(n, 8) > 1.5
    assert ratio(crossover_elements(8) // 2, 8) == 1.0
