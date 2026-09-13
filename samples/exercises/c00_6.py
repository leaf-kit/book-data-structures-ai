"""문제 0.6. 실험 0.1 의 각 행에서 두 순서가 건드리는 라인 수를 센다."""
from dsai.cost import blocks_touched, random_index, sequential_index

ELEM, LINE = 4, 64


def line_ratio(n: int, seed: int = 1) -> float:
    seq = blocks_touched(sequential_index(n), ELEM, LINE)
    rnd = blocks_touched(random_index(n, seed), ELEM, LINE)
    return rnd / seq


def table(sizes: list[int]) -> list[tuple[int, float]]:
    return [(n, line_ratio(n)) for n in sizes]


def test_line_ratio_near_line_width():
    # 라인 하나가 원소 16개이므로 무작위 순서의 라인 비율은 16 근처로 간다
    r = line_ratio(100_000)
    assert 12 < r <= 16


def test_table_shape():
    t = table([1000, 10_000])
    assert len(t) == 2 and all(r > 1 for _, r in t)
