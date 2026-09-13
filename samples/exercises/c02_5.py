"""문제 2.5. 성장 인자에 따른 옮긴 원소 수의 합을 정리 2.1 의 식과 대조한다."""
from dsai.dynarray import total_copied


def bound(n: int, factor: float) -> float:
    """정리 2.1 의 상한.

    마지막 용량이 factor * n 을 넘지 않으므로 옮긴 합은 그 아래다.
    """
    return factor * n / (factor - 1)


def test_bound_holds_for_many_factors():
    n = 10**5
    for f in (1.125, 1.5, 2.0, 4.0):
        assert total_copied(n, f, 4) <= bound(n, f)


def test_smaller_factor_copies_more():
    n = 10**5
    assert total_copied(n, 1.125, 4) > total_copied(n, 1.5, 4) > total_copied(n, 2.0, 4)
