"""문제 0.7. 원소별 곱의 연산 수와 바이트 수, 그리고 밀도."""
from dsai.roofline import flops_dot, intensity


def flops_elementwise(n: int) -> int:
    return n


def bytes_elementwise(n: int, elem_bytes: int) -> int:
    # 읽기 둘, 쓰기 하나
    return 3 * n * elem_bytes


def test_elementwise_intensity_below_dot():
    n = 1 << 20
    i_ew = intensity(flops_elementwise(n), bytes_elementwise(n, 4))
    i_dot = intensity(flops_dot(n), 2 * n * 4)
    assert abs(i_ew - 1 / 12) < 1e-9
    assert i_ew < i_dot
