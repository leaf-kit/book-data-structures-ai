"""문제 1.6. 간격 훑기의 원소당 이동량을 정리 1.1 로 예측하고 실측 열과 비교한다."""
from dsai.tensor import lines_for_traversal

N = 64 * 1024 * 1024


def predicted_lines_per_element(k: int, elem_bytes: int = 4, line: int = 64) -> float:
    touched = (N + k - 1) // k
    return lines_for_traversal(touched, k * elem_bytes, line) / touched


def test_prediction_saturates_at_line_width():
    ratios = [predicted_lines_per_element(k) for k in (1, 2, 4, 8, 16, 32, 64)]
    # 16 까지는 두 배씩 오르고, 그 뒤로는 1 에 머문다
    assert ratios[0] == 1 / 16
    assert abs(ratios[4] - 1.0) < 1e-9
    assert ratios[5] == ratios[6] == 1.0
