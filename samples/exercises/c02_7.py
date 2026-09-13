"""문제 2.7. 모델 크기에서 KV 캐시 토큰당 바이트를 구한다.

주어진 메모리에 드는 토큰 수를 센다.
"""
from dsai.kvcache import bytes_per_token


def tokens_that_fit(memory_bytes: int, layers: int, heads: int, head_dim: int,
                    elem_bytes: int) -> int:
    return memory_bytes // bytes_per_token(layers, heads, head_dim, elem_bytes)


def test_forty_layer_model():
    per = bytes_per_token(40, 40, 128, 2)
    assert per == 819_200
    # 40GB 중 가중치 26GB 를 빼고 14GB 를 캐시로 쓰면
    assert tokens_that_fit(14 * 1024**3, 40, 40, 128, 2) == 18_350


def test_grouped_heads_cut_cache():
    # 키 값 헤드를 8 개로 줄이면 토큰당 바이트가 5 분의 1
    assert bytes_per_token(40, 8, 128, 2) * 5 == bytes_per_token(40, 40, 128, 2)
