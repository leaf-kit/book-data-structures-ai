"""문제 12.8. 페이지 크기와 이웃 수.
페이지에 벡터와 이웃 M 개가 들어야 한 번의 접근이다.
"""


def fits_in_page(dim: int, dtype_bytes: int, m: int, page: int = 4096,
                 id_bytes: int = 4) -> bool:
    return dim * dtype_bytes + m * id_bytes <= page


def max_neighbors(dim: int, dtype_bytes: int, page: int = 4096,
                  id_bytes: int = 4) -> int:
    return max(0, (page - dim * dtype_bytes) // id_bytes)


def test_page_budget():
    assert fits_in_page(128, 4, 64)                 # 512 + 256 바이트
    assert not fits_in_page(1024, 4, 64)            # 벡터만 4 KB
    assert max_neighbors(768, 4) == 256      # 4096 - 3072 = 1024 바이트, 이웃 256
    assert max_neighbors(768, 1) == 832             # 1 바이트 양자화면 이웃 832 개
