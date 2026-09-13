"""문제 1.7. 블록 전치의 블록 크기를 캐시 용량에서 거꾸로 구한다."""
from dsai.tiling import tile_bytes


def largest_block(cache_bytes: int, itemsize: int) -> int:
    """입력 타일과 출력 타일이 함께 캐시에 드는 가장 큰 2의 거듭제곱 블록."""
    b = 1
    while tile_bytes(b * 2, itemsize) <= cache_bytes:
        b *= 2
    return b


def test_block_for_l2_and_l3():
    assert largest_block(256 * 1024, 4) == 128      # L2 256KB
    assert largest_block(8 * 1024 * 1024, 4) == 1024  # L3 8MB
    assert tile_bytes(64, 4) == 32 * 1024
