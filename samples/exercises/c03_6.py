"""문제 3.6. 블록 크기에 따른 BSR 의 인덱스 바이트와 라인 수."""
import numpy as np

from dsai.structured import dense_to_bsr


def bsr_costs(a: np.ndarray, block: int, line: int = 64) -> tuple[int, int]:
    b = dense_to_bsr(a, block)
    index_bytes = b.cols.nbytes + b.ptr.nbytes
    lines_for_x = len(b.cols) * max(1, (block * 4 + line - 1) // line)
    return index_bytes, lines_for_x


def test_bigger_block_fewer_index_bytes():
    rng = np.random.default_rng(1)
    keep = rng.random((64, 64)) < 0.5
    a = np.kron(keep, np.ones((8, 8), dtype=np.float32))   # 512 x 512
    i8, _ = bsr_costs(a, 8)
    i16, _ = bsr_costs(a, 16)
    i64, _ = bsr_costs(a, 64)
    assert i8 > i16 > i64
