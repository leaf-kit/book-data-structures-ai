"""문제 2.8. 블록 크기에 따른 내부 단편화의 기대값과 블록 테이블 크기의 맞바꿈."""
import numpy as np

from dsai.kvcache import BlockPool, PagedKV


def waste_and_table(lengths: np.ndarray, block: int, dim: int = 2) -> tuple[float, int]:
    pool = BlockPool(num_blocks=int(lengths.sum()) // block + len(lengths) + 1,
                     block_size=block, dim=dim)
    reserved, table = 0, 0
    for ln in lengths:
        kv = PagedKV(pool)
        for _ in range(int(ln)):
            kv.append(np.zeros(dim, dtype=np.float32))
        reserved += kv.reserved()
        table += len(kv.table)
    return reserved / lengths.sum() - 1.0, table


def test_waste_grows_table_shrinks():
    rng = np.random.default_rng(1)
    lens = rng.integers(50, 500, 200)
    w16, t16 = waste_and_table(lens, 16)
    w128, t128 = waste_and_table(lens, 128)
    assert w16 < w128            # 작은 블록이 자투리가 적고
    assert t16 > t128            # 테이블은 크다
    # 기대 자투리는 블록의 절반 근처다
    assert abs(w16 * lens.mean() - 7.5) < 3
