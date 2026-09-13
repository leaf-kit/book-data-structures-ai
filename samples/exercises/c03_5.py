"""문제 3.5. CSR 과 CSR 의 곱. 결과의 밀도가 어떻게 되는지 본다."""
import numpy as np

from dsai.sparse import CSR, dense_to_csr, random_sparse


def csr_matmul(a: CSR, b: CSR) -> CSR:
    """행 단위로 누적한다.
    결과 행 i 는 a 의 행 i 원소마다 b 의 행을 스케일해 더한 것이다.
    """
    m, k = a.shape
    _, n = b.shape
    ptr = [0]
    cols, vals = [], []
    for i in range(m):
        acc: dict[int, float] = {}
        ca, va = a.row(i)
        for c, v in zip(ca, va):
            cb, vb = b.row(int(c))
            for c2, v2 in zip(cb, vb):
                acc[int(c2)] = acc.get(int(c2), 0.0) + float(v) * float(v2)
        for c2 in sorted(acc):
            cols.append(c2)
            vals.append(acc[c2])
        ptr.append(len(cols))
    return CSR((m, n), np.array(ptr, dtype=np.int64), np.array(cols, dtype=np.int32),
               np.array(vals, dtype=np.float32))


def to_dense(c: CSR) -> np.ndarray:
    out = np.zeros(c.shape, dtype=np.float32)
    for i in range(c.shape[0]):
        cols, vals = c.row(i)
        out[i, cols] = vals
    return out


def test_csr_matmul_matches_dense():
    a = random_sparse(30, 40, 0.1, 1)
    b = random_sparse(40, 20, 0.1, 2)
    c = csr_matmul(dense_to_csr(a), dense_to_csr(b))
    assert np.allclose(to_dense(c), a @ b, atol=1e-4)


def test_product_is_denser():
    a = random_sparse(200, 200, 0.05, 3)
    c = csr_matmul(dense_to_csr(a), dense_to_csr(a))
    assert c.nnz / (200 * 200) > 0.05 * 3
