import numpy as np

from dsai.moe import (dense_forward, experts_touched_expected, moe_forward,
                      moe_intensity, route_top1)
from dsai.quant import dequantize, quant_error, quantize
from dsai.sparse import (coo_to_csr, csr_matvec, csr_transpose, dense_to_coo,
                         dense_to_csr, random_sparse)
from dsai.structured import (bsr_matvec, bytes_2of4, dense_to_bsr, matvec_2of4,
                             pack_2of4, prune_2of4)


def test_csr_roundtrip_and_matvec():
    a = random_sparse(64, 48, 0.1, 1)
    m = dense_to_csr(a)
    x = np.random.default_rng(2).standard_normal(48, dtype=np.float32)
    assert np.allclose(csr_matvec(m, x), a @ x, atol=1e-4)
    assert m.nnz == np.count_nonzero(a)
    cols, vals = m.row(3)
    assert np.array_equal(vals, a[3][a[3] != 0])


def test_csr_handles_empty_rows():
    a = np.zeros((5, 4), dtype=np.float32)
    a[1, 2] = 3.0
    a[4, 0] = 1.0
    m = dense_to_csr(a)
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    assert np.array_equal(csr_matvec(m, x), a @ x)


def test_csr_transpose_is_csc():
    a = random_sparse(20, 30, 0.2, 3)
    t = csr_transpose(dense_to_csr(a))
    x = np.random.default_rng(4).standard_normal(20, dtype=np.float32)
    assert np.allclose(csr_matvec(t, x), a.T @ x, atol=1e-4)


def test_coo_to_csr_sorts():
    c = dense_to_coo(random_sparse(10, 10, 0.3, 5))
    m = coo_to_csr(c)
    assert m.ptr[-1] == c.nnz and np.all(np.diff(m.ptr) >= 0)


def test_bsr_matvec():
    a = random_sparse(64, 64, 0.3, 6)
    b = dense_to_bsr(a, 16)
    x = np.random.default_rng(7).standard_normal(64, dtype=np.float32)
    assert np.allclose(bsr_matvec(b, x), a @ x, atol=1e-4)


def test_2of4_prune_pack_matvec():
    w = np.random.default_rng(8).standard_normal((32, 64), dtype=np.float32)
    p = prune_2of4(w)
    assert np.all(np.count_nonzero(p.reshape(32, 16, 4), axis=2) == 2)
    vals, meta = pack_2of4(p)
    x = np.random.default_rng(9).standard_normal(64, dtype=np.float32)
    assert np.allclose(matvec_2of4(vals, meta, x), p @ x, atol=1e-2)
    assert bytes_2of4(32, 64, 2) == 32 * 64 // 2 * 2 + 32 * 64 // 4


def test_quantize_error_bound():
    w = np.random.default_rng(10).standard_normal((64, 64), dtype=np.float32)
    q8 = quantize(w, 8, 64)
    q4 = quantize(w, 4, 64)
    # 정리 3.2. 오차는 블록마다 델타/2 를 넘지 않는다
    d = np.abs(dequantize(q8) - w).reshape(-1, 64)
    assert np.all(d <= q8.scales[:, None] / 2 + 1e-6)
    assert quant_error(w, q8) < quant_error(w, q4)
    assert q4.nbytes() < q8.nbytes() < w.nbytes


def test_block_scale_isolates_outlier():
    w = np.random.default_rng(11).standard_normal(4096).astype(np.float32)
    w[0] = 100.0
    whole = quantize(w, 8, 4096)
    blocked = quantize(w, 8, 64)
    assert quant_error(w, blocked) < quant_error(w, whole) / 10


def test_moe_matches_per_token():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((10, 8), dtype=np.float32)
    experts = rng.standard_normal((4, 8, 6), dtype=np.float32)
    assign = route_top1(rng.standard_normal((10, 4)))
    out = moe_forward(x, experts, assign)
    for i in range(10):
        assert np.allclose(out[i], x[i] @ experts[assign[i]], atol=1e-5)
    assert dense_forward(x, experts[0]).shape == (10, 6)


def test_experts_touched():
    assert abs(experts_touched_expected(8, 1) - 1.0) < 1e-9
    assert experts_touched_expected(8, 256) > 7.99
    assert moe_intensity(1, 2048, 2048, 8, 4) < moe_intensity(256, 2048, 2048, 8, 4)
