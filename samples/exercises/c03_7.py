"""문제 3.7. 이중 양자화. 스케일을 다시 8비트로 적으면 원소당 바이트가 얼마나 주는가."""
import numpy as np

from dsai.quant import QTensor, quant_error, quantize


def double_quantize(x: np.ndarray, bits: int,
                    block: int) -> tuple[QTensor, np.ndarray, float]:
    q = quantize(x, bits, block)
    smax = float(q.scales.max())
    codes8 = np.rint(q.scales / smax * 127).astype(np.uint8)
    return q, codes8, smax


def dequant_double(q: QTensor, codes8: np.ndarray, smax: float) -> np.ndarray:
    scales = codes8.astype(np.float32) / 127 * smax
    return (q.codes.astype(np.float32) * scales[:, None]).reshape(q.shape)


def bytes_per_elem(bits: int, block: int, scale_bytes: int) -> float:
    return bits / 8 + scale_bytes / block


def test_double_quant_saves_bytes_costs_error():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((256, 256)).astype(np.float32) * 0.02
    q, c8, smax = double_quantize(w, 4, 64)
    single = quant_error(w, q)
    d = dequant_double(q, c8, smax) - w
    double = float((d * d).sum() / (w * w).sum())
    assert bytes_per_elem(4, 64, 1) < bytes_per_elem(4, 64, 4)
    assert double >= single
    assert double < single * 3
