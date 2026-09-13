"""문제 15.5. 인과 마스크가 있는 타일 어텐션. 질의 블록보다 뒤의 키 블록은 건너뛴다."""
import numpy as np

from dsai.attention import attention_tiled
from dsai.kvevict import full_attention_causal


def attention_tiled_causal(q, k, v, block):
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((n, v.shape[1]))
    skipped = 0
    for qs in range(0, n, block):
        qe = min(qs + block, n)
        m_run = np.full(qe - qs, -np.inf)
        l_run = np.zeros(qe - qs)
        o_run = np.zeros((qe - qs, v.shape[1]))
        for ks in range(0, n, block):
            if ks >= qe:                                   # 전부 미래인 블록은 건너뛴다
                skipped += 1
                continue
            ke = min(ks + block, n)
            sc = q[qs:qe] @ k[ks:ke].T * scale
            rows = np.arange(qs, qe)[:, None]
            cols = np.arange(ks, ke)[None, :]
            sc = np.where(cols <= rows, sc, -np.inf)       # 블록 안의 미래는 마스크
            m_new = np.maximum(m_run, sc.max(axis=1))
            alpha = np.exp(m_run - m_new)
            p = np.exp(sc - m_new[:, None])
            l_run = alpha * l_run + p.sum(axis=1)
            o_run = alpha[:, None] * o_run + p @ v[ks:ke]
            m_run = m_new
        out[qs:qe] = o_run / l_run[:, None]
    return out, skipped


def test_causal_tiled_matches_full():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((200, 16)) for _ in range(3))
    ref = full_attention_causal(q, k, v)
    for block in (16, 50, 200):
        out, skipped = attention_tiled_causal(q, k, v, block)
        assert np.allclose(out, ref, atol=1e-9)
    _, skipped = attention_tiled_causal(q, k, v, 16)
    assert skipped == 13 * 12 // 2            # 블록 13 개 중 위 삼각이 사라진다
    assert not np.allclose(attention_tiled(q, k, v, 16), ref)
