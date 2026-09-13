"""문제 10.6. 길이 정규화 빔 탐색. 점수를 길이로 나눠 견준다."""
import numpy as np

from dsai.beam import beam_step


def beam_search_normalized(step_logp, beam, steps, vocab, eos=None):
    """열이 끝 토큰을 내면 멈춘 채로 두고, 견줄 때 점수를 길이로 나눈다."""
    scores = np.zeros(1)
    tokens = np.zeros(1, dtype=np.int64)
    lengths = np.zeros(1, dtype=np.int64)
    for t in range(steps):
        logp = step_logp(tokens, t)
        parent, token, raw = beam_step(scores, logp, min(beam, logp.size))
        lengths = lengths[parent] + 1
        scores, tokens = raw, token
    return scores / lengths, scores


def test_normalization_keeps_order_when_lengths_equal():
    rng = np.random.default_rng(1)
    table = rng.standard_normal((5, 6, 6))
    table = table - np.log(np.exp(table).sum(axis=2, keepdims=True))

    def m(prev, t):
        return table[t][prev]
    norm, raw = beam_search_normalized(m, 4, 5, 6)
    assert np.argmax(norm) == np.argmax(raw)         # 길이가 다 같으면 순서가 같다
    assert np.allclose(norm * 5, raw)
